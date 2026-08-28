#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,sys
from dataclasses import dataclass
from pathlib import Path
import cv2,numpy as np
from ultralytics import YOLO

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_yolo_pixel_benchmark as base

@dataclass(frozen=True)
class Sample:
    x:np.ndarray
    crop:np.ndarray
    reason:str

def args():
    p=argparse.ArgumentParser()
    for name in ("train-frames","validation-frames","output","train-source-url","validation-source-url"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--model",default="yolo11n.pt")
    p.add_argument("--seg-model",default="yolo11n-seg.pt")
    p.add_argument("--pose-model",default="yolo11n-pose.pt")
    p.add_argument("--confidence",type=float,default=.12)
    p.add_argument("--seg-confidence",type=float,default=.10)
    p.add_argument("--pose-confidence",type=float,default=.10)
    p.add_argument("--min-track-hits",type=int,default=2)
    return p.parse_args()

def crop(frame,d,margin=.18):
    h,w=frame.shape[:2]
    x1,y1,x2,y2=base.expand_box(d,w,h,margin)
    c=frame[y1:y2,x1:x2]
    return None if c.size==0 or min(c.shape[:2])<8 else c

def unit(v):
    v=v.astype(np.float32)
    n=float(np.linalg.norm(v))
    return v/(n if n>1e-6 else 1.0)

def gradient_descriptor(gray):
    g=gray.astype(np.float32)/255.0
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    ang=(np.degrees(np.arctan2(gy,gx))+180.0)%180.0
    bins=np.clip(np.floor(ang/20.0).astype(np.int32),0,8)
    desc=[]
    for y1 in range(0,128,8):
        for x1 in range(0,64,8):
            b=bins[y1:y1+8,x1:x1+8].reshape(-1)
            m=mag[y1:y1+8,x1:x1+8].reshape(-1)
            hist=np.bincount(b,weights=m,minlength=9).astype(np.float32)[:9]
            desc.append(unit(hist))
    return unit(np.concatenate(desc))

def feat(frame,d,circle):
    c=crop(frame,d)
    if c is None:return None
    r=cv2.resize(c,(64,128),interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    grad=gradient_descriptor(g)
    hsv=cv2.cvtColor(r,cv2.COLOR_BGR2HSV)
    hist=unit(cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).reshape(-1).astype(np.float32))
    edge=float((cv2.Canny(g,70,160)>0).mean())
    h,w=frame.shape[:2]
    sd,sr=1.0,0.0
    if circle:
        sx,sy,srr=circle
        sd=min(3.0,math.hypot(d.cx-sx,d.cy-sy)/max(srr,1.0))/3.0
        sr=min(1.0,srr/max(h,1))
    aspect=min(5.0,max(.0,d.aspect_hw))/5.0
    aux=unit(np.array([d.cx/w,d.cy/h,d.width/w,d.height/h,aspect,d.confidence,d.seg_overlap,d.pose_score,sd,sr,edge],np.float32))
    # Equalize feature families before cosine matching so the 1152-D gradient
    # block cannot drown out geometry/context simply because it is larger.
    descriptor=np.concatenate([math.sqrt(.65)*grad,math.sqrt(.15)*hist,math.sqrt(.20)*aux]).astype(np.float32)
    return unit(descriptor),c

def hard_negative_reason(d,w,h,circle):
    if base.scope_ring_reject(d,circle):return "scope_ring"
    if base.weapon_zone_risk(d,w,h) and d.pose_score<.22 and d.seg_overlap<.46:return "weapon_zone"
    if d.aspect_hw<.70 and d.pose_score<.20 and d.seg_overlap<.40:return "wide_nonhuman"
    if d.cy/h>.72 and d.pose_score<.18:return "lower_fps"
    return None

def evidence_from_tiled(frame,tiled,segmenter,poser,a):
    out=[]
    for d0 in tiled:
        seg=base.segmentation_overlap(segmenter,frame,d0,a.seg_confidence)
        if seg<.14:continue
        pose=base.pose_evidence(poser,frame,d0,a.pose_confidence)
        out.append(base.Detection(d0.x1,d0.y1,d0.x2,d0.y2,d0.confidence,d0.source,seg,pose,1,0.))
    return out

def mine_negatives(folder,detector,segmenter,poser,a):
    samples=[]; reasons={}; st={"frames":0,"candidates":0,"hard_negatives":0}
    for p in sorted(folder.glob("frame_*.jpg")):
        frame=cv2.imread(str(p))
        if frame is None:continue
        st["frames"]+=1; h,w=frame.shape[:2]; circle=base.detect_scope_circle(frame)
        tiled=base.detect_tiled(detector,frame,a.confidence); st["candidates"]+=len(tiled)
        for d in evidence_from_tiled(frame,tiled,segmenter,poser,a):
            reason=hard_negative_reason(d,w,h,circle)
            if reason is None:continue
            fv=feat(frame,d,circle)
            if fv is None:continue
            samples.append(Sample(fv[0],fv[1],reason)); reasons[reason]=reasons.get(reason,0)+1
    st["hard_negatives"]=len(samples); st["reasons"]=reasons
    return samples,st

def build_memory(samples):
    if len(samples)<12:raise SystemExit(f"insufficient hard-negative memory samples: {len(samples)}")
    bank=np.stack([unit(s.x) for s in samples]).astype(np.float32)
    sim=bank@bank.T
    np.fill_diagonal(sim,-1.0)
    nn=sim.max(axis=1)
    # Conservative one-class threshold: candidate must look at least as close
    # to a known hard-negative as a typical hard-negative looks to another.
    threshold=max(.84,min(.965,float(np.percentile(nn,35)-.015)))
    return {"bank":bank,"threshold":threshold,"nearest_neighbor":nn.astype(np.float32)}, {
        "samples":len(samples),"feature_dimensions":int(bank.shape[1]),"similarity_threshold":threshold,
        "nn_similarity_min":float(nn.min()),"nn_similarity_median":float(np.median(nn)),"nn_similarity_max":float(nn.max())}

def save_memory(mem,path):
    np.savez_compressed(str(path),bank=mem["bank"],threshold=np.array([mem["threshold"]],np.float32),nearest_neighbor=mem["nearest_neighbor"])

def negative_similarity(mem,x):
    return float(np.max(mem["bank"]@unit(x)))

def bank_sheet(samples,out):
    chosen=samples[:36]
    if not chosen:return
    tw,th,cols=180,140,6; rows=math.ceil(len(chosen)/cols)
    canvas=np.zeros((rows*th,cols*tw,3),np.uint8)
    for i,s in enumerate(chosen):
        scale=min((tw-8)/s.crop.shape[1],(th-30)/s.crop.shape[0])
        im=cv2.resize(s.crop,(max(1,int(s.crop.shape[1]*scale)),max(1,int(s.crop.shape[0]*scale))))
        rr,cc=divmod(i,cols); y0,x0=rr*th,cc*tw; ix=x0+(tw-im.shape[1])//2
        canvas[y0+22:y0+22+im.shape[0],ix:ix+im.shape[1]]=im
        cv2.putText(canvas,("NEG "+s.reason)[:22],(x0+4,y0+16),cv2.FONT_HERSHEY_SIMPLEX,.38,(60,80,230),1,cv2.LINE_AA)
    cv2.imwrite(str(out),canvas,[int(cv2.IMWRITE_JPEG_QUALITY),91])

def contact_sheet(images,out):
    if not images:return
    ims=[]
    for im in images[:30]:
        scale=520/im.shape[1]; ims.append(cv2.resize(im,(520,max(1,int(im.shape[0]*scale)))))
    rh=max(i.shape[0] for i in ims); rows=math.ceil(len(ims)/2)
    canvas=np.zeros((rows*rh,1040,3),np.uint8)
    for i,im in enumerate(ims):
        r,c=divmod(i,2); canvas[r*rh:r*rh+im.shape[0],c*520:(c+1)*520]=im
    cv2.imwrite(str(out),canvas,[int(cv2.IMWRITE_JPEG_QUALITY),90])

def draw(frame,tiled,final,rejected,name):
    c=frame.copy()
    for d in tiled:cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(35,50,245),1)
    for d,s in rejected:
        cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(0,170,255),2)
        cv2.putText(c,f"MEM {s:.2f}",(int(d.x1),max(14,int(d.y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,170,255),1)
    for d,s in final:
        cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(70,220,70),2)
        cv2.putText(c,f"V4 {s:.2f}",(int(d.x1),max(14,int(d.y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(70,220,70),1)
    cv2.rectangle(c,(0,0),(c.shape[1],28),(0,0,0),-1)
    cv2.putText(c,f"{name} cand={len(tiled)} v4={len(final)} memoryReject={len(rejected)}",(7,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
    return c

def validate(folder,detector,segmenter,poser,mem,a):
    tracks=[]; tid=1; rows=[]; visuals=[]
    st={"frames":0,"tiled":0,"seg_pose":0,"v2_accept":0,"memory_reject":0,"v4_accept":0,"v4_positive_frames":0}
    for p in sorted(folder.glob("frame_*.jpg")):
        frame=cv2.imread(str(p))
        if frame is None:continue
        st["frames"]+=1; h,w=frame.shape[:2]; circle=base.detect_scope_circle(frame)
        tiled=base.detect_tiled(detector,frame,a.confidence)
        ev=evidence_from_tiled(frame,tiled,segmenter,poser,a)
        tracks,tracked,tid=base.update_tracks(tracks,ev,w,h,tid)
        final=[]; rejected=[]; v2=0
        for d in tracked:
            if not base.final_accept(d,w,h,a.min_track_hits,circle):continue
            v2+=1; fv=feat(frame,d,circle)
            if fv is None:continue
            sim=negative_similarity(mem,fv[0])
            # Never let the memory override a strong pose observation. It only
            # suppresses candidates that resemble known FPS hard negatives.
            if sim>=mem["threshold"] and d.pose_score<.58:rejected.append((d,sim))
            else:final.append((d,sim))
        st["tiled"]+=len(tiled); st["seg_pose"]+=len(ev); st["v2_accept"]+=v2
        st["memory_reject"]+=len(rejected); st["v4_accept"]+=len(final); st["v4_positive_frames"]+=int(bool(final))
        rows.append({"frame":p.name,"width":w,"height":h,"tiled":len(tiled),"seg_pose":len(ev),"v2_accept":v2,
                     "memory_reject":len(rejected),"v4_accept":len(final),"threshold":mem["threshold"]})
        visuals.append(draw(frame,tiled,final,rejected,p.name))
    return st,rows,visuals

def main():
    a=args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    detector,segmenter,poser=YOLO(a.model),YOLO(a.seg_model),YOLO(a.pose_model)
    negatives,mining=mine_negatives(Path(a.train_frames),detector,segmenter,poser,a)
    bank_sheet(negatives,out/"hard_negative_memory_bank.jpg")
    mem,mem_stats=build_memory(negatives); save_memory(mem,out/"fps_hard_negative_memory.npz")
    vst,rows,visuals=validate(Path(a.validation_frames),detector,segmenter,poser,mem,a)
    if not rows:raise SystemExit("no validation rows produced")
    with (out/"validation_detections_v4.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    contact_sheet(visuals,out/"validation_contact_sheet_v4.jpg")
    report={"benchmark":"LiteView offline negative-only FPS memory validation v4","training_source_url":a.train_source_url,
      "validation_source_url":a.validation_source_url,"mining":mining,"memory":mem_stats,"validation":vst,
      "important_limit":"Negative-only memory is a false-positive suppressor, not an accuracy claim; visually audit the independent validation contact sheet.",
      "scope_limit":"Offline replay analysis only; no team/enemy identity, aiming, hidden-position inference, live coordinates, or anti-cheat behavior."}
    (out/"benchmark_domain_v4.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=f"""## LiteView offline negative-only FPS memory v4

- Training frames: **{mining['frames']}**
- Hard negatives stored: **{mining['hard_negatives']}**
- Hard-negative reasons: **{json.dumps(mining['reasons'],ensure_ascii=False)}**
- Feature dimensions: **{mem_stats['feature_dimensions']}**
- Memory similarity threshold: **{mem_stats['similarity_threshold']:.3f}**
- Independent validation frames: **{vst['frames']}**
- V2 accepted boxes: **{vst['v2_accept']}**
- Memory-suppressed boxes: **{vst['memory_reject']}**
- V4 boxes remaining: **{vst['v4_accept']}** on **{vst['v4_positive_frames']}/{vst['frames']}** frames

No pseudo-positive labels are used. This is not an accuracy claim; visually audit the independent validation sheet.
"""
    (out/"summary_domain_v4.md").write_text(summary,encoding="utf-8"); print(summary); return 0
if __name__=="__main__":raise SystemExit(main())
