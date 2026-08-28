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
    y:int
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

def feat(frame,d,circle):
    c=crop(frame,d)
    if c is None:return None
    r=cv2.resize(c,(64,128),interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    hog=cv2.HOGDescriptor((64,128),(16,16),(8,8),(8,8),9).compute(g).reshape(-1)
    hsv=cv2.cvtColor(r,cv2.COLOR_BGR2HSV)
    hist=cv2.normalize(cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]),None).reshape(-1)
    edge=np.array([(cv2.Canny(g,70,160)>0).mean()],np.float32)
    h,w=frame.shape[:2]
    sd,sr=2.,0.
    if circle:
        sx,sy,srr=circle
        sd=math.hypot(d.cx-sx,d.cy-sy)/max(srr,1.)
        sr=srr/max(h,1)
    geo=np.array([d.cx/w,d.cy/h,d.width/w,d.height/h,d.aspect_hw,d.confidence,d.seg_overlap,d.pose_score,sd,sr],np.float32)
    return np.concatenate([hog.astype(np.float32),hist.astype(np.float32),edge,geo]),c

def label(d,w,h,circle):
    if base.scope_ring_reject(d,circle):return 0,"scope_ring"
    if base.weapon_zone_risk(d,w,h) and d.pose_score<.22 and d.seg_overlap<.46:return 0,"weapon_zone"
    if d.aspect_hw<.70 and d.pose_score<.20 and d.seg_overlap<.40:return 0,"wide_nonhuman"
    if d.cy/h>.72 and d.pose_score<.18:return 0,"lower_fps"
    hr=d.height/h
    if (not base.weapon_zone_risk(d,w,h) and not base.scope_ring_reject(d,circle)
        and .035<=hr<=.68 and .72<=d.aspect_hw<=4.8 and d.seg_overlap>=.36 and d.pose_score>=.38):
        return 1,"human"
    return None

def evidence(frame,detector,segmenter,poser,a):
    out=[]
    for d0 in base.detect_tiled(detector,frame,a.confidence):
        seg=base.segmentation_overlap(segmenter,frame,d0,a.seg_confidence)
        if seg<.14:continue
        pose=base.pose_evidence(poser,frame,d0,a.pose_confidence)
        out.append(base.Detection(d0.x1,d0.y1,d0.x2,d0.y2,d0.confidence,d0.source,seg,pose,1,0.))
    return out

def mine(folder,detector,segmenter,poser,a):
    samples=[]; st={"frames":0,"candidates":0,"positive":0,"negative":0}
    for p in sorted(folder.glob("frame_*.jpg")):
        frame=cv2.imread(str(p))
        if frame is None:continue
        st["frames"]+=1; h,w=frame.shape[:2]; circle=base.detect_scope_circle(frame)
        ds=base.detect_tiled(detector,frame,a.confidence); st["candidates"]+=len(ds)
        for d0 in ds:
            seg=base.segmentation_overlap(segmenter,frame,d0,a.seg_confidence)
            if seg<.14:continue
            pose=base.pose_evidence(poser,frame,d0,a.pose_confidence)
            d=base.Detection(d0.x1,d0.y1,d0.x2,d0.y2,d0.confidence,d0.source,seg,pose,1,0.)
            lab=label(d,w,h,circle)
            if lab is None:continue
            fv=feat(frame,d,circle)
            if fv is None:continue
            y,reason=lab; samples.append(Sample(fv[0],y,fv[1],reason)); st["positive" if y else "negative"]+=1
    return samples,st

def balance(samples,limit=96):
    neg=[s for s in samples if s.y==0]; pos=[s for s in samples if s.y==1]
    n=min(len(neg),len(pos),limit)
    if n<8:return []
    def spread(v):
        ids=np.linspace(0,len(v)-1,n).round().astype(int)
        return [v[int(i)] for i in ids]
    return spread(neg)+spread(pos)

def train(samples):
    x=np.stack([s.x for s in samples]).astype(np.float32)
    y=np.array([s.y for s in samples],np.int32)
    svm=cv2.ml.SVM_create(); svm.setType(cv2.ml.SVM_C_SVC); svm.setKernel(cv2.ml.SVM_LINEAR); svm.setC(.75)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,1200,1e-6))
    if not svm.train(x,cv2.ml.ROW_SAMPLE,y):raise SystemExit("SVM training failed")
    _,pred=svm.predict(x); acc=float((pred.reshape(-1).astype(np.int32)==y).mean())
    _,raw=svm.predict(x,flags=cv2.ml.STAT_MODEL_RAW_OUTPUT); raw=raw.reshape(-1)
    direction=1. if raw[y==1].mean()>raw[y==0].mean() else -1.
    return svm,direction,{"samples":len(samples),"positive":int((y==1).sum()),"negative":int((y==0).sum()),"pseudo_train_agreement":acc}

def prob(svm,direction,x):
    _,raw=svm.predict(x.reshape(1,-1).astype(np.float32),flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    z=max(-12.,min(12.,direction*float(raw[0,0])))
    return 1./(1.+math.exp(-z))

def bank(samples,out):
    chosen=[s for s in samples if s.y==0][:18]+[s for s in samples if s.y==1][:18]
    if not chosen:return
    tw,th,cols=180,140,6; rows=math.ceil(len(chosen)/cols)
    sheet=np.zeros((rows*th,cols*tw,3),np.uint8)
    for i,s in enumerate(chosen):
        scale=min((tw-8)/s.crop.shape[1],(th-30)/s.crop.shape[0])
        im=cv2.resize(s.crop,(max(1,int(s.crop.shape[1]*scale)),max(1,int(s.crop.shape[0]*scale))))
        rr,cc=divmod(i,cols); y0,x0=rr*th,cc*tw; ix=x0+(tw-im.shape[1])//2
        sheet[y0+22:y0+22+im.shape[0],ix:ix+im.shape[1]]=im
        text=("POS human" if s.y else "NEG "+s.reason)[:22]
        cv2.putText(sheet,text,(x0+4,y0+16),cv2.FONT_HERSHEY_SIMPLEX,.38,(80,220,80) if s.y else (60,80,230),1,cv2.LINE_AA)
    cv2.imwrite(str(out),sheet,[int(cv2.IMWRITE_JPEG_QUALITY),91])

def sheet(images,out):
    if not images:return
    ims=[]
    for im in images[:30]:
        scale=520/im.shape[1]; ims.append(cv2.resize(im,(520,max(1,int(im.shape[0]*scale)))))
    rh=max(i.shape[0] for i in ims); rows=math.ceil(len(ims)/2)
    s=np.zeros((rows*rh,1040,3),np.uint8)
    for i,im in enumerate(ims):
        r,c=divmod(i,2); s[r*rh:r*rh+im.shape[0],c*520:(c+1)*520]=im
    cv2.imwrite(str(out),s,[int(cv2.IMWRITE_JPEG_QUALITY),90])

def draw(frame,tiled,final,rejected,name):
    c=frame.copy()
    for d in tiled:cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(35,50,245),1)
    for d,p in rejected:
        cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(0,170,255),2)
        cv2.putText(c,f"HN {p:.2f}",(int(d.x1),max(14,int(d.y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,170,255),1)
    for d,p in final:
        cv2.rectangle(c,(int(d.x1),int(d.y1)),(int(d.x2),int(d.y2)),(70,220,70),2)
        cv2.putText(c,f"V3 {p:.2f}",(int(d.x1),max(14,int(d.y1)-3)),cv2.FONT_HERSHEY_SIMPLEX,.35,(70,220,70),1)
    cv2.rectangle(c,(0,0),(c.shape[1],28),(0,0,0),-1)
    cv2.putText(c,f"{name} cand={len(tiled)} v3={len(final)} hardneg={len(rejected)}",(7,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return c

def validate(folder,detector,segmenter,poser,svm,direction,a):
    tracks=[]; tid=1; rows=[]; visuals=[]
    st={"frames":0,"tiled":0,"seg_pose":0,"v2_accept":0,"hard_negative_reject":0,"v3_accept":0,"v3_positive_frames":0}
    for p in sorted(folder.glob("frame_*.jpg")):
        frame=cv2.imread(str(p))
        if frame is None:continue
        st["frames"]+=1; h,w=frame.shape[:2]; circle=base.detect_scope_circle(frame)
        tiled=base.detect_tiled(detector,frame,a.confidence)
        ev=evidence(frame,detector,segmenter,poser,a)
        tracks,tracked,tid=base.update_tracks(tracks,ev,w,h,tid)
        final=[]; rejected=[]; v2=0
        for d in tracked:
            if not base.final_accept(d,w,h,a.min_track_hits,circle):continue
            v2+=1; fv=feat(frame,d,circle)
            if fv is None:continue
            pr=prob(svm,direction,fv[0])
            if pr<.47 and d.pose_score<.58:rejected.append((d,pr))
            else:final.append((d,pr))
        st["tiled"]+=len(tiled); st["seg_pose"]+=len(ev); st["v2_accept"]+=v2
        st["hard_negative_reject"]+=len(rejected); st["v3_accept"]+=len(final); st["v3_positive_frames"]+=int(bool(final))
        rows.append({"frame":p.name,"width":w,"height":h,"tiled":len(tiled),"seg_pose":len(ev),"v2_accept":v2,"hard_negative_reject":len(rejected),"v3_accept":len(final)})
        visuals.append(draw(frame,tiled,final,rejected,p.name))
    return st,rows,visuals

def main():
    a=args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    detector,segmenter,poser=YOLO(a.model),YOLO(a.seg_model),YOLO(a.pose_model)
    mined,mining=mine(Path(a.train_frames),detector,segmenter,poser,a)
    samples=balance(mined)
    if len(samples)<16:raise SystemExit(f"insufficient balanced calibration samples: {len(samples)}")
    bank(samples,out/"hard_negative_training_bank.jpg")
    svm,direction,cal=train(samples); svm.save(str(out/"fps_hard_negative_svm.xml"))
    vst,rows,visuals=validate(Path(a.validation_frames),detector,segmenter,poser,svm,direction,a)
    with (out/"validation_detections_v3.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    sheet(visuals,out/"validation_contact_sheet_v3.jpg")
    report={"benchmark":"LiteView offline cross-source visible-human domain calibration v3","training_source_url":a.train_source_url,
      "validation_source_url":a.validation_source_url,"mining":mining,"calibrator":cal,"validation":vst,
      "important_limit":"Pseudo-label calibration is not ground-truth accuracy; visually audit the independent validation contact sheet.",
      "scope_limit":"Offline replay analysis only; no team/enemy identity, aiming, hidden-position inference, live coordinates, or anti-cheat behavior."}
    (out/"benchmark_domain_v3.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    summary=f"""## LiteView offline cross-source domain calibration v3

- Training frames: **{mining['frames']}**
- Hard negatives mined: **{mining['negative']}**
- Strong visible-human positives mined: **{mining['positive']}**
- Balanced calibration samples: **{cal['samples']}**
- Pseudo-label training agreement: **{cal['pseudo_train_agreement']:.3f}**
- Independent validation frames: **{vst['frames']}**
- V2 accepted boxes: **{vst['v2_accept']}**
- Hard-negative boxes suppressed: **{vst['hard_negative_reject']}**
- V3 boxes remaining: **{vst['v3_accept']}** on **{vst['v3_positive_frames']}/{vst['frames']}** frames

Not an accuracy claim. The independent validation contact sheet must be visually audited before mobile integration.
"""
    (out/"summary_domain_v3.md").write_text(summary,encoding="utf-8"); print(summary); return 0
if __name__=="__main__":raise SystemExit(main())
