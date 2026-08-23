from __future__ import annotations

import argparse

from videobrain import GenerationRequest, VideoBrainEngine


def run_cli(args: argparse.Namespace) -> None:
    engine = VideoBrainEngine()
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    request = GenerationRequest(
        prompt=args.prompt, negative_prompt=args.negative_prompt, target_style=args.target_style,
        task=args.task, model_id=args.model, profile=args.profile, width=args.width, height=args.height,
        num_frames=args.frames, fps=args.fps, output_fps=args.output_fps, duration_seconds=args.duration,
        overlap_frames=args.overlap_frames, interpolation=args.interpolation, continuity_mode=args.continuity,
        steps=args.steps, guidance_scale=args.guidance, seed=args.seed, input_image=args.input_image,
        reference_video=args.reference_video, input_files=tuple(args.input or ()), allow_download=args.allow_download,
        analysis_frames=args.analysis_frames, output_formats=formats, output_name=args.output_name,
        offload_mode=args.offload, resume=not args.no_resume, keep_segments=not args.delete_segments,
        max_segments=args.max_segments, gpu_mode=args.gpu_mode, detail_enhance=not args.no_detail_enhance,
        detail_strength=args.detail_strength,
    )

    def progress(done: int, total: int, message: str) -> None:
        prefix = f"[{done}/{total}] " if total else ""
        print(prefix + message, flush=True)

    result = engine.generate(request, progress=progress)
    print(result.output_path)
    for output in result.output_paths:
        print(f"output={output}")
    print(f"model={result.model_name}")
    print(f"gpu_mode={result.gpu_mode}")
    print(f"gpu_roles={result.gpu_roles}")
    print(f"active_gpu_count={result.active_gpu_count}")
    print(f"duration={result.actual_duration_seconds:.3f}")
    print(f"output_fps={result.output_fps}")
    print(f"segments={len(result.segment_paths)}")
    for warning in result.warnings:
        print(f"warning={warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoBrain 本地多GPU长视频生成工作台")
    parser.add_argument("--cli", action="store_true", help="使用命令行模式")
    parser.add_argument("--prompt", default="", help="视频提示词")
    parser.add_argument("--negative-prompt", default="", help="负面提示词")
    parser.add_argument("--target-style", default="", help="目标风格/二创要求")
    parser.add_argument("--task", choices=["auto", "text_to_video", "image_to_video", "video_reference"], default="auto")
    parser.add_argument("--input", action="append", help="输入文件，可重复使用")
    parser.add_argument("--input-image", help="参考图片")
    parser.add_argument("--reference-video", help="参考视频；系统会抽关键帧理解后二创")
    parser.add_argument("--model", default="auto", help="模型ID，默认自动选择")
    parser.add_argument("--profile", choices=["fast", "standard", "quality"], default="standard")
    parser.add_argument("--gpu-mode", choices=["auto", "single", "dual", "triple"], default="auto", help="GPU模式；auto按实际GPU数量自动降级")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=49, help="每个生成片段的基础帧数")
    parser.add_argument("--fps", type=int, default=8, help="模型基础FPS")
    parser.add_argument("--output-fps", type=int, default=24, help="最终输出FPS")
    parser.add_argument("--duration", type=float, default=30.0, help="最终目标时长（秒）")
    parser.add_argument("--overlap-frames", type=int, default=6, help="片段之间用于融合的重叠帧")
    parser.add_argument("--interpolation", choices=["optical_flow", "blend", "none"], default="optical_flow")
    parser.add_argument("--continuity", choices=["auto", "i2v", "same_model", "blend_only"], default="auto")
    parser.add_argument("--offload", choices=["auto", "group", "model", "sequential", "none"], default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-frames", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=120)
    parser.add_argument("--no-detail-enhance", action="store_true", help="关闭逐段轻量细节增强")
    parser.add_argument("--detail-strength", type=float, default=0.12, help="细节增强强度，建议0.08-0.20")
    parser.add_argument("--formats", default="mp4,json,storyboard_txt", help="逗号分隔：mp4,webm,mov,mkv,avi,gif,png_frames,jpg_frames,json,storyboard_txt,storyboard_md")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="禁用断点续跑")
    parser.add_argument("--delete-segments", action="store_true", help="成功后删除中间片段")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cli:
        has_input = bool(args.prompt.strip() or args.input or args.input_image or args.reference_video)
        if not has_input:
            raise SystemExit("--cli 模式必须提供 --prompt、--input、--input-image 或 --reference-video 之一")
        run_cli(args)
        return
    from videobrain.ui import VideoBrainUI
    VideoBrainUI().run()


if __name__ == "__main__":
    main()
