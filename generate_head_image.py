"""Generate photorealistic head portraits from text with the local RV5.1 model.

Example:
    python generate_head_image.py --prompt "a young man with short black hair"
    python generate_head_image.py --prompt "Barack Obama, looking directly at the camera" --num-images 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR.parent / "HeadStudio_lib" / "realistic-vision-51"
).resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "text_to_head"

# 更强的完整头部构图约束，避免头顶被裁掉
# 1. 简化的构图 Prompt（大幅缩减 Token 占用，避免被 SD1.5 截断）
PORTRAIT_PROMPT = (
    "centered head-and-shoulders portrait, medium close-up, "
    "entire head visible, plain neutral background"
)

# 2. 针对 SD 1.5 强有效的“大张嘴/发音口型”触发词
# 使用强烈的视觉动作词（gasping / singing / gaping mouth）能强制拉开下巴
OPEN_MOUTH_PROMPT = (
    "wide open mouth, gaping mouth, gasping expression, singing, "
    "parted lips, visible upper and lower teeth, visible tongue"
)

QUALITY_PROMPT = (
    "RAW photo, professional DSLR portrait, realistic skin texture, sharp focus"
)

# 3. 补充强力负向提示词，防止嘴唇合拢、咬牙或死板的微笑
DEFAULT_NEGATIVE_PROMPT = (
    "closed mouth, lips closed, lips touching, pursed lips, clenched jaw, smile, smiling, "
    "teeth touching, shut mouth, "
    "extreme close-up, tight crop, cropped head, top of head cut off, out of frame, "
    "worst quality, low quality, lowres, blurry, out of focus, jpeg artifacts, "
    "watermark, text, logo, signature, duplicate, multiple people, extra face, "
    "deformed, disfigured, bad anatomy, bad proportions, asymmetry, cross-eyed, "
    "glasses, eyeglasses, strong shadow, harsh shadow, cgi, 3d render, cartoon, anime"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the local Realistic Vision 5.1 model to generate head images."
    )

    parser.add_argument(
        "-p",
        "--prompt",
        default="a DSLR portrait of 25-year-old male of Indian descent with a square face, neatly trimmed beard, and short curly black hair wearing a fitted gray suit",
        help="Person/appearance description.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Things that should not appear in the generated image.",
    )
    parser.add_argument(
        "--portrait-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add full-head portrait composition constraints.",
    )
    parser.add_argument(
        "--open-mouth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a natural open-mouth speaking expression.",
    )
    parser.add_argument(
        "--quality-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add built-in photorealistic quality prompt.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local Diffusers model directory (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output PNG path, or a directory when generating multiple images.",
    )
    parser.add_argument("--num-images", type=int, default=1, help="Number of images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=35, help="Denoising steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=8.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("dpmpp-2m-karras", "euler-a", "pndm"),
        default="dpmpp-2m-karras",
        help="Sampling scheduler; DPM++ 2M Karras is recommended for RV5.1.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Image height.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Model precision; auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        help="Enable attention/VAE slicing to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--enable-safety-checker",
        action="store_true",
        help="Enable the bundled safety checker.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.model_path = args.model_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.prompt.strip():
        raise ValueError("--prompt cannot be empty")
    if not (args.model_path / "model_index.json").is_file():
        raise FileNotFoundError(
            f"RV5.1 Diffusers model not found at: {args.model_path}"
        )
    if args.num_images < 1:
        raise ValueError("--num-images must be at least 1")
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.guidance_scale < 0:
        raise ValueError("--guidance-scale cannot be negative")
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be at least 64")
    if args.width % 8 or args.height % 8:
        raise ValueError("--width and --height must be multiples of 8")


def resolve_runtime(args: argparse.Namespace, torch: Any) -> tuple[str, Any]:
    device = args.device.lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device != "cpu" and not device.startswith("cuda"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "float16" if device.startswith("cuda") else "float32"

    if device == "cpu" and dtype_name == "float16":
        raise ValueError("float16 inference is not supported on CPU; use --dtype float32")

    return device, getattr(torch, dtype_name)


def build_output_paths(output: Path, count: int, seed: int) -> list[Path]:
    if output.suffix.lower() == ".png":
        output.parent.mkdir(parents=True, exist_ok=True)
        if count == 1:
            return [output]
        return [
            output.with_name(f"{output.stem}_{index:03d}{output.suffix}")
            for index in range(count)
        ]

    output.mkdir(parents=True, exist_ok=True)
    return [output / f"head_seed{seed + index}.png" for index in range(count)]


def configure_scheduler(pipe: Any, scheduler_name: str) -> None:
    if scheduler_name == "pndm":
        return

    if scheduler_name == "dpmpp-2m-karras":
        from diffusers import DPMSolverMultistepScheduler

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=True,
        )
        return

    if scheduler_name == "euler-a":
        from diffusers import EulerAncestralDiscreteScheduler

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        return

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_prompt(args: argparse.Namespace) -> str:
    prompt_parts = []

    # 优先放张嘴指令，赋予最高权重
    if args.open_mouth:
        prompt_parts.append(OPEN_MOUTH_PROMPT)

    # 主体描述
    prompt_parts.append(args.prompt.strip())

    # 构图与质量描述
    if args.portrait_prompt:
        prompt_parts.append(PORTRAIT_PROMPT)

    if args.quality_prompt:
        prompt_parts.append(QUALITY_PROMPT)

    return ", ".join(prompt_parts)


def warn_prompt_truncation(pipe: Any, text: str, name: str) -> None:
    tokenized = pipe.tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
    )
    token_count = len(tokenized["input_ids"])
    max_length = pipe.tokenizer.model_max_length

    print(f"{name} tokens: {token_count}/{max_length}")
    if token_count > max_length:
        print(
            f"Warning: {name} exceeds the tokenizer limit. "
            f"Tokens after position {max_length} will be ignored."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from diffusers import StableDiffusionPipeline
    from PIL.PngImagePlugin import PngInfo

    device, dtype = resolve_runtime(args, torch)
    prompt = build_prompt(args)

    print(f"Loading RV5.1 from {args.model_path}")
    print(f"Device: {device}; dtype: {dtype}; seed: {args.seed}")
    print(f"Final Prompt: {prompt}")
    print(f"Negative Prompt: {args.negative_prompt}")

    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
    }
    if not args.enable_safety_checker:
        load_kwargs.update(
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_path),
        **load_kwargs,
    )
    configure_scheduler(pipe, args.scheduler)
    pipe = pipe.to(device)

    if args.low_vram:
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    warn_prompt_truncation(pipe, prompt, "Prompt")
    warn_prompt_truncation(pipe, args.negative_prompt, "Negative prompt")

    generators = [
        torch.Generator(device=device).manual_seed(args.seed + index)
        for index in range(args.num_images)
    ]

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            generator=generators,
        )

    output_paths = build_output_paths(args.output, len(result.images), args.seed)
    nsfw_flags = result.nsfw_content_detected or [False] * len(result.images)

    for index, (image, output_path) in enumerate(zip(result.images, output_paths)):
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("negative_prompt", args.negative_prompt)
        metadata.add_text("model", str(args.model_path))
        metadata.add_text("seed", str(args.seed + index))
        metadata.add_text("scheduler", args.scheduler)
        metadata.add_text("steps", str(args.steps))
        metadata.add_text("guidance_scale", str(args.guidance_scale))
        metadata.add_text("width", str(args.width))
        metadata.add_text("height", str(args.height))
        metadata.add_text("portrait_prompt", str(args.portrait_prompt))
        metadata.add_text("open_mouth", str(args.open_mouth))
        metadata.add_text("quality_prompt", str(args.quality_prompt))
        image.save(output_path, pnginfo=metadata)

        warning = " [safety checker flagged this image]" if nsfw_flags[index] else ""
        print(f"Saved: {output_path}{warning}")


if __name__ == "__main__":
    main()