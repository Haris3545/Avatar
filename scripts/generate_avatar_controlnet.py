#!/usr/bin/env python3
"""Generate a VCCP-style avatar using ControlNet (Canny edges from the photo)
+ your trained LoRA weights, so facial structure and style strength are
controlled independently instead of trading off against each other.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/generate_avatar_controlnet.py path/to/photo.jpg --lora-weights https://.../trained_model.tar
"""
import argparse
import sys
from pathlib import Path

MODEL = "fermatresearch/sdxl-controlnet-lora:3bb13fe1c33c35987b33792b01b71ed6529d03f165d1c2416375859f09ca9fef"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--lora-weights",
        required=True,
        help="URL to your trained LoRA weights .tar (from scripts/get_lora_weights_url.py)",
    )
    parser.add_argument("--out", default="avatar_out_cn.png", help="Where to save the result")
    parser.add_argument(
        "--condition-scale",
        type=float,
        default=0.7,
        help="how strongly the canny edges (facial structure) are enforced, 0-2. Higher = more faithful to photo.",
    )
    parser.add_argument(
        "--prompt",
        default="TOK style, monochrome black and white line art portrait illustration, "
        "pure black ink outline on solid white background, thin clean outline, "
        "flat two-tone shading only, no gradients, halftone dot texture on beard and shadow areas, "
        "simplified facial features, graphic vector illustration, no photorealism",
    )
    parser.add_argument(
        "--negative-prompt",
        default="photo, photorealistic, color, colour, coloured, tinted, gradient background, "
        "sepia, muted tones, painterly, blurry, low quality, grayscale photo",
    )
    args = parser.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    import replicate

    print("Uploading photo...")
    with open(photo_path, "rb") as f:
        uploaded = replicate.files.create(f)
    photo_url = uploaded.urls["get"]

    print(f"Generating with {MODEL} (canny ControlNet + LoRA)...")
    output = replicate.run(
        MODEL,
        input={
            "image": photo_url,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "lora_weights": args.lora_weights,
            "condition_scale": args.condition_scale,
            "num_inference_steps": 40,
            "refine_steps": 20,
        },
    )

    result = output[0] if isinstance(output, list) else output
    with open(args.out, "wb") as f:
        f.write(result.read())

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
