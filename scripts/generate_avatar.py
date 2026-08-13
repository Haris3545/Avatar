#!/usr/bin/env python3
"""Generate a VCCP-style avatar from a photo using the trained LoRA + a
ControlNet (Canny edges) to keep the output following the photo's structure.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/generate_avatar.py path/to/photo.jpg --lora haris3545/vccp-avatar-lora:VERSION
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--lora",
        required=True,
        help="owner/model:version of your trained LoRA (find the version hash on the model's Replicate page)",
    )
    parser.add_argument("--out", default="avatar_out.png", help="Where to save the result")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="img2img prompt_strength: lower stays closer to the photo, higher restyles more freely (0-1)",
    )
    parser.add_argument(
        "--prompt",
        default="TOK style, monochrome black and white line art portrait illustration, "
        "pure black ink outline on solid white background, thick clean rounded outline, "
        "flat two or three tone shading only, no gradients, no halftone dot texture, no cross-hatching, "
        "no stipple, no shading device of any kind on beard, jaw, or hair, "
        "simplified facial features, graphic vector illustration, no photorealism",
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

    print(f"Generating with {args.lora} (img2img)...")
    output = replicate.run(
        args.lora,
        input={
            "prompt": args.prompt,
            "negative_prompt": "photo, photorealistic, color, colour, coloured, tinted, gradient background, "
            "sepia, muted tones, painterly, blurry, low quality, grayscale photo",
            "image": photo_url,
            # lower = stays closer to the input photo's structure/pose,
            # higher = more freedom to restyle. 0.6-0.7 is a reasonable start.
            "prompt_strength": args.strength,
            "num_inference_steps": 30,
        },
    )

    result = output[0] if isinstance(output, list) else output
    with open(args.out, "wb") as f:
        f.write(result.read())

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
