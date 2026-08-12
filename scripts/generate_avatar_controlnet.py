#!/usr/bin/env python3
"""Generate a VCCP-style avatar using ControlNet (Canny edges from the photo)
+ your trained LoRA weights, so facial structure and style strength are
controlled independently instead of trading off against each other.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/generate_avatar_controlnet.py path/to/photo.jpg --lora-weights https://.../trained_model.tar
"""
import argparse
import io
import sys
from pathlib import Path

from PIL import Image

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
        "--lora-scale",
        type=float,
        default=1.0,
        help="how strongly the trained style LoRA pulls the output toward flat/graphic vs. the base model's realism, 0-2",
    )
    parser.add_argument(
        "--crop-top",
        type=float,
        default=0.10,
        help="discard this fraction of the photo's height from the top, to remove excess headroom above the head",
    )
    parser.add_argument(
        "--crop-bottom",
        type=float,
        default=0.62,
        help="keep down to this fraction of the photo's height (measured from the original top), "
        "to crop out excess torso before generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="fixed seed so re-runs with tweaked params are comparable instead of randomly varying",
    )
    parser.add_argument(
        "--prompt",
        default="TOK style, monochrome black and white line art portrait illustration, "
        "pure solid white background, no background detail, "
        "flat solid black fills, minimal occasional cross-hatching only, mostly flat shapes, "
        "simple graphic dot eyes, simplified cartoon facial features, no fine detail, "
        "no gradients, halftone dot texture used sparingly only where the actual photo shows it, "
        "graphic vector illustration, no photorealism, centered head and shoulders portrait crop",
    )
    parser.add_argument(
        "--negative-prompt",
        default="photo, photorealistic, color, colour, coloured, tinted, gradient background, "
        "grey background, dark background, textured background, visible wall, chalkboard, vignette, "
        "sepia, muted tones, painterly, blurry, low quality, grayscale photo, "
        "realistic eyes, detailed iris, photorealistic skin texture, dense stippling, "
        "intricate fine detail, engraving texture, fabric texture detail, full body, torso, waist",
    )
    parser.add_argument(
        "--clean-shaven",
        action="store_true",
        help="this person has no beard/stubble in the photo -- explicitly suppress the model's learned "
        "tendency to add jaw/beard texture regardless of the actual photo",
    )
    args = parser.parse_args()

    negative_prompt = args.negative_prompt
    if args.clean_shaven:
        negative_prompt += ", beard, facial hair, stubble, goatee, moustache, mustache, five o'clock shadow"

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    import replicate

    print(f"Cropping photo to {int(args.crop_top * 100)}%-{int(args.crop_bottom * 100)}% of height...")
    im = Image.open(photo_path).convert("RGB")
    w, h = im.size
    im = im.crop((0, int(h * args.crop_top), w, int(h * args.crop_bottom)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    buf.name = "cropped.jpg"

    print("Uploading photo...")
    uploaded = replicate.files.create(buf)
    photo_url = uploaded.urls["get"]

    print(f"Generating with {MODEL} (canny ControlNet + LoRA, seed={args.seed})...")
    output = replicate.run(
        MODEL,
        input={
            "image": photo_url,
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "lora_weights": args.lora_weights,
            "lora_scale": args.lora_scale,
            "condition_scale": args.condition_scale,
            "num_inference_steps": 40,
            "refine_steps": 20,
            "seed": args.seed,
        },
    )

    result = output[0] if isinstance(output, list) else output
    with open(args.out, "wb") as f:
        f.write(result.read())

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
