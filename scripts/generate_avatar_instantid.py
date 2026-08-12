#!/usr/bin/env python3
"""Generate a VCCP-style avatar using InstantID (face-embedding identity
preservation) + depth ControlNet + your trained LoRA, via our patched fork
of fofr/face-to-many (haris3545/face-to-many-patched -- see vendor/DEPLOY.md
for why: the hosted original rejects our account's replicate.delivery URLs).

Unlike the Canny ControlNet approach, InstantID conditions on a facial
identity embedding rather than raw pixel edges, and depth conditioning
doesn't carry fine surface texture (wrinkles, fabric weave) the way Canny
edges do -- this should avoid the "sharpen structure = also render literal
photo detail" tradeoff we hit with the Canny pipeline.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/generate_avatar_instantid.py path/to/photo.jpg --lora-weights https://.../trained_model.tar
"""
import argparse
import sys
from pathlib import Path

MODEL = "haris3545/face-to-many-patched:2f26886c521b71658dfaa2b71a8f116b8626a1d9f7e641dd0025f966056ee5dc"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--lora-weights",
        required=True,
        help="URL to your trained LoRA weights .tar (from scripts/get_lora_weights_url.py)",
    )
    parser.add_argument("--out", default="avatar_out_iid.png", help="Where to save the result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instant-id-strength",
        type=float,
        default=0.8,
        help="how strongly facial identity is preserved, 0-1",
    )
    parser.add_argument(
        "--control-depth-strength",
        type=float,
        default=0.6,
        help="how strongly depth/pose structure is enforced, 0-1",
    )
    parser.add_argument(
        "--denoising-strength",
        type=float,
        default=0.95,
        help="how much the original photo is replaced (1 = full regeneration guided by "
        "prompt/LoRA/InstantID, 0 = keep the original photo), 0-1",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--prompt",
        default="TOK style, monochrome black and white line art portrait illustration, "
        "pure solid white background, no background detail, "
        "flat solid black fills, minimal occasional cross-hatching only, mostly flat shapes, "
        "simple graphic dot eyes, simplified cartoon facial features, no fine detail, "
        "no gradients, halftone dot texture used sparingly only where the actual photo shows it, "
        "clothing rendered as one flat solid grey shape with no pattern, closed mouth gentle smile, "
        "graphic vector illustration, no photorealism, centered head and shoulders portrait crop",
    )
    parser.add_argument(
        "--negative-prompt",
        default="photo, photorealistic, color, colour, coloured, tinted, gradient background, "
        "grey background, dark background, textured background, visible wall, chalkboard, vignette, "
        "sepia, muted tones, painterly, blurry, low quality, grayscale photo, "
        "open mouth, teeth, tongue, "
        "realistic eyes, detailed iris, photorealistic skin texture, dense stippling, "
        "intricate fine detail, engraving texture, fabric texture detail, fabric pattern, "
        "floral print, patterned clothing, full body, torso, waist",
    )
    parser.add_argument(
        "--clean-shaven",
        action="store_true",
        help="explicitly suppress the LoRA's learned tendency to add jaw/beard texture",
    )
    args = parser.parse_args()

    negative_prompt = args.negative_prompt
    if args.clean_shaven:
        negative_prompt += ", beard, facial hair, stubble, goatee, moustache, mustache"

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    import replicate

    print("Uploading photo...")
    with open(photo_path, "rb") as f:
        uploaded = replicate.files.create(f)
    photo_url = uploaded.urls["get"]

    print(f"Generating with {MODEL} (InstantID + depth ControlNet + LoRA)...")
    prediction = replicate.predictions.create(
        version=MODEL,
        input={
            "image": photo_url,
            "style": "3D",  # required enum, but custom_lora_url overrides the actual style used
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "custom_lora_url": args.lora_weights,
            "lora_scale": args.lora_scale,
            "instant_id_strength": args.instant_id_strength,
            "control_depth_strength": args.control_depth_strength,
            "denoising_strength": args.denoising_strength,
            "seed": args.seed,
        },
    )
    print(f"Prediction: https://replicate.com/p/{prediction.id}")
    prediction.wait()

    print(f"Status: {prediction.status}")
    if prediction.status != "succeeded":
        print(f"Logs:\n{prediction.logs}")
        sys.exit(f"Prediction did not succeed (status={prediction.status}): {prediction.error}")

    output = prediction.output
    if not output:
        print(f"Logs:\n{prediction.logs}")
        sys.exit("Prediction succeeded but returned no output -- see logs above")

    result = output[0] if isinstance(output, list) else output
    import urllib.request

    urllib.request.urlretrieve(result, args.out)

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
