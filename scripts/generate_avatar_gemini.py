#!/usr/bin/env python3
"""Generate a VCCP-style avatar from a photo using Gemini's image model
(gemini-2.5-flash-image, aka "Nano Banana") instead of the SDXL/InstantID
pipeline.

Gemini's image model has no fine-tuning/LoRA API -- there's no equivalent
of scripts/train_lora.py here. Instead, style consistency comes entirely
from what's sent in each request: a fixed set of curated reference
avatars (the same ones used to build the LoRA dataset) plus a fixed,
detailed style prompt, sent alongside the subject's photo on every call.
Keeping both fixed across runs is what stands in for "training" -- the
model doesn't need it to do strong style transfer from a few in-context
examples, which is exactly what made this worth trying: a single
reference image and a plain-language instruction already outperformed the
InstantID pipeline on a first try, with none of the halftone/stipple/
gappy-glasses fighting that pipeline needed.

Usage:
    export GEMINI_API_KEY=...   # from https://aistudio.google.com/apikey
    python3 scripts/generate_avatar_gemini.py path/to/photo.jpg
    python3 scripts/generate_avatar_gemini.py path/to/photo.jpg --references data/avatars/A.png,data/avatars/B.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A small, fixed set of existing house-style avatars, sent as in-context
# style references on every call -- fixed on purpose, the same way a
# training caption is fixed, so results stay comparable across different
# subjects instead of drifting with whatever reference happened to be
# picked that day. Deliberately picked for variety (different hair, one
# with glasses) rather than all-similar, so the model generalizes the
# *style* (linework weight, flat tones, feature simplification) instead
# of copying one person's specific features.
DEFAULT_REFERENCES = [
    ROOT / "data" / "avatars" / "Andrew Karallis 2021 (1).png",
    ROOT / "data" / "avatars" / "Becky Johnstone 2025.png",
    ROOT / "data" / "avatars" / "Greg Wocial Avatar 2025-2.png",
]

DEFAULT_PROMPT = (
    "Redraw the attached photo as a VCCP-style graphic portrait avatar, matching the exact "
    "illustration style of the other attached reference avatar images -- not just loosely "
    "inspired by them, but the same house style. "
    "Preserve this specific person's likeness and proportions (face shape, hairstyle, glasses if "
    "worn) from the photo; take only the drawing style from the references. "
    "Style rules: clean confident black outlines of one consistent bold weight throughout the "
    "whole image, fully anti-aliased, no wobble or pixel jaggedness. "
    "Hair is a flat solid black silhouette with a jagged/tufted hairline and a handful of individual "
    "tapered strand marks, not a plain smooth blob. "
    "Eyes are a small solid pupil dot plus one thin curved upper-eyelid arc -- no other eye detail. "
    "Eyebrows are a single thin arched stroke, clearly separated from the eyes and from glasses if worn. "
    "Nose is the smallest possible mark -- a tiny curved tick, not a full nostril outline. "
    "Mouth is a simple two-line smile: one upper curve, one short lower lip line, minimal detail. "
    "Glasses (if the subject wears them) are a bold, thick, fully continuous outlined frame with no "
    "gaps or breaks anywhere in the line. "
    "Clothing is a flat mid-grey silhouette with a second, slightly darker flat grey patch for "
    "shading on one side (no gradient), plus a couple of simple interior lines for a collar or "
    "seam if the photo shows one. "
    "Background is solid flat white. "
    "No photorealistic shading, no halftone dots, no stipple, no gradients anywhere -- every tone is "
    "a flat, evenly-colored shape. "
    "Crop tight to head and shoulders, filling most of the frame."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--references",
        default=",".join(str(p) for p in DEFAULT_REFERENCES),
        help="Comma-separated paths to reference avatar images sent as style examples. "
        "Fixed defaults are used unless overridden -- keep this consistent across runs "
        "for comparable results, the same reasoning as a fixed training caption.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Style instruction sent with the photo")
    parser.add_argument("--out", default="avatar_out_gemini.png", help="Where to save the result")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help="Gemini image model ID. Override if this ID has since been renamed/retired.",
    )
    args = parser.parse_args()

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    reference_paths = [Path(p.strip()) for p in args.references.split(",") if p.strip()]
    missing = [p for p in reference_paths if not p.exists()]
    if missing:
        sys.exit("Reference image(s) not found: " + ", ".join(str(p) for p in missing))

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey)")

    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)

    print(f"Loading {len(reference_paths)} reference avatar(s)...")
    contents = [args.prompt]
    for p in reference_paths:
        contents.append(Image.open(p))
    contents.append(Image.open(photo_path))

    print(f"Generating with {args.model}...")
    response = client.models.generate_content(model=args.model, contents=contents)

    saved = False
    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) is not None:
            from io import BytesIO

            Image.open(BytesIO(part.inline_data.data)).save(args.out)
            saved = True
            break

    if not saved:
        text = "".join(part.text for part in response.candidates[0].content.parts if getattr(part, "text", None))
        sys.exit(f"No image returned. Model response:\n{text}")

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
