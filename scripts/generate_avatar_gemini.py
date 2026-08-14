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

Default references are the *entire* existing avatar set (data/avatars/),
optionally filtered down by --glasses/--no-glasses and --beard/--no-beard
to only the references that match the subject's actual features (see
scripts/tag_avatar_references.py, which classifies each reference once
and caches the result in data/avatar_tags.json). Sending the whole
unfiltered set produced results that were more polished than the house
style but didn't really look like it: averaging in-context examples
across many different feature combinations (glasses vs not, beard vs
not) muddies exactly the per-feature drawing conventions -- how glasses
are drawn, how a beard is drawn -- that make an avatar read as "ours" for
a specific class of subject. Filtering to only matching references keeps
those conventions intact.

Usage:
    export GEMINI_API_KEY=...   # from https://aistudio.google.com/apikey
    python3 scripts/tag_avatar_references.py   # one-time, cached afterward
    python3 scripts/generate_avatar_gemini.py path/to/photo.jpg --glasses --no-beard
    python3 scripts/generate_avatar_gemini.py path/to/photo.jpg --references data/avatars/A.png,data/avatars/B.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The entire existing avatar set, sent as in-context style references on
# every call -- fixed on purpose (same directory, same sort order every
# run), so results stay comparable across different subjects instead of
# drifting with whatever subset happened to be picked that day. Sorted
# for a deterministic, reviewable request rather than directory-listing
# order, which can vary by filesystem. Narrowed at runtime by
# filter_references() when --glasses/--beard flags are given.
AVATARS_DIR = ROOT / "data" / "avatars"
TAGS_PATH = ROOT / "data" / "avatar_tags.json"
DEFAULT_REFERENCES = sorted(
    p for p in AVATARS_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
)


def filter_references(paths, want_glasses, want_beard):
    """Narrow the reference set to only avatars matching the requested
    glasses/beard category, using the cache from tag_avatar_references.py.
    Falls back to the unfiltered set (with a warning) if the cache is
    missing, or if a category is requested but fewer than 3 references
    match it -- too few in-context examples risks overfitting to those
    specific people's features rather than generalizing the style."""
    if want_glasses is None and want_beard is None:
        return paths

    if not TAGS_PATH.exists():
        print(f"Warning: {TAGS_PATH} not found -- run scripts/tag_avatar_references.py first. Using unfiltered references.")
        return paths

    tags = json.loads(TAGS_PATH.read_text())
    filtered = []
    for p in paths:
        t = tags.get(p.name)
        if t is None:
            continue
        if want_glasses is not None and t.get("glasses") != want_glasses:
            continue
        if want_beard is not None and t.get("beard") != want_beard:
            continue
        filtered.append(p)

    if len(filtered) < 3:
        print(f"Warning: only {len(filtered)} reference(s) matched the requested category -- falling back to the full set.")
        return paths

    print(f"Filtered to {len(filtered)} reference(s) matching glasses={want_glasses}, beard={want_beard}")
    return filtered

STRUCTURE_PREAMBLE = (
    "The FIRST attached image is a structural line-art drawing generated directly from this "
    "exact photo's measured face geometry (jaw shape, eyebrow/eye/nose/mouth position, hairline, "
    "glasses shape, shoulder/collar line). Use it as the authoritative source for proportions and "
    "feature positions -- match its face shape, feature spacing, and pose exactly, do not "
    "reinterpret or re-guess the structure from the photo. Your job is to redraw/polish that exact "
    "structural drawing into finished line art -- improving its stroke confidence, hair rendering, "
    "and shading -- not to generate a fresh interpretation of the photo. "
    "The SECOND attached image is the original photo, included only for actual visual detail the "
    "structural drawing doesn't capture (skin tone, real hair color/texture, precise glasses "
    "color) -- never let it override the first image's proportions. "
    "The remaining attached images are house-style reference avatars for line quality and "
    "rendering style only, not structure. "
)

DEFAULT_PROMPT = (
    "Redraw the attached photo as a VCCP-style graphic portrait avatar, matching the exact "
    "illustration style of the other attached reference avatar images -- not just loosely "
    "inspired by them, but the same house style. "
    "Preserve this specific person's likeness and proportions (face shape, hairstyle, glasses if "
    "worn) from the photo; take only the drawing style from the references. "
    "Style rules: this is a quick, confident SKETCH, not a heavily rendered vector illustration -- "
    "use the minimum number of strokes possible for every feature. Every line is thin and light, "
    "one single consistent weight throughout (not bold or thick anywhere), like a fine felt-tip "
    "pen, fully anti-aliased with no wobble or pixel jaggedness. "
    "Keep the head a natural, moderate size relative to the shoulders -- not an oversized/cartoon "
    "head. "
    "Hair is a flat solid fill matching this specific person's actual hair darkness (near-black "
    "for dark hair, grey only if their real hair is genuinely light or grey) with just a small "
    "handful of spiky tufts breaking the top edge -- no interior hatching, no directional strand "
    "linework covering the hair mass, regardless of hair color. "
    "Eyes are small and easy to miss: a tiny solid pupil dot plus one very thin, shallow upper-"
    "eyelid arc -- no other eye detail, and never large or prominent. "
    "Eyebrows are a single thin curved stroke, not a thick or filled shape, clearly separated from "
    "the eyes and from glasses if worn. "
    "Nose is the smallest possible mark, often barely visible -- a tiny curved tick at most, never "
    "a full nostril outline. "
    "Mouth is a simple, subtle smile -- one or two very thin short lines, minimal detail, not a "
    "pronounced or heavily-lined smile. "
    "Glasses (if the subject wears them) are a continuous outlined frame with no gaps or breaks "
    "anywhere in the line, drawn at the same thin weight as everything else, not thicker. "
    "Only draw beard/stubble texture if the subject clearly has visible facial hair in the photo, "
    "and keep it minimal and confined -- never spread dot texture across the whole lower face. If "
    "the photo shows a clean-shaven or barely-stubbled face, draw no facial hair texture at all. "
    "Clothing is a flat grey silhouette, with a second, slightly different flat grey patch for "
    "shading on one side (no gradient), plus at most one or two simple interior lines for a "
    "collar or button if the photo shows one -- don't over-detail the clothing. "
    "Background is solid flat white. "
    "No photorealistic shading, no gradients, no dense texture anywhere -- every tone is a flat, "
    "evenly-colored shape, and the overall impression should be sparse and quick-sketched, not "
    "densely rendered or 'finished-looking'. "
    "Crop tight to head and shoulders, filling most of the frame."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--references",
        default=",".join(str(p) for p in DEFAULT_REFERENCES),
        help="Comma-separated paths to reference avatar images sent as style examples. "
        f"Defaults to the entire data/avatars/ set ({len(DEFAULT_REFERENCES)} images) unless "
        "overridden -- keep this consistent across runs for comparable results, the same "
        "reasoning as a fixed training caption.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Style instruction sent with the photo")
    parser.add_argument("--out", default="avatar_out_gemini.png", help="Where to save the result")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help="Gemini image model ID. Override if this ID has since been renamed/retired.",
    )
    parser.add_argument(
        "--glasses",
        action="store_true",
        help="subject wears glasses -- only use glasses-wearing references (see tag_avatar_references.py)",
    )
    parser.add_argument(
        "--no-glasses",
        action="store_true",
        help="subject does not wear glasses -- exclude glasses-wearing references",
    )
    parser.add_argument(
        "--beard",
        action="store_true",
        help="subject has a beard/visible facial hair -- only use bearded references",
    )
    parser.add_argument(
        "--no-beard",
        action="store_true",
        help="subject is clean-shaven -- exclude bearded references",
    )
    parser.add_argument(
        "--structure-image",
        help="Path to a structural line-art image (composite_line_art.py's output) to anchor "
        "proportions/positions -- generated automatically from the photo if not given. Pass "
        "--no-structure to disable this entirely and generate from the photo alone.",
    )
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="skip the structural reference entirely and generate from the photo alone (the "
        "original behavior) -- useful for comparing against the structure-anchored result.",
    )
    args = parser.parse_args()

    if args.glasses and args.no_glasses:
        sys.exit("--glasses and --no-glasses are mutually exclusive")
    if args.beard and args.no_beard:
        sys.exit("--beard and --no-beard are mutually exclusive")

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    reference_paths = [Path(p.strip()) for p in args.references.split(",") if p.strip()]
    missing = [p for p in reference_paths if not p.exists()]
    if missing:
        sys.exit("Reference image(s) not found: " + ", ".join(str(p) for p in missing))

    want_glasses = True if args.glasses else (False if args.no_glasses else None)
    want_beard = True if args.beard else (False if args.no_beard else None)
    reference_paths = filter_references(reference_paths, want_glasses, want_beard)

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey)")

    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)

    structure_path = None
    if args.structure_image:
        structure_path = Path(args.structure_image)
        if not structure_path.exists():
            sys.exit(f"Structure image not found: {structure_path}")
    elif not args.no_structure:
        # Generate the same landmark-derived structural composite we use
        # for the InstantID pipeline's scaffold -- it's reliably accurate
        # on proportions (that's the whole point of deriving it from
        # measured face landmarks) even though its own line quality isn't
        # what we want as a final deliverable. Handing it to Gemini as the
        # thing to *polish* rather than asking Gemini to reinterpret the
        # raw photo from scratch is what should keep this from drifting
        # into a generic "AI cartoon" that merely resembles the subject.
        import composite_line_art

        print("Generating structural reference from photo...")
        structure_path = photo_path.with_name(photo_path.stem + "_structure_for_gemini.png")
        composite_line_art.composite_line_art(photo_path, structure_path, glasses=bool(want_glasses))

    print(f"Loading {len(reference_paths)} reference avatar(s)...")
    prompt = (STRUCTURE_PREAMBLE + args.prompt) if structure_path else args.prompt
    contents = [prompt]
    if structure_path:
        contents.append(Image.open(structure_path))
    contents.append(Image.open(photo_path))
    for p in reference_paths:
        contents.append(Image.open(p))

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
