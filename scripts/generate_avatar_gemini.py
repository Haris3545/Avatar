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


def cap_references(paths, max_references):
    """Cap the reference set to a small, fixed-size subset instead of
    sending every matching avatar. filter_references() already argues
    that averaging across many examples muddies specific drawing
    conventions -- the same reasoning applies to count, not just category:
    even within one category (e.g. glasses+no-beard), a dozen-plus
    examples give the model more room to average toward a generic middle
    ground than a handful of clear, consistent exemplars would. Takes the
    first N of the already-deterministically-sorted list, so the exact
    same subset is sent on every run."""
    if max_references is None or len(paths) <= max_references:
        return paths
    print(f"Capping references to the first {max_references} of {len(paths)} matches")
    return paths[:max_references]

STRUCTURE_PREAMBLE = (
    "The FIRST attached image is a structural line-art drawing generated directly from this "
    "exact photo's measured face geometry (jaw shape, eyebrow/eye/nose/mouth position, hairline, "
    "glasses shape, shoulder/collar line). Use it as the authoritative source for proportions and "
    "feature positions -- match its face shape, feature spacing, and pose exactly, do not "
    "reinterpret or re-guess the structure from the photo. Your job is to redraw/polish that exact "
    "structural drawing into finished line art -- improving its stroke confidence, hair rendering, "
    "and shading -- not to generate a fresh interpretation of the photo. "
    "Pay especially close attention to the exact jaw outline: its width, how much it tapers, and "
    "where the chin point sits -- copy it precisely from the structural image's silhouette. This is "
    "the single feature most responsible for actual recognizable likeness, and also the easiest to "
    "accidentally soften into a rounder, more generic face shape. If the structural image's jaw is "
    "narrow, keep it narrow; do not widen or round it out. "
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
    "Style rules: this is a quick, confident SKETCH, not a heavily rendered vector illustration. "
    "Every single line in the entire image is the exact same thickness, with no exceptions and no "
    "second weight anywhere -- a bold, rounded, confident stroke. Within a stroke the thickness "
    "never tapers, narrows, or comes to a point at either end; it starts and ends at the same full "
    "thickness with a rounded, not pointed, cap. Match the reference avatars' actual line weight "
    "exactly: bold and hand-drawn, never thin or machine-clean. Fully anti-aliased with no wobble "
    "or pixel jaggedness. "
    "Use the minimum number of strokes possible for every feature. When in doubt, leave a line "
    "out -- a sparse drawing is correct, not under-detailed. "
    "Keep the head a natural, moderate size relative to the shoulders -- not an oversized/cartoon "
    "head. "
    "CRITICAL: skin, face, AND NECK are bare white paper -- absolutely no fill, tint, tone, or "
    "shading of any kind, not even a light flat grey, and NO stray lines anywhere on the neck or "
    "across the cheeks/forehead. The only marks permitted anywhere on the face are the specific "
    "eyes/eyebrows/nose/mouth/ear marks described below -- nothing else, no exceptions, no "
    "cheekbone lines, no neck lines, no extra creases. "
    "Every line in the entire image, including hair, glasses, and clothing outlines, is pure solid "
    "black -- never a colored ink (no blue, no brown, nothing but black). "
    "Hair is a flat solid fill matching this specific person's actual hair darkness (near-black "
    "for dark hair, grey only if their real hair is genuinely light or grey), with just a couple of "
    "thin white gaps cut *into* the fill near the front hairline as shine/movement marks -- these "
    "are negative-space cuts in the black fill, not dark strokes drawn on top of it. No interior "
    "hatching or directional strand linework covering the hair mass, regardless of hair color. "
    "Eyes are small and simple: one small solid pupil dot, plus one short straight line directly "
    "above it (the eyelid mark) -- exactly two marks per eye, nothing more, no outline or eye "
    "shape drawn around them. "
    "Eyebrows are ONE single thick line per eyebrow -- a single stroke, not two, not a filled "
    "shape -- if the subject wears glasses, sitting close enough above the frame to almost touch "
    "its top rim. "
    "Nose is a single small curved line for the tip only -- one line, nothing else near the nose "
    "at all. No nostril marks, no philtrum mark, no second line of any kind. Absolutely no line "
    "runs upward from it toward the eyes; there must be no visual suggestion of a nose bridge. "
    "Mouth is a closed smile: one upper curve plus a shorter lower-lip line beneath it -- two "
    "lines total, nothing else near the mouth. "
    "Ears: exactly one short line inside each visible ear, nothing more. "
    "Glasses (if the subject wears them) are a continuous outlined frame with no gaps or breaks "
    "anywhere in the line -- even bolder/thicker than the already-heavy weight used for everything "
    "else, still solid black. "
    "Only draw beard/stubble texture if the subject clearly has visible facial hair in the photo, "
    "and keep it minimal and confined -- never spread dot texture across the whole lower face. If "
    "the photo shows a clean-shaven or barely-stubbled face, draw no facial hair texture at all. "
    "Clothing is a single uniform flat grey tone (a plain flat tone, not the photo's actual fabric "
    "pattern even if it's patterned) -- absolutely no second shading tone, gradient, or shadow "
    "patch anywhere on the clothing, just one flat grey shape. "
    "The collar is a proper button-up shirt collar: two straight, pointed collar flaps that meet "
    "in a V shape near the top button, with a visible gap of bare white neck showing between them "
    "and the shirt -- NOT a rounded polo or crew-neck collar, NOT a single unbroken neckline curve. "
    "Draw it as one clean pointed-collar shape (not a doubled or duplicated collar line), plus one "
    "or two small button shapes drawn as hollow ring outlines (a small circle outline, not a "
    "filled dot) down the front. "
    "Background is solid flat white. "
    "No photorealistic shading, no gradients, no dense texture anywhere -- every tone is a flat, "
    "evenly-colored shape, and the overall impression should be sparse and quick-sketched, not "
    "densely rendered or 'finished-looking'. "
    "Crop the final image very tightly: the head and shoulders should fill nearly the entire "
    "frame, with the bottom edge cutting off just below the shoulder line -- not showing the "
    "upper arms or torso -- and the top of the hair should sit close to the top edge of the "
    "image, with only a thin margin of empty space above it, not a large gap. Match the framing "
    "of the attached structural reference image, which is already cropped this way."
)

# A second, independent Gemini call that scores one generated draft against
# a short checklist of the failure modes actually observed in testing
# (thin/generic lines instead of bold, wrong collar shape, stray garbled
# marks) -- catches a bad sample automatically instead of only by eye.
# Deliberately narrow (4 checks) rather than re-litigating the whole
# prompt: this is a pass/fail gate on known failure modes, not a full
# style review.
QA_PROMPT = (
    "You are checking a generated avatar illustration against a strict house style checklist. "
    "For each numbered criterion below, answer only YES or NO, one per line, in the exact format "
    "'N: YES' or 'N: NO' with no other text before or after.\n"
    "1. Every line in the drawing is the exact same single thickness -- not two different "
    "weights, and not thin/hairline/generic-flat-vector-icon style overall.\n"
    "2. The face and neck are bare flat white, with zero shading, tint, gradient, or grey tone, "
    "AND no stray lines anywhere on the neck, cheeks, or forehead outside the eyes/eyebrows/nose/"
    "mouth/ear marks.\n"
    "3. The clothing shows a proper button-up shirt collar with two open collar points meeting "
    "near the top button -- not a polo or crew-neck collar.\n"
    "4. There are no garbled, duplicated, blurred, or nonsensical marks anywhere on the face -- "
    "every line reads as a clean, intentional stroke.\n"
    "5. Within any single stroke, thickness stays constant along its whole length with a rounded "
    "cap -- no stroke tapers, narrows, or comes to a point at either end.\n"
    "6. The nose is a single line for the tip only -- no nostril marks, no philtrum mark, no line "
    "running upward from it that could read as a nose bridge.\n"
    "7. Each eye is exactly two marks -- a solid pupil dot plus one short straight line above it -- "
    "with no outline or eye shape drawn around them.\n"
)


def qa_check(image, client, model):
    """Returns (passed: bool, checks: dict[int, bool], raw_text: str)."""
    response = client.models.generate_content(model=model, contents=[QA_PROMPT, image])
    text = "".join(part.text for part in response.candidates[0].content.parts if getattr(part, "text", None))
    checks = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        if key.isdigit():
            checks[int(key)] = val.strip().upper().startswith("Y")
    passed = bool(checks) and all(checks.values())
    return passed, checks, text


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
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Generate N independent drafts instead of one, each a fresh generation from the "
        "same photo/structure/references (not fed from each other -- feeding a draft back in as "
        "a 'refine this' step was tried and made quality worse each pass, since Gemini has no "
        "strong anchor pulling a stochastic re-interpretation back toward correctness). "
        "Independent samples don't compound errors the way sequential refinement did. Every "
        "sample is saved separately (<out>_sample1.png, _sample2.png, ...).",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=6,
        help="Cap the (category-filtered) reference set to the first N avatars instead of "
        "sending every match. A dozen-plus in-context examples give the model more room to "
        "average toward a generic middle ground than a handful of consistent exemplars do -- "
        "same reasoning filter_references() already applies to category, applied to count too. "
        "Pass 0 to disable capping and send the full filtered set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the generation seed for reproducibility while testing prompt changes. Image "
        "models don't always honor this reliably -- treat it as a best-effort nudge toward "
        "determinism, not a guarantee. Using this with --samples > 1 will likely produce "
        "identical or near-identical outputs, defeating the point of independent sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature, if supported by the model. Lower (e.g. 0.4) reduces "
        "run-to-run style variance (line weight, collar shape) at the cost of less creative "
        "range; unset uses the model's own default.",
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run a second Gemini call on each generated sample checking it against known "
        "failure modes (thin lines, wrong collar shape, garbled marks) and report pass/fail "
        "per sample. Doubles the API calls; off by default.",
    )
    args = parser.parse_args()

    if args.samples < 1:
        sys.exit("--samples must be at least 1")
    if args.seed is not None and args.samples > 1:
        print(
            "Warning: --seed with --samples > 1 will likely produce identical/near-identical "
            "outputs -- consider dropping one or the other."
        )

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
    reference_paths = cap_references(reference_paths, args.max_references if args.max_references > 0 else None)

    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first (get one at https://aistudio.google.com/apikey)")

    from google import genai
    from google.genai import types
    from PIL import Image

    client = genai.Client(api_key=api_key)
    config = None
    if args.seed is not None or args.temperature is not None:
        config = types.GenerateContentConfig(seed=args.seed, temperature=args.temperature)

    structure_path = None
    measurements_text = ""
    if args.structure_image:
        structure_path = Path(args.structure_image)
        if not structure_path.exists():
            sys.exit(f"Structure image not found: {structure_path}")
    if not args.no_structure:
        # Generate the same landmark-derived structural composite we use
        # for the InstantID pipeline's scaffold -- it's reliably accurate
        # on proportions (that's the whole point of deriving it from
        # measured face landmarks) even though its own line quality isn't
        # what we want as a final deliverable. Handing it to Gemini as the
        # thing to *polish* rather than asking Gemini to reinterpret the
        # raw photo from scratch is what should keep this from drifting
        # into a generic "AI cartoon" that merely resembles the subject.
        import composite_line_art

        if structure_path is None:
            print("Generating structural reference from photo...")
            structure_path = photo_path.with_name(photo_path.stem + "_structure_for_gemini.png")
            composite_line_art.composite_line_art(photo_path, structure_path, glasses=bool(want_glasses))

        # Proportions have so far only been communicated as an image Gemini
        # has to eyeball ("match this drawing"), which it's repeatedly
        # drifted from (jaw width especially). Explicit numeric ratios in
        # the text prompt are a harder signal to silently ignore.
        print("Measuring proportions from photo...")
        photo_im = Image.open(photo_path).convert("RGB")
        landmarks = composite_line_art.get_face_landmarks(photo_im)
        if landmarks is not None:
            foreground = composite_line_art.rembg_foreground(photo_im)
            w, h = photo_im.size
            measurements = composite_line_art.compute_measurements(landmarks, foreground, w, h)
            measurements_text = composite_line_art.format_measurements(measurements)
            if measurements_text:
                print(measurements_text)
        else:
            print("Warning: no face detected for measurements -- skipping numeric proportions.")

    def extract_image(response):
        """Returns a PIL Image, or None if this response didn't produce one
        (e.g. finish_reason=IMAGE_OTHER -- an occasional one-off failure to
        generate, not a code bug) -- printing why, so the caller can retry/
        skip instead of the whole run crashing on a single flaky call."""
        if not response.candidates or response.candidates[0].content is None:
            reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
            print(f"  No image in response (finish_reason={reason}) -- skipping this sample.")
            return None
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                from io import BytesIO

                return Image.open(BytesIO(part.inline_data.data))
        text = "".join(part.text for part in response.candidates[0].content.parts if getattr(part, "text", None))
        print(f"  No image in response -- skipping this sample. Model said:\n{text}")
        return None

    print(f"Loading {len(reference_paths)} reference avatar(s)...")
    out_path = Path(args.out)

    prompt = (STRUCTURE_PREAMBLE + args.prompt) if structure_path else args.prompt
    if measurements_text:
        prompt = prompt + "\n\n" + measurements_text
    contents = [prompt]
    if structure_path:
        contents.append(Image.open(structure_path))
    contents.append(Image.open(photo_path))
    for p in reference_paths:
        contents.append(Image.open(p))

    results = []  # (path, passed_or_None, checks_or_None)
    for i in range(1, args.samples + 1):
        print(f"Generating with {args.model} (sample {i}/{args.samples})...")
        response = client.models.generate_content(model=args.model, contents=contents, config=config)
        image = extract_image(response)
        if image is None:
            continue

        if args.samples == 1:
            sample_path = out_path
        else:
            sample_path = out_path.with_name(f"{out_path.stem}_sample{i}{out_path.suffix}")
        image.save(sample_path)
        print(f"Saved to {sample_path}")

        passed, checks, raw = (None, None, None)
        if args.qa:
            print(f"  Running QA check on sample {i}...")
            passed, checks, raw = qa_check(image, client, args.model)
            status = "PASS" if passed else "FAIL"
            print(f"  QA {status}: {checks}")
        results.append((sample_path, passed, checks))

    if not results:
        sys.exit(f"All {args.samples} sample(s) failed to generate an image -- see warnings above. Try again.")

    if args.samples > 1:
        print("\nSummary:")
        for path, passed, checks in results:
            tag = "" if passed is None else (" [QA PASS]" if passed else f" [QA FAIL {checks}]")
            print(f"  {path}{tag}")
        if args.qa:
            passing = [p for p, passed, _ in results if passed]
            if passing:
                print(f"\n{len(passing)}/{args.samples} sample(s) passed QA. Recommended: {passing[0]}")
            else:
                print("\nNo samples passed QA -- review all of them manually, none is auto-recommended.")


if __name__ == "__main__":
    main()
