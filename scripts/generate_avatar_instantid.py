#!/usr/bin/env python3
"""Generate a VCCP-style avatar using InstantID (face-embedding identity
preservation) + canny-edge ControlNet + your trained LoRA, via our patched
fork of fofr/face-to-many (haris3545/face-to-many-patched -- see
vendor/DEPLOY.md for why: the hosted original rejects our account's
replicate.delivery URLs).

InstantID conditions on a facial identity embedding from the main photo,
which is what makes it possible to also pass a separate --control-image
(e.g. scripts/composite_line_art.py's output) purely for canny-edge
structure conditioning: InstantID's identity embedding still comes from a
real photo, while the edge structure can come from an already-simplified
line-art composite instead of the photo's literal detail.

--control-image alone is soft conditioning, though -- it only ever nudges
generation toward the given structure, at any strength, and testing showed
it can't guarantee specific shape/likeness survives (weak strength: ignored;
strong strength: destroys the style instead of fixing likeness). Add --mask
for a hard guarantee instead: generation is then restricted to only the
masked region, so everything outside it renders exactly as --control-image,
pixel for pixel. Use with composite_line_art.py's --scaffold mode, which
produces a matched (scaffold image, mask) pair for exactly this.

Usage:
    export REPLICATE_API_TOKEN=r8_...
    python3 scripts/generate_avatar_instantid.py path/to/photo.jpg --lora-weights https://.../trained_model.tar
    # or, with soft structure conditioning -- use composite_line_art.py's
    # --structure mode (landmark-derived face geometry, not brightness-traced
    # detail) rather than its default full composite, which is too literal
    # for a ControlNet to condition on well:
    python3 scripts/composite_line_art.py path/to/photo.jpg structure.png --structure
    python3 scripts/generate_avatar_instantid.py path/to/photo.jpg --control-image structure.png --lora-weights https://.../trained_model.tar
    # or, for a hard guarantee that the face/hair shape survives -- use
    # --scaffold mode and pass its mask too:
    python3 scripts/composite_line_art.py path/to/photo.jpg scaffold.png --scaffold
    python3 scripts/generate_avatar_instantid.py path/to/photo.jpg --control-image scaffold.png --mask scaffold_mask.png --lora-weights https://.../trained_model.tar
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

MODEL = "haris3545/face-to-many-patched:0ce8eb4a35fb953df2f14515d5b080ba61819bb78ab0d43c6d1b99677dc75265"


def _color_dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def whiten_background(path: Path, thresh: int = 45, grid_step: int = 12) -> Image.Image:
    """Flood-fill the background to white, since prompting for a white
    background has been unreliable (the model keeps rendering a
    dark/vignetted background regardless of wording or parameters).

    Two passes:
    1. Border-seeded flood fill for the main open background.
    2. A grid scan over the whole image that flood-fills any remaining
       pixel close to the original background tone -- catches pockets of
       background enclosed by the subject's own outline (e.g. between an
       arm and the torso) that a border seed can't reach through, since
       flood fill only spreads through pixels connected to the seed.
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    bg_ref = im.getpixel((0, 0))

    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        seeds += [
            (int(w * frac), 0),
            (int(w * frac), h - 1),
            (0, int(h * frac)),
            (w - 1, int(h * frac)),
        ]
    for seed in seeds:
        if im.getpixel(seed) != (255, 255, 255):
            ImageDraw.floodfill(im, seed, (255, 255, 255), thresh=thresh)

    for y in range(0, h, grid_step):
        for x in range(0, w, grid_step):
            px = im.getpixel((x, y))
            if px != (255, 255, 255) and _color_dist(px, bg_ref) <= thresh:
                ImageDraw.floodfill(im, (x, y), (255, 255, 255), thresh=thresh)

    im.save(path)
    return im


def crop_to_shoulders(path: Path, margin_fraction: float = 0.12) -> None:
    """Crop to the subject's actual bounding box (against a white
    background), then cut off just below the shoulders -- a proper headshot
    crop, not too much torso and not cutting into the neck. A fixed height
    fraction broke across different photo framings, since how far down the
    shoulders fall relative to the head varies photo to photo. Instead, use
    the subject's own silhouette width profile: find the neck (the
    narrowest point below the head) and the shoulder line (where the
    silhouette re-widens to its full body width below that), then crop a
    small margin below the shoulder line."""
    from PIL import ImageChops
    import numpy as np

    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if not bbox:
        return
    left, top, right, bottom = bbox  # right/bottom are exclusive (PIL bbox convention)
    content_height = bottom - top
    fallback_bottom = min(top + int(content_height * 0.72), im.height)

    arr = np.array(im)
    non_white = np.any(arr < 250, axis=2)
    col_idx = np.arange(arr.shape[1])
    widths = np.zeros(arr.shape[0], dtype=int)
    for y in range(top, bottom):
        xs = col_idx[non_white[y]]
        if xs.size:
            widths[y] = xs[-1] - xs[0]

    # Only look for the neck within a plausible zone below the head's
    # widest point, so this doesn't get confused by torso/arm narrowing
    # further down the image.
    search_start = top + int(content_height * 0.35)
    search_end = min(top + int(content_height * 0.85), bottom)

    new_bottom = fallback_bottom
    if search_start < search_end:
        neck_y = search_start + int(np.argmin(widths[search_start:search_end]))
        below = widths[neck_y:bottom]
        shoulder_width = below.max() if below.size else 0
        if shoulder_width > widths[neck_y]:
            threshold = widths[neck_y] + (shoulder_width - widths[neck_y]) * 0.85
            flare_offset = int(np.argmax(below >= threshold))
            shoulder_y = neck_y + flare_offset
            new_bottom = min(
                shoulder_y + int(content_height * margin_fraction), im.height
            )

    im.crop((left, top, right, new_bottom)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", help="Path to the input photo")
    parser.add_argument(
        "--lora-weights",
        required=True,
        help="URL to your trained LoRA weights .tar (from scripts/get_lora_weights_url.py)",
    )
    parser.add_argument(
        "--control-image",
        help="Optional separate image (e.g. scripts/composite_line_art.py output) used only "
        "for canny-edge structure conditioning. The main photo is still used for InstantID's "
        "facial identity embedding, which needs real photographic features and fails on a "
        "pure line-art image. Ignored when --mask is given.",
    )
    parser.add_argument(
        "--mask",
        help="Optional inpaint mask (white = editable, black = kept pixel-identical to "
        "the scaffold). Pair with scripts/composite_line_art.py --scaffold's two outputs: "
        "pass the scaffold image itself as --control-image and its companion _mask.png here. "
        "Generation is restricted to the masked region only, guaranteeing the scaffold's "
        "measured face/hair shape survives outside it -- unlike --control-image alone, which "
        "only ever nudges the output toward a structure without guaranteeing any of it "
        "survives. --control-image is required when --mask is given (it becomes the base "
        "artwork for the protected, unmasked region).",
    )
    parser.add_argument("--out", default="avatar_out_iid.png", help="Where to save the result")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--instant-id-strength",
        type=float,
        default=0.95,
        help="how strongly facial identity is preserved, 0-1. Raised from the model's "
        "default of 0.8 -- likeness was reading as generic, and this is the one input "
        "that's actually tied to this specific face rather than style/structure.",
    )
    parser.add_argument(
        "--control-image-strength",
        type=float,
        default=0.6,
        help="how strongly the canny-edge structure (from --control-image, or the main photo "
        "if not given) is enforced, 0-1",
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
        "--prompt-strength",
        type=float,
        default=4.5,
        help="CFG scale: how strongly both the positive and negative prompt are enforced. "
        "Model default is 4.5; higher makes negative-prompt suppression (e.g. facial hair) bite harder.",
    )
    parser.add_argument(
        "--prompt",
        default="TOK style, monochrome black and white line art portrait illustration, "
        "solid pure white background (like blank white paper), plain white background, no background detail, "
        "extremely minimal linework, as few thick rounded strokes as possible, flat solid black fills, "
        "mostly flat shapes, simple graphic dot eyes, simplified cartoon facial features, no fine detail, "
        "no shading device of any kind -- no halftone, no cross-hatching, no stipple, no gradient, "
        "anywhere including beard, jaw, or hair, only flat solid fills or bare white, "
        "at most three or four flat tones total in the entire image, never more, never a texture standing in for a tone, "
        "clothing rendered as one flat solid black shape with no pattern, warm natural smile, "
        "graphic vector illustration, no photorealism, centered head and shoulders portrait crop",
    )
    parser.add_argument(
        "--negative-prompt",
        default="photo, photorealistic, color, colour, coloured, tinted, gradient background, "
        "grey background, dark background, black background, textured background, visible wall, "
        "chalkboard, vignette, "
        "sepia, muted tones, painterly, blurry, low quality, grayscale photo, "
        "visible teeth, tongue, "
        "realistic eyes, detailed iris, photorealistic skin texture, dense stippling, "
        "intricate fine detail, engraving texture, fabric texture detail, fabric pattern, "
        "floral print, patterned clothing, full body, torso, waist",
    )
    parser.add_argument(
        "--no-facial-hair",
        action="store_true",
        help="this specific photo shows no beard/stubble -- suppress the LoRA's learned tendency "
        "to render jaw shadow as facial hair. Not a permanent default: pass this per-photo based on "
        "what the photo actually shows (eventually this should be an automated visual check, not manual)",
    )
    parser.add_argument(
        "--no-whiten-background",
        action="store_true",
        help="skip the post-processing step that flood-fills the background to white "
        "(the model has been unreliable at rendering white background via prompting alone)",
    )
    args = parser.parse_args()

    if args.no_facial_hair:
        args.prompt += ", no visible facial hair, completely smooth clean shaven jawline and neck"
        args.negative_prompt += (
            ", beard, facial hair, stubble, goatee, moustache, mustache, beard shadow, "
            "beard, facial hair, stubble"  # repeated for extra negative-prompt weight
        )

    negative_prompt = args.negative_prompt

    photo_path = Path(args.photo)
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    import replicate

    print("Uploading photo...")
    with open(photo_path, "rb") as f:
        uploaded = replicate.files.create(f)
    photo_url = uploaded.urls["get"]

    predict_input = {"image": photo_url}

    if args.mask and not args.control_image:
        sys.exit("--mask requires --control-image (the scaffold it was generated from)")

    if args.control_image:
        control_path = Path(args.control_image)
        if not control_path.exists():
            sys.exit(f"Control image not found: {control_path}")
        print("Uploading control image...")
        with open(control_path, "rb") as f:
            uploaded_control = replicate.files.create(f)
        predict_input["control_image"] = uploaded_control.urls["get"]

    if args.mask:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            sys.exit(f"Mask not found: {mask_path}")
        print("Uploading mask...")
        with open(mask_path, "rb") as f:
            uploaded_mask = replicate.files.create(f)
        predict_input["mask"] = uploaded_mask.urls["get"]

    print(f"Generating with {MODEL} (InstantID + canny ControlNet + LoRA)...")
    prediction = replicate.predictions.create(
        version=MODEL,
        input={
            **predict_input,
            "style": "3D",  # required enum, but custom_lora_url overrides the actual style used
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "custom_lora_url": args.lora_weights,
            "lora_scale": args.lora_scale,
            "instant_id_strength": args.instant_id_strength,
            "control_image_strength": args.control_image_strength,
            "denoising_strength": args.denoising_strength,
            "prompt_strength": args.prompt_strength,
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

    if not args.no_whiten_background:
        print("Whitening background...")
        whiten_background(Path(args.out))
        print("Cropping to head and shoulders...")
        crop_to_shoulders(Path(args.out))

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
