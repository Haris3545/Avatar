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

from PIL import Image, ImageDraw

MODEL = "haris3545/face-to-many-patched:2f26886c521b71658dfaa2b71a8f116b8626a1d9f7e641dd0025f966056ee5dc"


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


def crop_to_shoulders(path: Path, keep_fraction: float = 0.72) -> None:
    """Crop to the subject's actual bounding box (against a white
    background) and cut off the bottom portion, since the model has been
    consistently framing too much torso and too much empty headroom above
    the head instead of a tight head-and-shoulders crop."""
    from PIL import ImageChops

    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if not bbox:
        return
    left, top, right, bottom = bbox
    content_height = bottom - top
    new_bottom = min(top + int(content_height * keep_fraction), im.height)
    im.crop((left, top, right, new_bottom)).save(path)


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
        "flat solid black fills, minimal occasional cross-hatching only, mostly flat shapes, "
        "simple graphic dot eyes, simplified cartoon facial features, no fine detail, "
        "no gradients, halftone dot texture as a stylistic shading device on areas like beard, "
        "jaw shadow, or hair volume -- matching the reference avatar art style, not the photo's texture, "
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
