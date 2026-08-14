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
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODEL = "haris3545/face-to-many-patched:212d4f36ac8fd39c2e9b32a2b7279008a1216d51fa8c7aa3eae491fb1f10e846"

# vendor/TRAINING.md tracks the latest LoRA weights URL from
# scripts/train_lora.py -- pass it via --lora-weights, it's not baked in
# here since it changes independently of the generation model version above.


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


def align_style_to_control(style_path: Path, control_path: Path) -> Path:
    """Warp style_path (e.g. Gemini's free-form output) onto control_path's
    exact pixel coordinate frame before it's used as --style-image.

    Without this, the two images only share a canvas by ImageResize+ fitting
    each into the same 1024x1024 box independently -- their faces end up at
    different positions/scales within it, since Gemini composes its own
    framing rather than literally matching the scaffold pixel-for-pixel. The
    inpaint mask is built in control_image's coordinate space, so applying it
    to a style_image whose face doesn't line up produces double-exposure
    ghosting in the generated region (two differently-positioned faces
    blended together), not a style/quality problem. Fit a similarity
    transform (rotation+scale+translation, no shear) from outer-eye-corner +
    chin landmarks -- points present in both a photoreal render and Gemini's
    stylized line art -- and warp style_path into control_path's frame so the
    mask actually lines up with what it's masking.
    """
    import composite_line_art as cla
    import cv2

    control_im = Image.open(control_path).convert("RGB")
    style_im = Image.open(style_path).convert("RGB")

    control_lm = cla.get_face_landmarks(control_im)
    style_lm = cla.get_face_landmarks(style_im)
    if control_lm is None or style_lm is None:
        print(
            "Warning: couldn't detect a face in the scaffold and/or style image -- "
            "uploading style_image unaligned, which risks ghosting."
        )
        return style_path

    idxs = [33, 263, 152]  # right eye outer corner, left eye outer corner, chin
    cw, ch = control_im.size
    sw, sh = style_im.size
    dst = np.array([[control_lm[i].x * cw, control_lm[i].y * ch] for i in idxs], dtype=np.float32)
    src = np.array([[style_lm[i].x * sw, style_lm[i].y * sh] for i in idxs], dtype=np.float32)

    transform, _ = cv2.estimateAffinePartial2D(src, dst)
    warped = cv2.warpAffine(
        np.array(style_im), transform, (cw, ch),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    out_path = Path(tempfile.mkdtemp()) / f"{style_path.stem}_aligned.png"
    Image.fromarray(warped).save(out_path)
    return out_path


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
    parser.add_argument(
        "--style-image",
        help="Optional image (e.g. scripts/generate_avatar_gemini.py's output) used as the "
        "starting point for the masked interior region instead of generating it from a blank "
        "scaffold. Only used when --mask is also given -- the scaffold's own pixels outside the "
        "mask (jaw, ears, silhouette -- the actual likeness-carrying structure) are unaffected "
        "either way, so this only changes what the interior facial detail is seeded from: "
        "Gemini's polished rendering, corrected back toward the true structure/identity by "
        "ControlNet/InstantID rather than generated from scratch.",
    )
    parser.add_argument(
        "--style-denoise",
        type=float,
        default=0.85,
        help="Denoise strength for the masked region when --style-image is given, 0-1. Lower "
        "keeps more of style_image's actual rendering (style) but also more of its flaws (e.g. "
        "Gemini's tendency to render soft grey shading on the face, which the flat-style prompt "
        "then has to fight against instead of simply overwriting); higher gives the sampler more "
        "room to fully repaint over that while still following style_image's rough tone/detail "
        "distribution as a starting point rather than blank canvas. Now that style_image is "
        "aligned to the scaffold's coordinate frame before this step, higher values no longer "
        "risk the ghosting they used to when the two were misaligned.",
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
        default=7.5,
        help="CFG scale: how strongly both the positive and negative prompt are enforced. "
        "Model default is 4.5; raised to 7.5 -- at instant_id_strength=0.95 (kept high "
        "deliberately for likeness), the model was compromising between strong identity "
        "pressure and the flat-style prompt by rendering halftone/stipple shading on the "
        "face, which the negative prompt already forbids but wasn't being enforced hard "
        "enough to win that fight. Higher makes negative-prompt suppression bite harder.",
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
        "skin is pure flat white with absolutely no grey tone, no shading, and no gradient anywhere on the face or neck, "
        "clothing rendered as one flat solid black shape with no pattern, warm natural smile, "
        "graphic vector illustration, no photorealism, centered head and shoulders portrait crop",
    )
    parser.add_argument(
        "--negative-prompt",
        default="halftone dots on face, halftone dots on skin, stipple texture on face, "
        "dot pattern shading, dot texture skin, pointillism, dotted shading, noisy skin texture, "
        "smooth gradient shading on face, soft shadow on cheeks, soft shadow on forehead, "
        "grey tone on skin, grey shading on face, airbrushed shading, soft blended shadow, "
        "contoured shading, 3D shaded face, rendered lighting on skin, "
        "individual hair strands, visible individual stubble hairs, fine stubble detail, "
        "realistic beard texture, textured beard, scattered facial hair marks, "
        "photo, photorealistic, color, colour, coloured, tinted, gradient background, "
        "grey background, dark background, black background, textured background, visible wall, "
        "chalkboard, vignette, "
        "sepia, muted tones, painterly, blurry, low quality, grayscale photo, "
        "visible teeth, tongue, "
        "realistic eyes, detailed iris, photorealistic skin texture, dense stippling, "
        "intricate fine detail, engraving texture, fabric texture detail, fabric pattern, "
        "floral print, patterned clothing, full body, torso, waist, "
        "broken glasses frame, gapped glasses outline, dashed glasses lines, "
        "disconnected glasses frame, glasses frame with gaps, fragmented eyewear outline, "
        "glasses drawn as dots or dashes",
    )
    parser.add_argument(
        "--glasses",
        action="store_true",
        help="this specific photo shows the subject wearing glasses -- reinforce that the frame "
        "should render as one continuous solid black shape. The model tends to draw the frame as "
        "many small disconnected dark marks (echoing how it renders stubble) instead of a single "
        "unbroken outline, since nothing in the base prompt says a frame should be continuous.",
    )
    parser.add_argument(
        "--no-facial-hair",
        action="store_true",
        help="this specific photo shows no beard/stubble -- suppress the LoRA's learned tendency "
        "to render jaw shadow as facial hair. Not a permanent default: pass this per-photo based on "
        "what the photo actually shows (eventually this should be an automated visual check, not manual)",
    )
    parser.add_argument(
        "--whiten-background",
        action="store_true",
        help="run the post-processing step that flood-fills the background to white. Off by "
        "default: its grid-scan pass flood-fills any pixel close to the background tone "
        "anywhere in the image, not just the background region -- once the model started "
        "rendering real dot/stipple facial texture, that pass started whitening individual "
        "dots that happened to be light enough, leaving a mottled speckle behind instead of "
        "the clean texture the model actually generated.",
    )
    args = parser.parse_args()

    if args.no_facial_hair:
        args.prompt += ", no visible facial hair, completely smooth clean shaven jawline and neck"
        args.negative_prompt += (
            ", beard, facial hair, stubble, goatee, moustache, mustache, beard shadow, "
            "beard, facial hair, stubble"  # repeated for extra negative-prompt weight
        )

    if args.glasses:
        args.prompt += (
            ", wearing glasses, glasses frame drawn as one continuous solid black unbroken "
            "outline, thick solid uninterrupted glasses frame lines, single fluid stroke per "
            "lens rim with no breaks"
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

    if args.style_image and not args.mask:
        sys.exit("--style-image requires --mask (it only affects the masked interior region)")

    if args.style_image:
        style_path = Path(args.style_image)
        if not style_path.exists():
            sys.exit(f"Style image not found: {style_path}")
        print("Aligning style image to scaffold's face position/scale...")
        style_path = align_style_to_control(style_path, control_path)
        print("Uploading style image...")
        with open(style_path, "rb") as f:
            uploaded_style = replicate.files.create(f)
        predict_input["style_image"] = uploaded_style.urls["get"]
        predict_input["style_denoise"] = args.style_denoise

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

    if args.whiten_background:
        print("Whitening background...")
        whiten_background(Path(args.out))
    print("Cropping to head and shoulders...")
    crop_to_shoulders(Path(args.out))

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
