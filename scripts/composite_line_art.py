#!/usr/bin/env python3
"""Composite a line-art base from a photo using classical image processing
+ a small semantic segmentation model -- a deterministic alternative to
the generation pipeline for the structural parts of the avatar.

Pipeline:
1. Segment the photo into background / hair / face-skin / body-skin /
   clothes / other using MediaPipe's multiclass selfie segmentation model.
   This replaces an earlier version of this script that used brightness
   heuristics to guess hair vs skin vs clothing -- that broke completely
   on blonde/light hair, since skin and light hair are tonally similar
   and no threshold can separate them. Semantic segmentation classifies
   by learned features, not raw brightness, so it works regardless of
   hair color.
2. Hair and clothes become flat black fills. Skin stays white.
3. Within skin regions, a brightness threshold picks out fine dark detail
   (eyebrows, glasses, pupils, beard texture) as thin linework rather
   than flattening it -- this part is still a real, deterministic
   distinction (not a learned bias), so it doesn't have the diffusion
   pipeline's problem of hallucinating a full beard from stray dark
   pixels.
4. The foreground/background split from the same segmentation model
   provides the outer silhouette outline.

Pass --structure for a second, minimal mode meant as generation-model
conditioning input (scripts/generate_avatar_instantid.py's --control-image)
rather than a finished composite: same silhouette/hair/clothes fill, but
facial detail comes from face-landmark geometry (jaw/face shape, eyebrows,
nose) instead of brightness-traced texture, so it encodes this face's actual
proportions without dictating beard/skin surface detail for a ControlNet to
force verbatim into the output.

Usage:
    python3 scripts/composite_line_art.py path/to/photo.jpg [out.png]
    python3 scripts/composite_line_art.py path/to/photo.jpg [out.png] --structure
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_CACHE = Path.home() / ".cache" / "avatar_models" / "selfie_multiclass_256x256.tflite"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_MODEL_CACHE = Path.home() / ".cache" / "avatar_models" / "face_landmarker.task"

# MediaPipe face mesh landmark indices
EYES = [
    {"corners": (33, 133), "iris": 468},  # subject's right eye
    {"corners": (362, 263), "iris": 473},  # subject's left eye
]

# Category indices from MediaPipe's multiclass selfie segmenter
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2


def get_model_path() -> Path:
    if not MODEL_CACHE.exists():
        print("Downloading segmentation model (one-time, ~16MB)...")
        MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
    return MODEL_CACHE


def segment(im: Image.Image) -> np.ndarray:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(get_model_path()))
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(im.convert("RGB")))
        result = segmenter.segment(mp_image)
        return np.squeeze(result.category_mask.numpy_view())


def get_face_model_path() -> Path:
    if not FACE_MODEL_CACHE.exists():
        print("Downloading face landmark model (one-time, ~4MB)...")
        FACE_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_CACHE)
    return FACE_MODEL_CACHE


def get_face_landmarks(im: Image.Image):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(get_face_model_path()))
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(im.convert("RGB")))
        result = detector.detect(mp_image)
        return result.face_landmarks[0] if result.face_landmarks else None


def draw_dot_eyes(out: np.ndarray, landmarks, w: int, h: int) -> np.ndarray:
    """Replace traced eye detail with the house style's extreme
    simplification: a solid dot for the pupil and a single curved arc for
    the upper eyelid, positioned and scaled from real landmark geometry
    (not guessed) so it still lines up with the actual face."""
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)

    for eye in EYES:
        c1, c2 = eye["corners"]
        p1 = np.array([landmarks[c1].x * w, landmarks[c1].y * h])
        p2 = np.array([landmarks[c2].x * w, landmarks[c2].y * h])
        iris = landmarks[eye["iris"]]
        cx, cy = iris.x * w, iris.y * h
        eye_width = np.linalg.norm(p2 - p1)

        # Clear a generous region around the eye back to white first, to
        # erase whatever traced detail (eyelid lines, eyelashes) was there.
        clear_r = eye_width * 0.75
        draw.ellipse([cx - clear_r, cy - clear_r * 0.7, cx + clear_r, cy + clear_r * 0.7], fill=(255, 255, 255))

        # Pupil: solid dot
        pupil_r = max(eye_width * 0.13, 3)
        draw.ellipse([cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r], fill=(0, 0, 0))

        # Upper eyelid: single arc spanning the eye corners, curving
        # upward above the pupil
        arc_left = min(p1[0], p2[0]) - eye_width * 0.08
        arc_right = max(p1[0], p2[0]) + eye_width * 0.08
        arc_top = cy - eye_width * 0.45
        arc_bottom = cy + eye_width * 0.15
        stroke_w = max(int(eye_width * 0.09), 2)
        draw.arc([arc_left, arc_top, arc_right, arc_bottom], start=200, end=340, fill=(0, 0, 0), width=stroke_w)

    return np.array(img)


def _base_layers(im: Image.Image, cat_mask: np.ndarray):
    """Shared groundwork for both composite modes: a white canvas with flat
    black hair/clothes fills and a thick rounded outer silhouette outline,
    plus the raw foreground mask for callers that need it. This part is
    already fairly abstract (flat fills, no strand-level texture), so it's
    fine to condition generation on -- the fine facial detail each mode adds
    on top is where the two modes diverge."""
    gray = np.array(im.convert("L")).astype(np.float64)
    foreground = cat_mask != BACKGROUND

    # Round every shape's edges/corners to match the house style's thick,
    # rounded-cap strokes instead of raw pixel-jagged boundaries: a
    # closing (dilate then erode) rounds concave corners and smooths
    # jagged edges, an opening (erode then dilate) rounds convex corners
    # and clips small spurs. Both use a disk structuring element so the
    # rounding is actually circular, not the diamond shape a default
    # cross-shaped structure would give.
    hair_clothes = (cat_mask == HAIR) | (cat_mask == CLOTHES) | (cat_mask == OTHER)
    kernel = disk(6)
    hair_clothes = ndimage.binary_closing(hair_clothes, structure=kernel)
    hair_clothes = ndimage.binary_opening(hair_clothes, structure=kernel)

    out = np.full(gray.shape + (3,), 255, dtype=np.uint8)  # white canvas
    out[hair_clothes] = (0, 0, 0)

    # Outer silhouette outline, thick with rounded caps/corners
    eroded = ndimage.binary_erosion(foreground, structure=disk(7))
    outline = foreground & ~eroded
    out[outline] = (0, 0, 0)

    return out, foreground, gray


def composite_line_art(photo_path: Path, out_path: Path):
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)
    out, foreground, gray = _base_layers(im, cat_mask)

    skin_mask = (cat_mask == FACE_SKIN) | (cat_mask == BODY_SKIN)
    fg_vals = gray[foreground]
    lo, hi = np.percentile(fg_vals, [2, 98]) if fg_vals.size else (0, 255)
    gray_norm = np.clip((gray - lo) / max(hi - lo, 1) * 255, 0, 255)
    dark_mask = skin_mask & (gray_norm < 120)

    # Fine facial/skin detail (eyebrows, glasses, pupils, beard texture):
    # small dark blobs within skin only. Big ones (e.g. sunglasses) still
    # become solid fills rather than noisy detail.
    fg_area = foreground.sum()
    big_blob_thresh = fg_area * 0.015
    labeled, num = ndimage.label(dark_mask)
    for i in range(1, num + 1):
        blob = labeled == i
        area = blob.sum()
        out[blob] = (0, 0, 0) if area >= big_blob_thresh else (60, 60, 60)

    print("Detecting face landmarks for dot eyes...")
    landmarks = get_face_landmarks(im)
    if landmarks is not None:
        h, w = gray.shape
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping dot-eye replacement")

    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def draw_face_structure_lines(out: np.ndarray, landmarks, w: int, h: int) -> np.ndarray:
    """Draw only geometric likeness cues -- face/jaw shape, eyebrow shape and
    position, nose bridge and width -- as thin lines from real landmark
    positions, instead of tracing brightness/texture from the photo. This is
    for feeding a generation model as structure conditioning: a brightness
    threshold picks up beard shadow, stubble, and skin texture as literal
    detail (which a ControlNet then forces the output to reproduce almost
    pixel-for-pixel, defeating the point of asking for a simplified style),
    but these landmark connections only encode this specific face's actual
    proportions, so they guide likeness without dictating surface detail."""
    from mediapipe.tasks.python import vision

    connections = vision.FaceLandmarksConnections
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)

    def draw_conns(conns, width):
        for conn in conns:
            p1, p2 = landmarks[conn.start], landmarks[conn.end]
            draw.line(
                [(p1.x * w, p1.y * h), (p2.x * w, p2.y * h)],
                fill=(0, 0, 0),
                width=width,
            )

    draw_conns(connections.FACE_LANDMARKS_FACE_OVAL, width=2)
    draw_conns(connections.FACE_LANDMARKS_LEFT_EYEBROW, width=2)
    draw_conns(connections.FACE_LANDMARKS_RIGHT_EYEBROW, width=2)
    draw_conns(connections.FACE_LANDMARKS_NOSE, width=2)

    return np.array(img)


def structure_composite(photo_path: Path, out_path: Path):
    """A minimal composite meant for feeding a generation model as structure
    conditioning (unlike composite_line_art's fuller composite, which is
    meant to look finished on its own). Keeps the silhouette and flat
    hair/clothes fill for framing, but replaces all facial detail with
    landmark-derived geometry instead of brightness-traced texture."""
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)
    out, foreground, gray = _base_layers(im, cat_mask)

    print("Detecting face landmarks for face structure...")
    landmarks = get_face_landmarks(im)
    if landmarks is not None:
        h, w = gray.shape
        out = draw_face_structure_lines(out, landmarks, w, h)
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping face structure lines")

    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: composite_line_art.py path/to/photo.jpg [out.png] [--structure]"
        )

    structure_mode = "--structure" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--structure"]

    photo_path = Path(args[0])
    default_suffix = "_structure" if structure_mode else "_lineart"
    out_path = Path(args[1]) if len(args) > 1 else photo_path.with_name(
        photo_path.stem + default_suffix + ".png"
    )

    if structure_mode:
        structure_composite(photo_path, out_path)
    else:
        composite_line_art(photo_path, out_path)


if __name__ == "__main__":
    main()
