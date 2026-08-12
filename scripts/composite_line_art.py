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

Usage:
    python3 scripts/composite_line_art.py path/to/photo.jpg [out.png]
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_CACHE = Path.home() / ".cache" / "avatar_models" / "selfie_multiclass_256x256.tflite"

# Category indices from MediaPipe's multiclass selfie segmenter
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = range(6)


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


def composite_line_art(photo_path: Path, out_path: Path):
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)

    gray = np.array(im.convert("L")).astype(np.float64)
    foreground = cat_mask != BACKGROUND
    skin_mask = (cat_mask == FACE_SKIN) | (cat_mask == BODY_SKIN)

    fg_vals = gray[foreground]
    lo, hi = np.percentile(fg_vals, [2, 98]) if fg_vals.size else (0, 255)
    gray_norm = np.clip((gray - lo) / max(hi - lo, 1) * 255, 0, 255)
    dark_mask = skin_mask & (gray_norm < 120)

    out = np.full(gray.shape + (3,), 255, dtype=np.uint8)  # white canvas
    out[cat_mask == HAIR] = (0, 0, 0)
    out[cat_mask == CLOTHES] = (0, 0, 0)
    out[cat_mask == OTHER] = (0, 0, 0)

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

    # Outer silhouette outline
    eroded = ndimage.binary_erosion(foreground, iterations=4)
    outline = foreground & ~eroded
    out[outline] = (0, 0, 0)

    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: composite_line_art.py path/to/photo.jpg [out.png]")

    photo_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else photo_path.with_name(
        photo_path.stem + "_lineart.png"
    )
    composite_line_art(photo_path, out_path)


if __name__ == "__main__":
    main()
