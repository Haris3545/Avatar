#!/usr/bin/env python3
"""Thicken a generated avatar's linework into bold, rounded strokes via
deterministic morphological dilation, instead of relying on the LoRA to
produce thick lines on its own. InstantID + ControlNet conditioning has
consistently pulled generation toward thin, literal outlines regardless of
how explicitly the training captions asked for thick rounded strokes, so
this fixes stroke weight as a guaranteed post-process step instead of
another training gamble.

Any sufficiently dark pixel (outlines, fine facial detail, solid black
fills) is treated as "ink" and grown outward with a circular structuring
element, which both thickens thin strokes and naturally rounds their end
caps and corners (a disk kernel can't produce sharp corners). Solid fills
just grow their boundary by a few pixels, which is harmless at these
resolutions.

Note: this thickens every dark pixel uniformly, including fine traced
detail like beard/stubble stipple -- it makes existing dots bigger, it
doesn't turn them into a proper halftone pattern. That's a separate,
follow-up problem.

Usage:
    python3 scripts/thicken_strokes.py avatar_out_iid.png [out.png] [--thickness 3] [--dark-threshold 100]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x**2 + y**2 <= radius**2


def thicken_strokes(
    path: Path, out_path: Path, thickness: int = 3, dark_threshold: int = 100
) -> None:
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    gray = np.array(im.convert("L"))

    ink = gray < dark_threshold
    thickened = ndimage.binary_dilation(ink, structure=disk(thickness))

    out = arr.copy()
    out[thickened] = (0, 0, 0)
    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to the generated avatar image")
    parser.add_argument(
        "out", nargs="?", help="Where to save the result (defaults to <image>_thick.png)"
    )
    parser.add_argument(
        "--thickness", type=int, default=3, help="dilation radius in pixels"
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=100,
        help="grayscale value below which a pixel counts as ink",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = (
        Path(args.out) if args.out else image_path.with_name(image_path.stem + "_thick.png")
    )
    thicken_strokes(
        image_path, out_path, thickness=args.thickness, dark_threshold=args.dark_threshold
    )


if __name__ == "__main__":
    main()
