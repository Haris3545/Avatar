#!/usr/bin/env python3
"""Prototype: derive deterministic contrast/threshold maps straight from a
photo, as building blocks for a composited avatar instead of relying
entirely on a diffusion model to hallucinate structure it may get wrong
(e.g. confusing jaw shadow for a beard). No AI involved -- pure classical
image processing, so it's free and fully repeatable.

Usage:
    python3 scripts/generate_contrast_maps.py path/to/photo.jpg [out_dir]
"""
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: generate_contrast_maps.py path/to/photo.jpg [out_dir]")

    photo_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("contrast_maps_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    im = Image.open(photo_path).convert("L")
    im.save(out_dir / "01_grayscale.png")

    autocontrast = ImageOps.autocontrast(im, cutoff=2)
    autocontrast.save(out_dir / "02_autocontrast.png")

    # Binary thresholds at a few levels -- different levels separate
    # different features (higher threshold isolates only the very darkest
    # regions like pupils/eyebrows/hair; lower threshold pulls in beard
    # shadow, jaw contours, etc.)
    for level in (60, 90, 120, 150, 180):
        binary = autocontrast.point(lambda p, l=level: 255 if p > l else 0)
        binary.save(out_dir / f"03_threshold_{level}.png")

    # Posterize into a few flat tone bands, closer to how the actual
    # avatars use discrete flat fills rather than continuous shading
    posterized = ImageOps.posterize(autocontrast, bits=2)
    posterized.save(out_dir / "04_posterize_4level.png")

    # Edge maps -- two different operators, useful for outline tracing
    edges = im.filter(ImageFilter.FIND_EDGES)
    edges.save(out_dir / "05_edges.png")

    contour = autocontrast.filter(ImageFilter.CONTOUR)
    contour.save(out_dir / "06_contour.png")

    print(f"Wrote maps to {out_dir}/")


if __name__ == "__main__":
    main()
