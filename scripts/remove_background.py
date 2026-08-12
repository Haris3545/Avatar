#!/usr/bin/env python3
"""Isolate the person from a photo's background, composited onto white.

This is a useful preprocessing step on its own (feeds a much cleaner
image into scripts/generate_contrast_maps.py -- plain edge/threshold
detection can't tell background clutter from the subject, so it needs a
clean isolated subject to work reliably on candid/busy photos), and is
also worth trying as a preprocessing step before the InstantID generation
pipeline, since depth ControlNet conditioning on a photo's real busy
background may be contributing to that pipeline's persistent
dark-background problem.

Usage:
    python3 scripts/remove_background.py path/to/photo.jpg [out.png]
"""
import sys
from pathlib import Path

from PIL import Image


def remove_background(photo_path: Path) -> Image.Image:
    from rembg import remove

    im = Image.open(photo_path)
    cutout = remove(im).convert("RGBA")
    bg = Image.new("RGB", cutout.size, (255, 255, 255))
    bg.paste(cutout, mask=cutout.split()[-1])
    return bg


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: remove_background.py path/to/photo.jpg [out.png]")

    photo_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else photo_path.with_name(
        photo_path.stem + "_nobg.png"
    )

    result = remove_background(photo_path)
    result.save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
