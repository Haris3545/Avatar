#!/usr/bin/env python3
"""Prepare the VCCP avatar set for SDXL LoRA training: composite onto white,
resize/crop to square, caption, and zip for upload to Replicate."""
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AVATAR_DIR = ROOT / "data" / "avatars"
STAGING_DIR = ROOT / "data" / "lora_dataset"
OUT_ZIP = ROOT / "data" / "lora_dataset.zip"

TARGET_SIZE = 1024
CAPTION = (
    "TOK style, black and white graphic icon portrait illustration. "
    "Extremely minimal linework: as few individual strokes as possible for the whole image. "
    "Every outline stroke is thick, with fully rounded ends (rounded line caps), never thin or tapering "
    "at the ends of the main outline. "
    "Eyes are constructed from just two marks: one small solid dot for the pupil, and one short curved "
    "arc above it for the top eyelid -- no other eye detail, no bottom lid, no eyelashes. "
    "Eyebrows are a single thick flat curved stroke. Nose is one short simple line or small curve. "
    "Hair is a small number of long strokes, each one tapering to a fine point at its tip, drawn over "
    "a solid flat black hair shape -- the taper is a deliberate contrast to the uniformly thick rounded "
    "outline used everywhere else on the face and body. "
    "Clothing is one flat solid black silhouette shape with a simple outlined collar, no fabric folds, "
    "no pattern, no texture -- the clothing silhouette is cropped just below the shoulders. "
    "Halftone dot shading is used only as a deliberate stylistic texture in specific areas like a beard "
    "or hair volume, never as general shading. "
    "Large, uninterrupted flat white or flat black fill areas dominate the image -- there is no "
    "cross-hatching, no gradients, and no fine surface detail anywhere. "
    "Flat white background."
)


def prep_image(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    im = im.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return im


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    avatars = sorted(f for f in AVATAR_DIR.iterdir() if f.is_file() and f.name != ".gitkeep")

    count = 0
    for f in avatars:
        try:
            im = prep_image(f)
        except Exception as e:
            print(f"skip {f.name}: {e}")
            continue
        stem = f"avatar_{count:03d}"
        im.save(STAGING_DIR / f"{stem}.png")
        (STAGING_DIR / f"{stem}.txt").write_text(CAPTION)
        count += 1

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(STAGING_DIR.iterdir()):
            zf.write(f, arcname=f.name)

    print(f"Prepared {count} training images")
    print(f"Zipped to {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
