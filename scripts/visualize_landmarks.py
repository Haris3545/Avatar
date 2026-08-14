#!/usr/bin/env python3
"""Overlay the mediapipe face-landmark points actually used by
composite_line_art.py directly onto the source photo, so it's possible to
SEE what geometric data the classical pipeline is working from -- rather
than inferring it indirectly from how good or bad the final scaffold
looks. Useful for judging whether a shape (e.g. the jaw or nose) reads
badly because there just aren't enough/the right landmark points defining
it, versus a rendering problem downstream of good data.

Draws:
    - yellow dots: every point in the face-oval sequence (jaw/cheek/temple
      outline) -- this is the exact point set the jaw silhouette gets
      spliced/compared against
    - cyan dots: the specific points composite_line_art.py picks out for
      the nose ("gull-wing" spline control points) and mouth (upper-lip
      spline control points)
    - magenta dots: all other detected face landmarks, small and dim, for
      general reference (mediapipe returns ~478 total)

Usage:
    python3 scripts/visualize_landmarks.py path/to/photo.jpg out.png
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import composite_line_art as cla


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 scripts/visualize_landmarks.py path/to/photo.jpg out.png")

    photo_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not photo_path.exists():
        sys.exit(f"Photo not found: {photo_path}")

    im = Image.open(photo_path).convert("RGB")
    w, h = im.size

    print("Detecting face landmarks...")
    landmarks = cla.get_face_landmarks(im)
    if landmarks is None:
        sys.exit("No face detected")

    oval_idx = set(cla._ordered_face_oval_indices())
    nose_idx = {49, 129, 98, 2, 327, 358, 279}
    mouth_idx = {61, 40, 37, 0, 267, 270, 291}
    highlighted = oval_idx | nose_idx | mouth_idx

    out = im.copy()
    draw = ImageDraw.Draw(out)
    r_small, r_big = max(1, w // 400), max(2, w // 150)

    for i, lm in enumerate(landmarks):
        x, y = lm.x * w, lm.y * h
        if i in oval_idx:
            color, r = (255, 220, 0), r_big  # yellow -- jaw/face-oval outline
        elif i in nose_idx or i in mouth_idx:
            color, r = (0, 220, 255), r_big  # cyan -- nose/mouth spline points
        else:
            color, r = (255, 0, 220), r_small  # magenta -- every other detected point
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    out.save(out_path)
    print(f"Saved to {out_path}")
    print(
        f"{len(oval_idx)} face-oval (jaw/cheek) points, {len(nose_idx)} nose points, "
        f"{len(mouth_idx)} mouth points highlighted out of {len(landmarks)} total detected."
    )


if __name__ == "__main__":
    main()
