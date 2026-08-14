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

# Every stroke in the house style reads as "the same pen" -- one bold weight
# for the outer silhouette/hair/jaw, one slightly lighter but still bold
# weight for interior facial detail (eyebrows, nose, mouth, eyes, glasses).
# Centralized here instead of a magic number at each call site so the two
# bands stay in sync as the style gets tuned.
OUTLINE_WIDTH = 5
DETAIL_WIDTH = 3
# The widths above were tuned by eye against a ~900px-tall photo. Source
# headshots range from 200px thumbnails to 1300px photos, and a fixed pixel
# width doesn't track that: on a small photo the face (and especially small
# features like the nose/mouth curves) shrinks, but an unscaled stroke
# doesn't, so it overwhelms the shape it's supposed to trace and reads as a
# solid blob instead of a thin mark. Every width is scaled by this factor,
# computed once per photo from its actual pixel dimensions.
WIDTH_REFERENCE_PX = 900

CLOTHES_FILL = (55, 55, 55)
CLOTHES_SHADOW_FILL = (35, 35, 35)
# Default/fallback only -- _base_layers computes an actual hair fill per
# photo from the subject's real measured hair darkness (see hair_fill in
# _base_layers). No reference avatar uses a universal tone or dense
# interior hatching regardless of the subject's real hair color; this
# constant only matters if hair_only ends up empty.
HAIR_FILL = (30, 30, 30)


def stroke_scale(w: int, h: int) -> float:
    return min(w, h) / WIDTH_REFERENCE_PX


def crop_to_content(out: np.ndarray, foreground: np.ndarray, margin_frac: float = 0.08) -> tuple:
    """Crop the canvas to the subject's own bounding box plus a margin --
    trimming excess headroom above the head -- and then to just below the
    shoulders, instead of leaving whatever headroom/torso framing the
    source photo happened to have. The reference avatars crop tight and
    bold, head-and-shoulders only, filling most of the frame.

    The shoulder cut uses the subject's own silhouette width profile
    (same approach as generate_avatar_instantid.py's crop_to_shoulders):
    find the neck (the narrowest point below the head), then the
    shoulder line (where the silhouette widens back out to full body
    width below that), and crop a small margin below it. A fixed height
    fraction doesn't work here since how far down the shoulders fall
    relative to the head varies photo to photo.

    Returns the crop bounds too, so a caller with a second same-size
    array (scaffold_composite's mask) can apply the exact same crop and
    stay pixel-aligned."""
    ys, xs = np.where(foreground)
    if ys.size == 0:
        return out, (0, out.shape[0], 0, out.shape[1])
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    content_height = y1 - y0

    any_row = foreground.any(axis=1)
    first_idx = foreground.argmax(axis=1)
    last_idx = foreground.shape[1] - 1 - foreground[:, ::-1].argmax(axis=1)
    widths = np.where(any_row, last_idx - first_idx, 0)

    search_start = y0 + int(content_height * 0.35)
    search_end = min(y0 + int(content_height * 0.85), y1)
    fallback_bottom = min(y0 + int(content_height * 0.72), out.shape[0] - 1)
    shoulder_bottom = fallback_bottom
    if search_start < search_end:
        neck_y = search_start + int(np.argmin(widths[search_start:search_end]))
        below = widths[neck_y : y1 + 1]
        shoulder_width = below.max() if below.size else 0
        if shoulder_width > widths[neck_y]:
            threshold = widths[neck_y] + (shoulder_width - widths[neck_y]) * 0.85
            candidates = np.where(below >= threshold)[0]
            if candidates.size:
                shoulder_y = neck_y + int(candidates[0])
                shoulder_bottom = min(shoulder_y + int(content_height * margin_frac), out.shape[0] - 1)
    y1 = min(shoulder_bottom, y1)

    my = int((y1 - y0) * margin_frac)
    mx = int((x1 - x0) * margin_frac)
    top = max(y0 - my, 0)
    bottom = min(y1 + my, out.shape[0])
    left = max(x0 - mx, 0)
    right = min(x1 + mx, out.shape[1])
    return out[top:bottom, left:right], (top, bottom, left, right)


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
        # .numpy_view() is a non-owning view into the segmenter's internal
        # C++ buffer, only valid while `result` (and the `with` block) are
        # alive. Reading it immediately after usually still works because
        # the freed memory hasn't been reused yet, but any heavy allocation
        # afterward (e.g. rembg's model) can silently overwrite it, making
        # later reads of the returned array return garbage. .copy() forces
        # an owned array that survives independently.
        return np.squeeze(result.category_mask.numpy_view()).copy()


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
    (not guessed) so it still lines up with the actual face.

    Every mark here is drawn through the supersample+Lanczos helpers
    (draw_smooth_dot / draw_smooth_open_stroke) instead of plain
    ImageDraw calls -- PIL's native ellipse/arc/line drawing has no
    anti-aliasing, and eyes are small enough that the difference is
    visible next to the rest of the face, which already gets this
    treatment."""
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
        # This is a plain fill with nothing drawn right at its own edge, so
        # it doesn't need anti-aliasing.
        clear_r = eye_width * 0.75
        draw.ellipse([cx - clear_r, cy - clear_r * 0.7, cx + clear_r, cy + clear_r * 0.7], fill=(255, 255, 255))

        # Pupil: solid dot -- small and subtle (shrunk from an earlier,
        # noticeably larger version that read as cartoonish/doll-eyed
        # next to the confirmed reference style's small, easy-to-miss eyes).
        pupil_r = max(eye_width * 0.08, 2)
        img = draw_smooth_dot(img, cx, cy, pupil_r)

        # Upper eyelid: a short, thick horizontal tick directly above the
        # pupil -- not a long thin arc spanning the whole eye corner-to-
        # corner (an earlier version). The confirmed reference style
        # marks the eyelid with a small, noticeably thick stroke right
        # over the eyeball, distinct in both length and weight from the
        # thin eyebrow further above it.
        lid_half_w = eye_width * 0.16
        lid_y = cy - eye_width * 0.22
        lid_w = max(eye_width * 0.18, 2)
        img = draw_smooth_open_stroke(img, [(cx - lid_half_w, lid_y), (cx + lid_half_w, lid_y)], width=lid_w)

    return np.array(img)


def _ordered_face_oval_indices():
    """FACE_LANDMARKS_FACE_OVAL is a set of disjoint (start, end) segment
    pairs that together form one loop, not a pre-ordered walk around it --
    reconstruct the walk by following each point's two neighbors."""
    from mediapipe.tasks.python import vision

    connections = vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    adjacency = {}
    for c in connections:
        adjacency.setdefault(c.start, []).append(c.end)
        adjacency.setdefault(c.end, []).append(c.start)
    start = next(iter(adjacency))
    order = [start]
    prev, cur = None, start
    while True:
        neighbors = [n for n in adjacency[cur] if n != prev]
        if not neighbors:
            break
        nxt = neighbors[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def compute_measurements(landmarks, foreground: np.ndarray, w: int, h: int) -> dict:
    """Extract actual numeric proportions for this specific person, all
    normalized to the distance between the pupils (a stable,
    subject-independent unit -- unlike absolute pixel distances, which
    depend on how close the camera was). Structure has so far only been
    communicated to Gemini as an image ("match this drawing's
    proportions"), which it has to eyeball and has repeatedly drifted
    from (jaw width, in particular). Explicit numeric ratios in the text
    prompt are a much harder signal to silently ignore than a visual
    reference.

    Anchored on the same interpupillary distance (landmarks 33/263) used
    elsewhere in this file as eye_span, so these ratios are consistent
    with any other landmark-derived measurement already in use."""
    eye_span = float(np.linalg.norm(
        np.array([landmarks[263].x * w, landmarks[263].y * h])
        - np.array([landmarks[33].x * w, landmarks[33].y * h])
    ))
    if eye_span <= 0:
        return {}

    oval_order = _ordered_face_oval_indices()
    oval_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in oval_order])
    eye_y = (landmarks[33].y + landmarks[263].y) / 2 * h
    chin_y = landmarks[152].y * h

    face_width = float(oval_pts[:, 0].max() - oval_pts[:, 0].min())
    face_height = float(chin_y - oval_pts[:, 1].min())

    jaw_pts = oval_pts[oval_pts[:, 1] > eye_y]
    jaw_width = float(jaw_pts[:, 0].max() - jaw_pts[:, 0].min()) if len(jaw_pts) else None

    # Same neck/shoulder width-profile approach as crop_to_content, run
    # here on the un-cropped foreground so shoulder_width and collar_drop
    # are in the same original-photo coordinate space as the landmarks.
    ys, xs = np.where(foreground)
    shoulder_width = collar_drop = None
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        content_height = y1 - y0
        any_row = foreground.any(axis=1)
        first_idx = foreground.argmax(axis=1)
        last_idx = foreground.shape[1] - 1 - foreground[:, ::-1].argmax(axis=1)
        widths = np.where(any_row, last_idx - first_idx, 0)

        search_start = y0 + int(content_height * 0.35)
        search_end = min(y0 + int(content_height * 0.85), y1)
        if search_start < search_end:
            neck_y = search_start + int(np.argmin(widths[search_start:search_end]))
            below = widths[neck_y : y1 + 1]
            sw = below.max() if below.size else 0
            if sw > widths[neck_y]:
                threshold = widths[neck_y] + (sw - widths[neck_y]) * 0.85
                candidates = np.where(below >= threshold)[0]
                if candidates.size:
                    shoulder_y = neck_y + int(candidates[0])
                    shoulder_width = float(sw)
                    collar_drop = float(shoulder_y - chin_y)

    measurements = {"face_width": face_width, "face_height": face_height, "jaw_width": jaw_width}
    if shoulder_width is not None:
        measurements["shoulder_width"] = shoulder_width
        measurements["collar_drop"] = collar_drop

    return {k: round(v / eye_span, 2) for k, v in measurements.items() if v is not None}


def format_measurements(measurements: dict) -> str:
    """Render compute_measurements()'s output as the text block appended
    to the Gemini prompt."""
    if not measurements:
        return ""
    labels = {
        "face_width": "face width (cheek to cheek)",
        "face_height": "face height (forehead to chin)",
        "jaw_width": "jaw width",
        "shoulder_width": "shoulder width",
        "collar_drop": "collar drop (chin to collar line)",
    }
    lines = [
        f"- {labels[k]} = {v} units"
        for k, v in measurements.items()
        if k in labels
    ]
    return (
        "Measured proportions for this specific person, in units of their own "
        "interpupillary distance (the distance between their pupils = 1.00 unit exactly):\n"
        + "\n".join(lines)
        + "\nTreat these as strict numeric targets, not just a visual approximation to eyeball "
        "from the structural image -- if the drawing's proportions don't match these ratios, "
        "the drawing is wrong even if it looks plausible on its own."
    )


def smooth_contours(mask: np.ndarray, epsilon_frac: float = 0.0004, samples: int = 400, min_area_frac: float = 0.002, include_holes: bool = False, smoothing: float = 0.15):
    """Fit a smooth closed curve through each significant contour of a mask
    (hair and clothes are frequently two disconnected blobs, split by a
    visible neck), instead of using the mask's raw pixel-jagged boundary
    directly. cv2.approxPolyDP strips pixel-level jitter while keeping this
    specific mask's actual shape (not a generic template), then a periodic
    spline through those points gives a fluid curve rather than a polygon
    of straight segments.

    With include_holes=True, returns (outer_contours, hole_contours)
    instead of a flat list -- needed whenever the mask can have a hole in
    it (e.g. hair that fully rings a visible face, like long hair framing
    both sides of it), since filling only the external contour would
    ignore the hole and paint solid straight over the face inside it."""
    import cv2
    from scipy.interpolate import splev, splprep

    mask_u8 = mask.astype(np.uint8) * 255
    mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(mask_u8, mode, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return ([], []) if include_holes else []
    min_area = mask.size * min_area_frac

    def smooth_one(contour):
        contour = contour.astype(np.float32)
        if cv2.contourArea(contour) < min_area or contour.shape[0] < 8:
            return None

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_frac * peri, True).squeeze(1)
        if approx.shape[0] < 4:
            return None

        x, y = approx[:, 0].astype(np.float64), approx[:, 1].astype(np.float64)
        try:
            # `smoothing` trades real shape detail against noise removal --
            # kept light by default so real concave/convex features (a jaw
            # angle, a chin point) survive, but callers with a genuinely
            # noisy boundary (e.g. dark hair against a dark, textured
            # background, where segmentation itself is less certain
            # pixel-to-pixel) can turn it up.
            tck, _ = splprep([x, y], s=len(x) * smoothing, per=True)
            u = np.linspace(0, 1, samples)
            xs, ys = splev(u, tck)
            return np.stack([xs, ys], axis=1)
        except Exception:
            return approx

    if not include_holes:
        return [s for c in contours if (s := smooth_one(c)) is not None]

    outer, holes = [], []
    for idx, contour in enumerate(contours):
        smoothed = smooth_one(contour)
        if smoothed is None:
            continue
        # hierarchy[0][idx] = (next, previous, first_child, parent);
        # parent == -1 means top-level (outer), anything else is a hole.
        parent = hierarchy[0][idx][3]
        (outer if parent == -1 else holes).append(smoothed)
    return outer, holes


def draw_smooth_strokes(canvas: Image.Image, contours, width: int = 4, supersample: int = 6) -> Image.Image:
    """Render each closed point path as a single anti-aliased stroke with
    rounded joins, by drawing it oversized on a supersampled layer and
    downsampling with a high-quality filter -- PIL's native line drawing has
    no anti-aliasing, which is exactly what makes a raw-mask outline look
    low-resolution and stair-stepped instead of a fluid drawn line."""
    if not contours:
        return canvas

    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    stroke_w = width * supersample
    r = stroke_w / 2
    for points in contours:
        scaled = [(px * supersample, py * supersample) for px, py in points]
        closed = scaled + [scaled[0]]
        draw.line(closed, fill=(0, 0, 0, 255), width=stroke_w, joint="curve")
        for px, py in scaled:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 0, 0, 255))

    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def draw_tapered_stroke(canvas: Image.Image, points, mid_width: float, end_width: float, supersample: int = 6) -> Image.Image:
    """Draw a single open stroke whose width tapers from end_width at each
    tip up to mid_width at its center, like a brush stroke -- unlike
    draw_smooth_strokes' constant width, this is for facial marks (an
    eyebrow) that read as hand-drawn precisely because they're thicker in
    the middle and thin out at the ends, not a uniform-diameter line.
    Approximated as a chain of overlapping circles sized per-point along
    the path, supersampled and downsampled for anti-aliasing like the
    other stroke helpers here."""
    if len(points) < 2:
        return canvas

    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    n = len(points)
    for i, (px, py) in enumerate(points):
        # Triangular taper: 0 at both ends, 1 at the midpoint.
        t = i / (n - 1)
        taper = 1 - abs(t - 0.5) * 2
        radius = (end_width + (mid_width - end_width) * taper) / 2 * supersample
        sx, sy = px * supersample, py * supersample
        draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=(0, 0, 0, 255))
        if i > 0:
            px0, py0 = points[i - 1]
            draw.line([(px0 * supersample, py0 * supersample), (sx, sy)], fill=(0, 0, 0, 255), width=max(int(radius * 1.6), 1))

    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def draw_smooth_open_stroke(canvas: Image.Image, points, width: float, fill=(0, 0, 0), supersample: int = 6) -> Image.Image:
    """Anti-aliased constant-width open stroke -- the open-path analog of
    draw_smooth_strokes (which always closes the path back to its start).
    Used for the nose/mouth/eye marks, which were previously drawn with
    plain ImageDraw.line/arc/ellipse calls straight onto the full-resolution
    canvas -- PIL's native drawing has no anti-aliasing, so those specific
    features looked visibly more jagged than everything else (the
    silhouette, hair, eyebrows, glasses) which already got this same
    supersample+Lanczos treatment."""
    if len(points) < 2:
        return canvas

    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    stroke_w = max(1, int(width * supersample))
    r = stroke_w / 2
    scaled = [(px * supersample, py * supersample) for px, py in points]
    draw.line(scaled, fill=fill + (255,), width=stroke_w, joint="curve")
    for px, py in (scaled[0], scaled[-1]):
        draw.ellipse([px - r, py - r, px + r, py + r], fill=fill + (255,))

    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def draw_smooth_dot(canvas: Image.Image, cx: float, cy: float, radius: float, fill=(0, 0, 0), supersample: int = 6) -> Image.Image:
    """Anti-aliased filled circle -- same rationale as draw_smooth_open_stroke,
    for the pupil dot."""
    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    r = radius * supersample
    sx, sy = cx * supersample, cy * supersample
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=fill + (255,))
    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def draw_pointed_stroke(canvas: Image.Image, points, base_width: float, fill=(0, 0, 0), supersample: int = 6) -> Image.Image:
    """Draw an open stroke that starts at base_width and tapers linearly to
    a fine point at its last point -- for a hair strand escaping the solid
    fill (thick where it leaves the scalp, tapering to nothing at the tip),
    unlike draw_tapered_stroke's symmetric thick-middle taper for eyebrows."""
    if len(points) < 2:
        return canvas

    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    n = len(points)
    for i, (px, py) in enumerate(points):
        t = i / (n - 1)
        radius = base_width / 2 * (1 - t) * supersample
        sx, sy = px * supersample, py * supersample
        draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=fill + (255,))
        if i > 0:
            px0, py0 = points[i - 1]
            draw.line([(px0 * supersample, py0 * supersample), (sx, sy)], fill=fill + (255,), width=max(int(radius * 1.6), 1))

    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def jag_fringe(contour: np.ndarray, n_teeth: int = 6, depth_frac: float = 0.012) -> np.ndarray:
    """Perturb the top band of a hair contour with a few deliberate pointed
    teeth -- a fixed-frequency sine ripple, not random noise, so it reads
    as a deliberate tufted/jagged fringe edge (matching the reference
    avatars) instead of the smooth, plain curve smooth_contours would
    otherwise produce for hair like it does for everything else. Depth
    reduced from an earlier 0.035 -- a direct correction against a real
    photo traced a smooth hairline with no visible jag at that depth, only
    a subtle one."""
    top_y = contour[:, 1].min()
    bbox_h = contour[:, 1].max() - top_y
    band = contour[:, 1] < top_y + bbox_h * 0.15
    if not band.any():
        return contour
    depth = bbox_h * depth_frac
    xs = contour[band, 0]
    span = max(xs.max() - xs.min(), 1)
    phase = (xs - xs.min()) / span
    teeth = np.abs(np.sin(np.pi * n_teeth * phase))
    result = contour.copy()
    result[band, 1] = result[band, 1] - depth * teeth
    return result


    return canvas


def draw_hair_strands(canvas: Image.Image, hair_outer_contours, detail_width: float = DETAIL_WIDTH, n_strands: int = 3) -> Image.Image:
    """A small number of thin white shine/highlight streaks cut into the
    solid hair fill near the fringe -- negative-space gaps, not strands
    drawn on top. Matches the confirmed reference style: a few short white
    cuts breaking the crown edge read as movement/shine, unlike an earlier
    version of this that drew dark tapered strokes escaping past the
    silhouette edge, which doesn't match. Each streak starts at the
    fringe/crown boundary and cuts a short way *into* the fill (toward
    increasing y, since that boundary is the hair's top edge), not beyond
    it."""
    if not hair_outer_contours:
        return canvas

    contour = max(
        hair_outer_contours,
        key=lambda c: (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min()),
    )
    top_y = contour[:, 1].min()
    bbox_h = contour[:, 1].max() - top_y
    cx = contour[:, 0].mean()

    near_top = contour[contour[:, 1] < top_y + bbox_h * 0.12]
    if len(near_top) < n_strands:
        return canvas
    near_top = near_top[np.argsort(near_top[:, 0])]
    idxs = np.linspace(0, len(near_top) - 1, n_strands).astype(int)

    for idx in idxs:
        bx, by = near_top[idx]
        dx = 1.0 if bx >= cx else -1.0
        length = bbox_h * 0.16
        base = (bx, by)
        mid = (bx + dx * length * 0.12, by + length * 0.5)
        tip = (bx + dx * length * 0.25, by + length * 0.9)
        canvas = draw_pointed_stroke(canvas, [base, mid, tip], base_width=detail_width * 0.8, fill=(255, 255, 255))

    return canvas


def draw_smooth_fills(canvas: Image.Image, contours, fill=(0, 0, 0), supersample: int = 6) -> Image.Image:
    """Fill each smoothed contour as a solid polygon instead of pasting a
    mask's raw pixels, so the fill's own edge is fluid and anti-aliased
    too -- filling the raw mask directly would leave a pixel-jagged edge
    even with a smooth stroke drawn on top of it."""
    if not contours:
        return canvas

    w, h = canvas.size
    big = Image.new("RGBA", (w * supersample, h * supersample), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    for points in contours:
        scaled = [(px * supersample, py * supersample) for px, py in points]
        draw.polygon(scaled, fill=fill + (255,))

    big = big.resize((w, h), Image.LANCZOS)
    canvas.paste(big, (0, 0), big)
    return canvas


def draw_collar_hint(canvas: Image.Image, clothes_outer, detail_width: float) -> Image.Image:
    """A thin inset line paralleling the top (shoulder/neckline) edge of
    the clothing silhouette -- a cheap, generically-derivable stand-in for
    a collar seam. There's no landmark source for an actual collar shape
    (MediaPipe's face landmarker doesn't extend to the shoulders), so
    this doesn't claim to trace a real collar -- it only adds the kind of
    interior structure line the reference avatars have and a single flat
    silhouette otherwise completely lacks."""
    if not clothes_outer:
        return canvas

    contour = max(
        clothes_outer,
        key=lambda c: (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min()),
    )
    top_y = contour[:, 1].min()
    bbox_h = contour[:, 1].max() - top_y
    cx, cy = contour[:, 0].mean(), contour[:, 1].mean()

    band = contour[contour[:, 1] < top_y + bbox_h * 0.12]
    if len(band) < 6:
        return canvas
    band = band[np.argsort(band[:, 0])]

    inset = bbox_h * 0.045
    pts = []
    for px, py in band:
        dx, dy = cx - px, cy - py
        norm = max((dx**2 + dy**2) ** 0.5, 1)
        pts.append((px + dx / norm * inset, py + dy / norm * inset))

    canvas = draw_smooth_open_stroke(canvas, pts, width=max(1, detail_width - 2))
    return draw_shirt_buttons(canvas, contour, top_y, bbox_h, cx, detail_width)


def draw_shirt_buttons(canvas: Image.Image, clothes_contour, top_y: float, bbox_h: float, cx: float, detail_width: float) -> Image.Image:
    """A couple of small solid button dots down the vertical center of the
    clothing, below the collar -- the reference style shows a clean
    collar with visible buttons, not a featureless flat shape."""
    for frac in (0.22, 0.34):
        by = top_y + bbox_h * frac
        r = max(detail_width * 0.35, 1)
        canvas = draw_smooth_dot(canvas, cx, by, r)
    return canvas


def draw_clothes_shading(canvas: Image.Image, clothes_only: np.ndarray, gray: np.ndarray) -> Image.Image:
    """A second, darker flat grey patch within the clothing silhouette,
    covering whichever half of it the photo's own lighting shows as
    darker -- built the same way the reference avatars build dimension
    (a couple of adjacent flat tones, no gradient), instead of leaving
    clothing as one single flat grey with no shading vocabulary at all."""
    vals = gray[clothes_only]
    if vals.size < 50:
        return canvas

    # Splitting on the raw per-pixel gray value picks up fabric texture/
    # pattern at full resolution (every fold, every weave highlight),
    # which produces a chaotic scatter of tiny regions instead of one
    # clean "shadow side" shape -- blur heavily first (relative to the
    # clothing region's own size) so only the large-scale lighting
    # gradient survives, then clean up the resulting binary mask before
    # tracing it.
    ys, xs = np.where(clothes_only)
    region_size = max(ys.max() - ys.min(), xs.max() - xs.min(), 1)
    sigma = max(region_size * 0.08, 5)
    blurred = ndimage.gaussian_filter(gray, sigma=sigma)

    median = np.median(blurred[clothes_only])
    dark_clothes = clothes_only & (blurred <= median)
    cleanup = disk(max(int(sigma), 2))
    dark_clothes = ndimage.binary_closing(dark_clothes, structure=cleanup)
    dark_clothes = ndimage.binary_opening(dark_clothes, structure=cleanup)

    dark_contours = smooth_contours(dark_clothes, min_area_frac=0.01, smoothing=2.0)
    return draw_smooth_fills(canvas, dark_contours, fill=CLOTHES_SHADOW_FILL)


def reveal_ears(hair_clothes: np.ndarray, cat_mask: np.ndarray, im: Image.Image, landmarks, w: int, h: int, search_frac: float = 0.35) -> np.ndarray:
    """A shadowed ear against dark hair can get misclassified as hair
    entirely by the segmenter (similar tones, similar local texture),
    swallowing the ear into the solid hair fill with no indication it was
    ever there. Within a small window near each ear (anchored at the face
    oval's widest point, the landmark nearest each ear -- MediaPipe's face
    mesh has no ear landmarks of its own), reclassify hair-labeled pixels
    that actually match this photo's real skin tone back to skin, carving
    the ear back out instead of leaving it guessed away."""
    rgb = np.array(im.convert("RGB")).astype(np.float64)
    skin_mask = (cat_mask == FACE_SKIN) | (cat_mask == BODY_SKIN)
    if not skin_mask.any():
        return hair_clothes

    # Skin tone varies with lighting (shadow side vs lit side), so match
    # against the full observed range, not just a single mean color --
    # an ear in shadow needs to compare against shadowed skin, not the
    # brightest lit skin elsewhere on the face.
    skin_pixels = rgb[skin_mask]
    skin_mean = skin_pixels.mean(axis=0)
    skin_cov = np.cov(skin_pixels.T) + np.eye(3) * 1e-3
    skin_cov_inv = np.linalg.inv(skin_cov)

    oval_order = _ordered_face_oval_indices()
    oval_pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in oval_order])
    face_width = oval_pts[:, 0].max() - oval_pts[:, 0].min()
    radius = max(int(face_width * search_frac), 8)

    corrected = hair_clothes.copy()
    for lid in (234, 454):
        lm = landmarks[lid]
        cx, cy = int(lm.x * w), int(lm.y * h)
        y0, y1 = max(cy - radius, 0), min(cy + radius, h)
        x0, x1 = max(cx - radius, 0), min(cx + radius, w)
        if y1 <= y0 or x1 <= x0:
            continue

        window = rgb[y0:y1, x0:x1] - skin_mean
        # Mahalanobis distance to the observed skin-tone distribution --
        # accounts for skin's actual brightness/hue spread instead of a
        # fixed radius in raw RGB, which would either miss shadowed skin
        # or false-positive on warm-toned hair.
        dist = np.sqrt(np.einsum("...i,ij,...j->...", window, skin_cov_inv, window))
        skin_like = dist < 5.0

        hair_window = hair_clothes[y0:y1, x0:x1]
        ear_notch = hair_window & skin_like

        # A per-pixel color match is noisy at this scale (JPEG blocking,
        # a stray hair strand crossing the ear) and previously carved a
        # jagged, scattered notch instead of one clean ear-shaped gap. Close
        # small internal gaps, then keep only the single largest connected
        # blob (a real ear is one shape; scattered single-pixel matches
        # elsewhere in the window aren't) before subtracting it out.
        notch_kernel = disk(max(2, int(radius * 0.08)))
        ear_notch = ndimage.binary_closing(ear_notch, structure=notch_kernel)
        ear_notch = ndimage.binary_opening(ear_notch, structure=notch_kernel)
        labeled, num = ndimage.label(ear_notch)
        if num > 1:
            sizes = ndimage.sum(ear_notch, labeled, range(1, num + 1))
            ear_notch = labeled == (np.argmax(sizes) + 1)

        corrected[y0:y1, x0:x1][ear_notch] = False

    return corrected


_REMBG_SESSION = None


def rembg_foreground(im: Image.Image, threshold: int = 127) -> np.ndarray:
    """A dedicated matting model for just the foreground vs. background
    question, instead of MediaPipe's multiclass segmenter -- which has to
    simultaneously classify hair/skin/clothes/etc, and (unlike a model
    built for this one job) runs at a fixed, fairly low 256x256 internal
    resolution regardless of the photo's real size.

    Uses rembg's birefnet-portrait model by default, not its faster u2net
    default: on a photo with dark hair against a dark, textured background
    (a chalkboard), u2net's foreground mask included a chunk of the
    shadowed wall next to the head as if it were hair, extending the
    silhouette well past the real hairline -- birefnet-portrait (a heavier
    transformer model, trained specifically on portraits) doesn't make
    that mistake on the same photo. It's meaningfully slower on CPU. Set
    the AVATAR_REMBG_MODEL env var (e.g. to "u2net") to override, for
    quick iteration when you don't need the extra accuracy on a
    particular photo."""
    import os

    from rembg import new_session, remove

    model_name = os.environ.get("AVATAR_REMBG_MODEL", "birefnet-portrait")

    global _REMBG_SESSION
    if _REMBG_SESSION is None or _REMBG_SESSION.model_name != model_name:
        # Force plain CPU execution. onnxruntime's macOS wheel also
        # registers CoreMLExecutionProvider, and rembg's default provider
        # list lets onnxruntime pick it -- which hands the model to
        # ANECompilerService (Apple Neural Engine compilation) instead of
        # just running it, sometimes taking many minutes for a single
        # session and hanging in a way Ctrl+C can't interrupt (it's a
        # separate OS daemon, not the python process).
        _REMBG_SESSION = new_session(model_name, providers=["CPUExecutionProvider"])

    cutout = remove(im, session=_REMBG_SESSION).convert("RGBA")
    alpha = np.array(cutout)[:, :, 3]
    return alpha > threshold


def _base_layers(im: Image.Image, cat_mask: np.ndarray, landmarks=None):
    """Shared groundwork for both composite modes: a white canvas with flat
    black hair/clothes fills and a thick rounded outer silhouette outline,
    plus the raw foreground mask for callers that need it. This part is
    already fairly abstract (flat fills, no strand-level texture), so it's
    fine to condition generation on -- the fine facial detail each mode adds
    on top is where the two modes diverge."""
    gray = np.array(im.convert("L")).astype(np.float64)
    foreground = rembg_foreground(im)
    scale = stroke_scale(*gray.shape[::-1])
    outline_width = max(2, round(OUTLINE_WIDTH * scale))
    detail_width = max(1, round(DETAIL_WIDTH * scale))

    # Round every shape's edges/corners to match the house style's thick,
    # rounded-cap strokes instead of raw pixel-jagged boundaries: a
    # closing (dilate then erode) rounds concave corners and smooths
    # jagged edges, an opening (erode then dilate) rounds convex corners
    # and clips small spurs. Both use a disk structuring element so the
    # rounding is actually circular, not the diamond shape a default
    # cross-shaped structure would give.
    # "Everything in the silhouette that isn't visible skin" instead of
    # "whatever MediaPipe specifically labelled hair/clothes/other" --
    # rembg's silhouette and MediaPipe's category boundaries don't agree
    # down to the pixel (e.g. a shadow fold MediaPipe didn't confidently
    # classify as anything), and defining the fill this way makes that
    # kind of mismatch structurally impossible rather than something to
    # patch over: there's no category MediaPipe could get "wrong" here
    # that would leave a gap, since we're not asking it to positively
    # identify hair/clothes, only to say what's skin.
    skin = (cat_mask == FACE_SKIN) | (cat_mask == BODY_SKIN)
    hair_clothes = foreground & ~skin
    if landmarks is not None:
        h, w = gray.shape
        hair_clothes = reveal_ears(hair_clothes, cat_mask, im, landmarks, w, h)
    kernel = disk(4)
    hair_clothes = ndimage.binary_closing(hair_clothes, structure=kernel)
    hair_clothes = ndimage.binary_opening(hair_clothes, structure=kernel)
    hair_clothes = hair_clothes & foreground

    # Split the combined silhouette-derived mask back into hair (flat
    # black) vs clothes (flat mid-grey) using the segmenter's own
    # category labels -- the two tones the reference avatars actually use,
    # instead of one undifferentiated black mass. A pixel inside
    # hair_clothes that the segmenter didn't confidently call HAIR or
    # CLOTHES (labelled OTHER, e.g. a low-contrast collar edge) is
    # resolved to whichever of the two is spatially nearest, so it still
    # gets a definite tone instead of a coin-flip per pixel.
    hair_only = hair_clothes & (cat_mask == HAIR)
    clothes_only = hair_clothes & (cat_mask == CLOTHES)
    unclassified = hair_clothes & ~hair_only & ~clothes_only
    if unclassified.any() and (hair_only.any() or clothes_only.any()):
        labels = np.zeros(cat_mask.shape, dtype=np.uint8)
        labels[hair_only] = 1
        labels[clothes_only] = 2
        known = labels > 0
        _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
        nearest = labels[iy, ix]
        hair_only = hair_only | (unclassified & (nearest == 1))
        clothes_only = clothes_only | (unclassified & (nearest == 2))
    elif unclassified.any():
        # No confidently-classified pixels at all to resolve against
        # (rare) -- clothes is the safer default fill.
        clothes_only = clothes_only | unclassified

    out_im = Image.fromarray(np.full(gray.shape + (3,), 255, dtype=np.uint8))

    # Fill and stroke hair/clothes (and the outer silhouette, for the parts
    # of the edge where skin is directly visible against the background,
    # e.g. jaw/cheek) from smoothed contours throughout, not a raw pixel
    # mask -- a single fluid line that still follows this specific photo's
    # actual shape, instead of a stair-stepped edge. min_area is kept low
    # since hair and clothes are frequently two disconnected blobs, split
    # by a visible neck, and both matter.
    #
    # include_holes=True matters here specifically: hair that frames both
    # sides of a visible face (rather than just sitting above it) makes a
    # ring shape with the face as a hole in the middle. Filling only the
    # outer contour would ignore that hole and paint solid black straight
    # over the face.
    #
    # Extra smoothing here (vs. the tighter default used for FACE_SKIN):
    # dark hair against a dark, textured background (a chalkboard, a
    # shadowed wall) is exactly where the segmenter's pixel-to-pixel
    # boundary is least certain, since there's little real contrast to go
    # on -- unlike the jaw/face edge, this noise isn't real shape detail
    # worth preserving.
    hair_outer, hair_holes = smooth_contours(hair_only, min_area_frac=0.001, include_holes=True, smoothing=1.5)
    clothes_outer, clothes_holes = smooth_contours(
        clothes_only, min_area_frac=0.001, include_holes=True, smoothing=1.5
    )
    # Unlike hair_only/clothes_only just above, this is rembg's own
    # foreground cutout, not the segmenter's category mask -- rembg is
    # trained specifically for clean person cutouts, so it stays reliable
    # even against a noisy background (e.g. a chalkboard) where the
    # category segmenter's hair boundary genuinely isn't. That means the
    # jaw/cheek edge here is real measured shape, not segmentation noise --
    # it's the single feature most responsible for an avatar actually
    # looking like this specific person (per repeated feedback that jaw
    # shape is where most of the likeness lives), so it shouldn't get the
    # same heavy denoising smoothing the genuinely-noisy hair/clothes
    # edges need. A much lighter pass here still removes pixel jaggedness
    # without spline-averaging away a jaw angle or chin point into a
    # generic oval.
    silhouette_contours = smooth_contours(foreground, smoothing=0.3)
    # A jagged/tufted fringe (see jag_fringe) instead of hair's otherwise
    # smooth spline-fit edge, matching the reference avatars' hairline.
    hair_outer = [jag_fringe(c) for c in hair_outer]

    # Hair fill tracks this specific photo's actual hair darkness instead
    # of a fixed tone -- none of the reference avatars use a universal
    # grey; a lighter-haired subject gets a lighter fill and a dark-haired
    # subject (like most of the reference library, Greg included) gets
    # solid near-black, same as they'd photograph.
    hair_fill = HAIR_FILL
    if hair_only.any():
        hair_gray = gray[hair_only].mean()
        hair_fill = tuple(int(v) for v in np.clip([hair_gray * 0.55] * 3, 15, 165))

    out_im = draw_smooth_fills(out_im, hair_outer, fill=hair_fill)
    out_im = draw_smooth_fills(out_im, hair_holes, fill=(255, 255, 255))
    out_im = draw_smooth_fills(out_im, clothes_outer, fill=CLOTHES_FILL)
    out_im = draw_smooth_fills(out_im, clothes_holes, fill=(255, 255, 255))
    out_im = draw_clothes_shading(out_im, clothes_only, gray)
    out_im = draw_smooth_strokes(out_im, silhouette_contours, width=outline_width)
    out_im = draw_smooth_strokes(out_im, hair_outer, width=outline_width)
    out_im = draw_smooth_strokes(out_im, hair_holes, width=outline_width)
    out_im = draw_smooth_strokes(out_im, clothes_outer, width=outline_width)
    out_im = draw_smooth_strokes(out_im, clothes_holes, width=outline_width)
    out_im = draw_collar_hint(out_im, clothes_outer, detail_width)
    out_im = draw_hair_strands(out_im, hair_outer, detail_width=detail_width)
    # No dense interior hatching: none of the reference avatars actually
    # use it (that was a misreading of a couple of naturally lighter-
    # haired references) -- a handful of crown tufts from
    # draw_hair_strands above is the full extent of hair texture in the
    # house style, not additional hatching strokes across the whole mass.

    return np.array(out_im), foreground, gray, hair_clothes


def draw_dark_face_detail(out_im: Image.Image, cat_mask: np.ndarray, gray: np.ndarray, foreground: np.ndarray, landmarks, detail_width: float = DETAIL_WIDTH, scale: float = 1.0, glasses: bool = False) -> Image.Image:
    """Threshold real dark detail within the skin -- currently just
    glasses, drawn as a single continuous outline -- and drop everything
    else (beard shadow, stubble, skin texture) instead of painting it as a
    mid-grey stipple. A flat reference avatar has no halftone texture and
    only ever uses a couple of flat shades, so tracing brightness noise as
    "detail" just reads as washed out. Eyebrows/nose/mouth are instead
    drawn separately as clean landmark lines, not thresholded from the
    photo."""
    out = np.array(out_im)
    fg_vals = gray[foreground]
    lo, hi = np.percentile(fg_vals, [2, 98]) if fg_vals.size else (0, 255)
    gray_norm = np.clip((gray - lo) / max(hi - lo, 1) * 255, 0, 255)

    # A thin glasses frame is made of many small, individually-tiny dark
    # blobs (reflections and pixel gaps break it into disconnected
    # pieces), so it's only looked for within a padded box around the
    # eyes/eyebrows -- restricting the search region is what lets "keep
    # every blob size here" not also mean "keep stray dark pixels
    # anywhere on the face."
    if landmarks is not None:
        from mediapipe.tasks.python import vision

        h, w = gray.shape
        eye_idx = set()
        for conns in (
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        ):
            for c in conns:
                eye_idx.add(c.start)
                eye_idx.add(c.end)
        xs = [landmarks[i].x * w for i in eye_idx]
        ys = [landmarks[i].y * h for i in eye_idx]
        pad_x, pad_y = (max(xs) - min(xs)) * 0.35, (max(ys) - min(ys)) * 1.4
        gx0, gx1 = min(xs) - pad_x, max(xs) + pad_x
        gy0, gy1 = min(ys) - pad_y, max(ys) + pad_y
        glasses_region = np.zeros(gray_norm.shape, dtype=bool)
        glasses_region[int(gy0) : int(gy1), int(gx0) : int(gx1)] = True
    else:
        glasses_region = np.zeros(gray_norm.shape, dtype=bool)

    out_im = Image.fromarray(out)

    # Whether the subject wears glasses is taken as an explicit input
    # (the `glasses` parameter) rather than guessed from pixels: a
    # brightness/area heuristic here can't actually distinguish "this is
    # a glasses frame" from "this person has visible dark eyebrows" --
    # real eyebrow pixels alone routinely cover well over the area
    # fraction a real glasses frame would, since eyebrows are a
    # legitimate, substantial dark feature within this same padded
    # search box. Trying to out-guess that from pixel area produced a
    # false-positive "glasses" outline (plus sparkle flourish) drawn
    # around a subject's own eyebrow/eye-socket shadow. An explicit flag
    # sidesteps the ambiguity entirely.
    #
    # Not dark_mask (which requires skin_mask to be true at that pixel):
    # the glasses frame itself is exactly what the segmenter does *not*
    # classify as skin, so restricting to skin_mask was excluding almost
    # the entire frame and leaving only stray edge/reflection pixels that
    # happened to fall on adjacent skin -- which is what the old "many tiny
    # disconnected blobs" fragmentation actually was. Within the padded
    # eye/eyebrow box that already constrains the search spatially, just
    # threshold brightness directly against the real photo.
    #
    # Previously filled every surviving small dark blob solid black
    # individually, which left visible gaps wherever a reflection or
    # pixel-level noise broke the frame into disconnected pieces. A
    # morphological closing first bridges those small gaps into one
    # connected shape per lens/frame, then it's drawn as a single smoothed
    # *outline* (not a filled blob) -- matching the reference avatars'
    # hollow rounded-rectangle frames instead of a solid dark mass, and
    # guaranteeing the drawn line has no breaks regardless of how
    # fragmented the raw pixels were.
    glasses_raw = glasses_region & foreground & (gray_norm < 120)
    if glasses and glasses_raw.any():
        glasses_closed = ndimage.binary_closing(glasses_raw, structure=disk(max(2, round(5 * scale))))
        glasses_outer, glasses_holes = smooth_contours(
            glasses_closed, min_area_frac=0.0006, include_holes=True, smoothing=0.4
        )
        # Bolder than the general detail_width (everything else is thin
        # in this style, but the reference glasses frame is consistently
        # drawn thick/confident, not at the same weight as the eyebrows
        # or nose).
        glasses_width = detail_width + 2
        out_im = draw_smooth_strokes(out_im, glasses_outer, width=glasses_width)
        out_im = draw_smooth_strokes(out_im, glasses_holes, width=glasses_width)
        # No sparkle/reflection tick flourish here (a prior version added
        # one): the confirmed target reference style is a sparse, minimal
        # sketch, and extra flourish marks cut against "as few strokes as
        # possible" -- unclear the sparkle convention even belongs to this
        # substyle rather than the denser one it was drawn from.

    return out_im


def composite_line_art(photo_path: Path, out_path: Path, glasses: bool = False):
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)
    print("Detecting face landmarks...")
    landmarks = get_face_landmarks(im)
    out, foreground, gray, hair_clothes = _base_layers(im, cat_mask, landmarks)
    h, w = gray.shape
    scale = stroke_scale(w, h)
    detail_width = max(1, round(DETAIL_WIDTH * scale))

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=detail_width)
    out_im = draw_dark_face_detail(
        out_im, cat_mask, gray, foreground, landmarks, detail_width=detail_width, scale=scale, glasses=glasses
    )
    out = np.array(out_im)

    if landmarks is not None:
        out = draw_face_structure_lines(out, landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping face structure lines and dot-eye replacement")

    out, _ = crop_to_content(out, foreground)
    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def face_skin_contours(cat_mask: np.ndarray):
    """The segmenter's own FACE_SKIN category already traces almost
    exactly the jawline (see the FACE_SKIN vs BODY_SKIN comparison that
    motivated this) -- a real semantic distinction it learned (face vs
    neck), not a blind local pixel search. Using it directly beats
    reconstructing the same boundary from a landmark position and
    gradient-snapping along a short search line, which has no such
    understanding and can grab a stronger but wrong nearby edge (glasses,
    a collar seam, a shirt pattern) instead of the real, sometimes subtle,
    jaw shadow.

    smoothing=0.5 (up from the function default of 0.15): the segmenter's
    pixel-to-pixel boundary along the jaw has real per-pixel jitter that a
    light smoothing pass leaves visible as a wobble rather than a clean
    curve -- this is the same fix already applied to the hair/clothes/
    silhouette contours below, just also needed here."""
    return smooth_contours(cat_mask == FACE_SKIN, min_area_frac=0.01, smoothing=0.5)


def draw_face_structure_lines(out: np.ndarray, landmarks, w: int, h: int, detail_width: float = DETAIL_WIDTH) -> np.ndarray:
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
    line_width = max(1, detail_width - 1)

    def ordered_points(conns):
        """FACE_LANDMARKS_*_EYEBROW is a set of disjoint segment pairs, not a
        pre-ordered walk -- reconstruct the walk the same way
        _ordered_face_oval_indices() does, so a path can be tapered
        end-to-middle-to-end instead of drawn as unordered straight
        segments."""
        adjacency = {}
        for c in conns:
            adjacency.setdefault(c.start, []).append(c.end)
            adjacency.setdefault(c.end, []).append(c.start)
        start = next(idx for idx, nbrs in adjacency.items() if len(nbrs) == 1)
        order = [start]
        prev, cur = None, start
        while True:
            neighbors = [n for n in adjacency[cur] if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            order.append(nxt)
            prev, cur = cur, nxt
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in order]

    # Tapered (thick middle, thin ends) instead of a constant-width line --
    # matches the reference avatars' brush-stroke eyebrows rather than a
    # uniform-diameter bar. A small lift off their raw landmark position
    # keeps the brow from literally overlapping the glasses' top rim pixel
    # for pixel, but the actual house style (confirmed both by direct
    # correction against a real photo and by the same design intent
    # already established in generate_avatar_gemini.py's prompt: "sit
    # close enough above the frame to almost touch or lightly overlap its
    # top rim") wants the brow sitting close to/nearly touching the frame,
    # not a large deliberate gap -- 0.08 was too much lift.
    eye_span = np.linalg.norm(
        np.array([landmarks[263].x * w, landmarks[263].y * h]) - np.array([landmarks[33].x * w, landmarks[33].y * h])
    )
    brow_lift = eye_span * 0.02
    # Thin, only lightly tapered -- the confirmed reference style draws
    # eyebrows as a single simple curved stroke, not a thick filled shape
    # (an earlier version of this went bold/flat based on a different,
    # non-representative pair of avatars -- reverted).
    for conns in (connections.FACE_LANDMARKS_LEFT_EYEBROW, connections.FACE_LANDMARKS_RIGHT_EYEBROW):
        pts = [(px, py - brow_lift) for px, py in ordered_points(conns)]
        img = draw_tapered_stroke(img, pts, mid_width=detail_width + 1, end_width=max(1, detail_width - 1))

    # Just the line under the nose (nostril hook to nostril hook), not the
    # full nose mesh (bridge + nostril wings + tip outline) -- matches the
    # reference avatars, which only ever mark the nose with a single simple
    # under-nose curve that hooks up at each nostril. A raw polyline through
    # these landmarks has visible straight-segment kinks at each point; a
    # light spline keeps the actual up-down-up-down nostril shape (unlike a
    # heavier smoothing, which averages it into one plain arc) while making
    # it read as one fluid stroke.
    from scipy.interpolate import splev, splprep

    nose_bottom_idx = [49, 129, 98, 2, 327, 358, 279]
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in nose_bottom_idx])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
    # Reference avatars mark the nose with a small two-nostril "gull-wing"
    # mark -- two visible hooks either side of a shallow center dip, not
    # a flat/straight line. The actual hook shape lives in the OUTER
    # portion of this spline's parameter range (near landmarks 49/279,
    # the real nostril wing corners) -- a narrow central crop (previous
    # versions used 0.44-0.56, then 0.34-0.66) keeps only the flattest
    # part near the tip (landmark 2, at u=0.5) and cuts away exactly the
    # hooks that make it read as a nose, leaving a near-straight line
    # that combined with the philtrum tick below reads as a vertical
    # "bridge" instead. Widened to actually include the hooks.
    xs, ys = splev(np.linspace(0.15, 0.85, 24), tck)
    img = draw_smooth_open_stroke(img, list(zip(xs, ys)), width=max(1, line_width - 1))

    # A short philtrum tick between nose and mouth -- the reference style
    # includes it as a small, clearly separated mark, not touching the
    # nose curve above it (touching is what read as one continuous
    # vertical "bridge" stroke in an earlier version).
    nose_center_y = ys[len(ys) // 2]
    gap = eye_span * 0.05
    philtrum_len = eye_span * 0.06
    philtrum_top = (landmarks[2].x * w, nose_center_y + gap)
    philtrum_bottom = (landmarks[2].x * w, nose_center_y + gap + philtrum_len)
    img = draw_smooth_open_stroke(img, [philtrum_top, philtrum_bottom], width=max(1, line_width - 1))

    # Mouth: a closed smile with real structure -- an upper curve through
    # the real outer-lip landmarks (mouth corners to cupid's bow), plus a
    # shorter lower-lip line beneath it, offset from the upper curve's own
    # midsection rather than a second set of guessed landmark indices, so
    # it can't drift out of proportion to the mouth width/height this
    # specific face actually measured. An earlier, overly-minimized
    # version of this read as flat/expressionless -- these two lines are
    # the confirmed minimum for the mouth to still read as a smile.
    # A direct correction against a real photo measured the whole mouth
    # (not just the lower lip) sitting about 5% of eye_span too high here
    # versus its actual position -- shift both lines down by that amount
    # rather than just the landmarks' raw position.
    mouth_shift = eye_span * 0.05
    upper_lip_idx = [61, 40, 37, 0, 267, 270, 291]
    upts = np.array([(landmarks[i].x * w, landmarks[i].y * h + mouth_shift) for i in upper_lip_idx])
    utck, _ = splprep([upts[:, 0], upts[:, 1]], s=0, k=3)
    uxs, uys = splev(np.linspace(0, 1, 40), utck)
    img = draw_smooth_open_stroke(img, list(zip(uxs, uys)), width=line_width)

    mouth_h = abs(landmarks[17].y - landmarks[0].y) * h
    lower_xs, lower_ys = splev(np.linspace(0.3, 0.7, 20), utck)
    lower_ys = np.array(lower_ys) + mouth_h * 0.55
    img = draw_smooth_open_stroke(img, list(zip(lower_xs, lower_ys)), width=line_width)

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
    print("Detecting face landmarks for face structure...")
    landmarks = get_face_landmarks(im)
    out, foreground, gray, hair_clothes = _base_layers(im, cat_mask, landmarks)
    h, w = gray.shape
    detail_width = max(1, round(DETAIL_WIDTH * stroke_scale(w, h)))

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=detail_width)
    out = np.array(out_im)

    if landmarks is not None:
        out = draw_face_structure_lines(out, landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping face structure lines")

    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def scaffold_composite(photo_path: Path, out_path: Path, mask_path: Path, glasses: bool = False):
    """A third mode: the classical pipeline's own outer silhouette/hair
    shape and jaw/cheek contour, locked in as final art, with a companion
    inpaint mask marking the interior face region as editable. Meant for
    scripts/generate_avatar_instantid.py's --mask, so the generation model
    can only draw within that masked region (eyes, nose, mouth, stubble
    texture) while every pixel outside it -- the actual face/hair shape --
    stays exactly as measured, not subject to a ControlNet's best guess.

    Soft conditioning (a ControlNet, at any strength) only ever nudges the
    output; it can't guarantee the shape survives, which is exactly what
    happened testing composite_line_art.py's --structure output as
    --control-image at low and high strength alike. A hard mask is the
    only way to actually guarantee it."""
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)
    print("Detecting face landmarks for jaw line and inpaint mask...")
    landmarks = get_face_landmarks(im)
    out, foreground, gray, hair_clothes = _base_layers(im, cat_mask, landmarks)
    h, w = gray.shape
    scale = stroke_scale(w, h)
    detail_width = max(1, round(DETAIL_WIDTH * scale))

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=detail_width)
    out_im = draw_dark_face_detail(
        out_im, cat_mask, gray, foreground, landmarks, detail_width=detail_width, scale=scale, glasses=glasses
    )

    # Eyebrows, nose, mouth, and eyes are drawn classically here too
    # (previously only composite_line_art()/structure_composite() did this)
    # and locked into the protected/unmasked region below -- these are
    # exactly the features a landmark computation gets right reliably, so
    # there's no reason to leave them for the generation model to
    # freehand from a blank masked hole with only a text prompt to go on,
    # which is what was actually producing the stubble/gap artifacts.
    if landmarks is not None:
        out = draw_face_structure_lines(np.array(out_im), landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping face structure lines and dot-eye replacement")
        out = np.array(out_im)

    # The editable region is the segmenter's own FACE_SKIN area, minus a
    # dilated margin around every classically-drawn feature above -- the
    # model's job shrinks from "draw an entire face" down to just skin
    # polish in the gaps between those features, instead of being able to
    # redraw (and potentially break) eyebrows/nose/mouth/eyes/glasses that
    # are already correct.
    locked = np.any(out != 255, axis=-1) & (cat_mask == FACE_SKIN)
    locked = ndimage.binary_dilation(locked, structure=disk(6))
    mask = (cat_mask == FACE_SKIN) & ~locked

    # Crop scaffold and mask together, using the same bounds, so they stay
    # pixel-aligned for generate_avatar_instantid.py's --control-image/--mask.
    out, (top, bottom, left, right) = crop_to_content(out, foreground)
    mask = mask[top:bottom, left:right]

    Image.fromarray(out).save(out_path)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    print(f"Saved scaffold to {out_path}, mask to {mask_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: composite_line_art.py path/to/photo.jpg [out.png] [--structure|--scaffold] [--glasses]"
        )

    structure_mode = "--structure" in sys.argv
    scaffold_mode = "--scaffold" in sys.argv
    glasses = "--glasses" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--structure", "--scaffold", "--glasses")]

    photo_path = Path(args[0])
    default_suffix = "_structure" if structure_mode else "_scaffold" if scaffold_mode else "_lineart"
    out_path = Path(args[1]) if len(args) > 1 else photo_path.with_name(
        photo_path.stem + default_suffix + ".png"
    )

    if scaffold_mode:
        mask_path = out_path.with_name(out_path.stem + "_mask.png")
        scaffold_composite(photo_path, out_path, mask_path, glasses=glasses)
    elif structure_mode:
        structure_composite(photo_path, out_path)
    else:
        composite_line_art(photo_path, out_path, glasses=glasses)


if __name__ == "__main__":
    main()
