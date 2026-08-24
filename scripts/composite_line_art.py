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
OUTLINE_WIDTH = 7
DETAIL_WIDTH = 5
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


FACE_PARSING_MODEL_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx"
FACE_PARSING_MODEL_CACHE = Path.home() / ".cache" / "avatar_models" / "face_parsing_resnet18.onnx"

# The CelebAMask-HQ 19-class scheme this model was trained on. Class 0 is
# background; classes 1..18 map to ATTRIBUTES[0..17] (i.e. mask value ==
# index + 1).
FACE_PARSING_ATTRIBUTES = [
    "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g", "l_ear", "r_ear",
    "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck", "neck_l", "cloth",
    "hair", "hat",
]
FACE_PARSING_CLASS = {name: i + 1 for i, name in enumerate(FACE_PARSING_ATTRIBUTES)}


def get_face_parsing_model_path() -> Path:
    if not FACE_PARSING_MODEL_CACHE.exists():
        print("Downloading face parsing model (one-time, ~50MB)...")
        FACE_PARSING_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_PARSING_MODEL_URL, FACE_PARSING_MODEL_CACHE)
    return FACE_PARSING_MODEL_CACHE


def parse_face_regions(im: Image.Image, landmarks, w: int, h: int):
    """Runs a CelebAMask-HQ-trained BiSeNet face parser (see
    FACE_PARSING_ATTRIBUTES) and returns (class_mask, crop_box), where
    class_mask is a per-pixel class-index array covering crop_box = (x0,
    y0, x1, y1) in the original photo's pixel coordinates.

    This model expects a tight face-only crop, like its CelebAMask-HQ
    training images -- feeding it the full photo (with shoulders/
    background) starves the face of resolution and the parser silently
    fails on most classes: confirmed directly, nose/eyes/lips all came
    back with zero pixels on a full-frame input, and detected fine once
    cropped to the landmark bounding box padded generously for forehead/
    hair/some neck, matching CelebAMask-HQ's own framing.

    Unlike the landmark-index approach this replaces for facial-feature
    outlines, this returns the actual measured pixel boundary of each
    feature in this specific photo -- not an approximation from a fixed,
    generic point scheme."""
    import onnxruntime as ort
    import cv2

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = (x1 - x0) * 0.5
    pad_y_top = (y1 - y0) * 0.9
    pad_y_bot = (y1 - y0) * 0.5
    cx0, cx1 = max(0.0, x0 - pad_x), min(float(w), x1 + pad_x)
    cy0, cy1 = max(0.0, y0 - pad_y_top), min(float(h), y1 + pad_y_bot)

    im_bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    crop = im_bgr[int(cy0):int(cy1), int(cx0):int(cx1)]
    ch, cw = crop.shape[:2]

    sess = ort.InferenceSession(str(get_face_parsing_model_path()), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) / 255.0 - mean) / std
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    out = sess.run(["output"], {input_name: x})[0]
    mask512 = out.squeeze(0).argmax(0).astype(np.uint8)
    mask = cv2.resize(mask512, (cw, ch), interpolation=cv2.INTER_NEAREST)

    return mask, (int(cx0), int(cy0), int(cx1), int(cy1))


def vector_trace_bottom_arc(mask_crop: np.ndarray, class_id: int, crop_box, upscale: int = 8):
    """Given a parsing mask crop and a class index, vector-traces that
    class's region boundary with vtracer (real Bezier curve fitting, not a
    hand-rolled spline) and returns just its bottom arc -- the portion of
    the closed loop between the shape's leftmost and rightmost points that
    runs through the lower half, i.e. the part that reads as "the bottom
    edge of this feature" rather than its sides or top. Points come back
    in original-photo pixel coordinates.

    Splitting at the shape's own left/right extrema (rather than a fixed
    y-fraction threshold) is what keeps this from grabbing too much: a
    threshold loose enough to reach the hook height at the ends is also
    loose enough to walk back up the sides and close into a full loop --
    confirmed directly, a 0.55-of-height threshold produced a closed oval,
    not an open arc. The extrema split has no such failure mode: the
    bottom and top arcs are geometrically exactly the two halves either
    side of the shape's widest point, by construction.

    The raw parsing mask is native to the 512x512 model input, so its
    boundary is blocky at the crop's actual resolution -- upscaling,
    blurring, and re-thresholding before tracing (rather than tracing the
    blocky mask directly) is what gives vtracer a clean edge to fit a
    smooth curve to instead of amplifying the blockiness into jagged
    Bezier segments."""
    import cv2
    import vtracer
    from svgpathtools import parse_path
    import re
    import tempfile

    region = (mask_crop == class_id).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    big = cv2.resize(region * 255, (region.shape[1] * upscale, region.shape[0] * upscale), interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=upscale * 0.8)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    pad = 20
    big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    with tempfile.TemporaryDirectory() as tmp:
        png_path = f"{tmp}/mask.png"
        svg_path = f"{tmp}/mask.svg"
        cv2.imwrite(png_path, big)
        vtracer.convert_image_to_svg_py(
            png_path, svg_path,
            colormode="binary", mode="spline", filter_speckle=20,
            corner_threshold=100, length_threshold=8.0, splice_threshold=60,
            path_precision=2,
        )
        svg = open(svg_path).read()

    d = re.search(r'd="([^"]+)"', svg).group(1)
    tx, ty = 0.0, 0.0
    tmatch = re.search(r"translate\(([^,]+),([^)]+)\)", svg)
    if tmatch:
        tx, ty = float(tmatch.group(1)), float(tmatch.group(2))
    subpaths = re.findall(r"M[^M]*", d)
    areas = []
    for sp in subpaths:
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", sp)]
        sxs, sys_ = nums[0::2], nums[1::2]
        areas.append((max(sxs) - min(sxs)) * (max(sys_) - min(sys_)))
    # The smallest-bbox subpath is the traced feature itself; a larger one
    # is the outer canvas rectangle vtracer also emits in binary mode.
    best = subpaths[int(np.argmin(areas))]

    path = parse_path(best)
    N = 300
    pts = np.array([[path.point(i / N).real + tx, path.point(i / N).imag + ty] for i in range(N)])

    i_left, i_right = int(np.argmin(pts[:, 0])), int(np.argmax(pts[:, 0]))

    def arc_between(i0, i1):
        return list(range(i0, i1 + 1)) if i0 <= i1 else list(range(i0, N)) + list(range(0, i1 + 1))

    arc_a = pts[arc_between(i_left, i_right)]
    arc_b = pts[arc_between(i_right, i_left)]
    bottom = arc_a if arc_a[:, 1].mean() > arc_b[:, 1].mean() else arc_b

    cx0, cy0, _, _ = crop_box
    bottom_full = (bottom - pad) / upscale + np.array([cx0, cy0])
    return bottom_full


def draw_dot_eyes(out: np.ndarray, landmarks, w: int, h: int, detail_width: float = DETAIL_WIDTH) -> np.ndarray:
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
        # Sized up from an earlier 0.08 -- a direct correction against a
        # real photo traced both irises noticeably larger than that.
        pupil_r = max(eye_width * 0.12, 2)
        img = draw_smooth_dot(img, cx, cy, pupil_r)

        # Upper eyelid: a short curved arc directly over the pupil, not a
        # flat straight bar (the previous version) -- a straight tick with
        # no curve and a visible gap to the pupil below it reads as an
        # abstract icon mark, not an eye; a shallow arc that sits close
        # over the pupil is what actually makes it read as an eyelid with
        # an eye underneath, matching reference avatars where the pupil
        # sits snugly inside the lid curve's concavity rather than
        # floating separately below an unrelated bar. Constant width
        # (detail_width, the same as every other facial mark) instead of
        # tapered -- a direct comparison against a reference avatar's face
        # confirmed every stroke there is the same confident weight with
        # no taper anywhere, and this was the last facial mark still
        # tapered and sized off eye_width instead of detail_width.
        lid_half_w = eye_width * 0.24
        lid_y = cy - eye_width * 0.15
        bulge = eye_width * 0.07
        arc_t = np.linspace(-1.0, 1.0, 7)
        arc_pts = [(cx + t * lid_half_w, lid_y - bulge * (1 - t**2)) for t in arc_t]
        img = draw_smooth_open_stroke(img, arc_pts, width=detail_width)

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


def trace_glasses_contours(mask: np.ndarray, epsilon_frac: float = 0.006, smoothing: float = 0.05, defect_depth_frac: float = 0.12):
    """Like smooth_contours(include_holes=True), but built specifically for
    a manufactured, geometric shape (a glasses frame) rather than an
    organic one: a much larger epsilon_frac and much smaller smoothing
    than smooth_contours' defaults, so real corners stay crisp turns
    instead of being rounded into the same soft blob a hair/silhouette
    contour wants.

    Also actively removes deep, narrow inward notches via convexity-defect
    detection before smoothing: a real photo's glasses region routinely
    has a thin dark nose-pad/hinge wire threading in from the frame toward
    the lens interior (confirmed directly -- it isn't a segmentation
    fluke, it's real dark material in the crop, positioned well clear of
    the eyebrows), which the ordinary contour of the lens hole then has to
    detour around, showing up as an ugly self-crossing loop right at the
    inner corner of each lens once traced and smoothed. A real glasses
    lens opening is convex (a rounded rectangle), so a hull-relative
    defect deeper than defect_depth_frac of the shape's own diagonal is
    unambiguously that stray wire, not genuine lens-opening geometry --
    bridging straight across it (dropping the points strictly between the
    defect's start and end) removes exactly that notch while leaving the
    rest of the boundary's real shape untouched."""
    import cv2
    from scipy.interpolate import splev, splprep

    mask_u8 = mask.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], []

    def bridge_deep_defects(contour):
        contour = contour.reshape(-1, 1, 2).astype(np.int32)
        hull_idx = np.sort(cv2.convexHull(contour, returnPoints=False).flatten())
        if len(hull_idx) < 3:
            return contour.reshape(-1, 2).astype(np.float64)
        defects = cv2.convexityDefects(contour, hull_idx.reshape(-1, 1))
        pts = contour.reshape(-1, 2)
        if defects is None:
            return pts.astype(np.float64)
        diag = np.hypot(*(pts.max(0) - pts.min(0)))
        n = len(pts)
        skip = np.zeros(n, dtype=bool)
        for s, e, _f, d in defects:
            if d / 256.0 <= diag * defect_depth_frac:
                continue
            i = (s + 1) % n
            while i != e:
                skip[i] = True
                i = (i + 1) % n
        return pts[~skip].astype(np.float64)

    def smooth_one(contour):
        pts = bridge_deep_defects(contour.astype(np.float32))
        peri = cv2.arcLength(pts.reshape(-1, 1, 2).astype(np.float32), True)
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32), epsilon_frac * peri, True).squeeze(1)
        if approx.shape[0] < 4:
            return None
        x, y = approx[:, 0].astype(np.float64), approx[:, 1].astype(np.float64)
        try:
            tck, _ = splprep([x, y], s=len(x) * smoothing, per=True)
            xs, ys = splev(np.linspace(0, 1, 300), tck)
            return np.stack([xs, ys], axis=1)
        except Exception:
            return approx

    outer, holes = [], []
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour) < mask.size * 0.0006:
            continue
        smoothed = smooth_one(contour)
        if smoothed is None:
            continue
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



def draw_hair_flow_lines(canvas: Image.Image, hair_outer_contours, hair_fill, detail_width: float = DETAIL_WIDTH):
    """Long, confident directional strand lines instead of the earlier
    rejected pale cut-marks (both the fixed 3-strand version and the
    photo-derived hatched-highlight version read as odd pale patches
    breaking up the solid hair mass on sight, not as a shine/highlight
    cue -- see the removed draw_hair_strands/hair_highlight_lines).

    Insets of the hair silhouette's own upper curve (parallel offsets
    toward its centroid) instead of independently-invented strand paths --
    real hair layers near the crown roughly parallel the outer silhouette,
    so tracing concentric insets of a shape that's already this specific
    photo's actual hair silhouette gives lines with real directional
    structure for free, rather than a plausible-looking but arbitrary
    flow field. Drawn as a mid-tone between the hair fill and black (not
    white) so they read as strand structure within the mass, the same way
    the reference avatars' hair linework does, rather than as a cut into
    it."""
    if not hair_outer_contours:
        return canvas

    contour = max(
        hair_outer_contours,
        key=lambda c: (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min()),
    )
    top_y = contour[:, 1].min()
    bbox_h = contour[:, 1].max() - top_y
    cx, cy = contour[:, 0].mean(), contour[:, 1].mean()

    # Only the upper ~55% (crown down to roughly ear height) -- the lower
    # portion is the sideburn/nape area, thin and already close to the
    # outer edge, where an inset line would crowd the silhouette stroke
    # rather than read as interior structure. Longest contiguous run in
    # real path order (same technique used for the nose/collar arcs
    # elsewhere), not a naive x-sort, since the contour can wrap.
    in_band = contour[:, 1] < top_y + bbox_h * 0.55
    if in_band.sum() < 10:
        return canvas
    n = len(contour)
    idx2 = np.concatenate([np.where(in_band)[0], np.where(in_band)[0] + n])
    splits = np.where(np.diff(idx2) > 1)[0]
    run_starts = np.concatenate([[0], splits + 1])
    run_ends = np.concatenate([splits, [len(idx2) - 1]])
    best = np.argmax(run_ends - run_starts)
    best_idx = idx2[run_starts[best]:run_ends[best] + 1] % n
    band = contour[best_idx]

    line_color = tuple(min(int(v * 1.9 + 10), 90) for v in hair_fill)
    for inset_frac in (0.06, 0.13, 0.21):
        inset = bbox_h * inset_frac
        pts = []
        for px, py in band:
            dx, dy = cx - px, cy - py
            norm = max((dx**2 + dy**2) ** 0.5, 1)
            pts.append((px + dx / norm * inset, py + dy / norm * inset))
        canvas = draw_smooth_open_stroke(canvas, pts, width=max(1, detail_width - 1), fill=line_color)

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

    # Sorting by x (the old approach) assumes the top band is a simple
    # left-to-right sweep, which breaks at a V-neck collar: the collar's
    # two edges dip down and back up, so a point on one edge and a point
    # on the other can share nearly the same x while being far apart in
    # y. Re-sorting by x then treats those as adjacent, drawing a stroke
    # that crosses back on itself right at the collar notch -- confirmed
    # directly, a stray short double-tick mark right where the collar
    # dips. contour is already ordered along the actual perimeter (from
    # cv2's contour walk), so instead take the single longest contiguous
    # run of in-band points in that real path order, which follows the
    # true shoulder-to-collar-to-shoulder route including the dip,
    # rather than reconstructing a path from scratch.
    in_band = contour[:, 1] < top_y + bbox_h * 0.12
    if in_band.sum() < 6:
        return canvas
    n = len(contour)
    idx2 = np.concatenate([np.where(in_band)[0], np.where(in_band)[0] + n])
    splits = np.where(np.diff(idx2) > 1)[0]
    run_starts = np.concatenate([[0], splits + 1])
    run_ends = np.concatenate([splits, [len(idx2) - 1]])
    run_lengths = run_ends - run_starts
    best = np.argmax(run_lengths)
    best_idx = idx2[run_starts[best]:run_ends[best] + 1] % n
    band = contour[best_idx]

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
    canvas = draw_smooth_fills(canvas, dark_contours, fill=CLOTHES_SHADOW_FILL)

    # A second, deeper threshold band traced as a thin line (not filled)
    # instead of another flat patch -- garment folds are gradual shading
    # transitions, and one flat shadow shape only shows where the darkest
    # half is, not any structure within it. This is the same blur radius
    # already tuned to survive a busy fabric print (confirmed directly: a
    # plain Canny edge pass on this same blurred image mostly just found
    # the silhouette's own edge against the background, not interior
    # shading, on a dark, low-contrast, patterned shirt -- the print's
    # texture and the real lighting gradient are both too subtle there
    # for a generic edge detector to tell apart). Reusing the blur that
    # already isolates the real lighting gradient and tracing one more of
    # its own threshold bands as a line is a smaller, more reliable step
    # than asking edge detection to find fold structure from scratch.
    deep = np.percentile(blurred[clothes_only], 30)
    deep_clothes = clothes_only & (blurred <= deep)
    deep_clothes = ndimage.binary_closing(deep_clothes, structure=cleanup)
    deep_clothes = ndimage.binary_opening(deep_clothes, structure=cleanup)
    deep_contours = smooth_contours(deep_clothes, min_area_frac=0.015, smoothing=2.5)
    return draw_smooth_strokes(canvas, deep_contours, width=1)


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
        # The 0.08 kernel above only clears pixel-level jaggedness; a real
        # ear's actual anatomy (tragus notch, helix fold) still leaves the
        # blob's own boundary jagged at a coarser scale, which traced
        # straight into the outer silhouette as a thin spike/fork poking
        # out of the hair mass where none belongs (confirmed directly:
        # persisted with glasses drawing off, so it wasn't a glasses
        # detection artifact bleeding in -- this notch's own shape was
        # the source). A second, larger rounding pass smooths that coarser
        # jaggedness into a plain ear-sized blob before it ever reaches
        # the outer contour.
        round_kernel = disk(max(3, int(radius * 0.22)))
        ear_notch = ndimage.binary_closing(ear_notch, structure=round_kernel)
        ear_notch = ndimage.binary_opening(ear_notch, structure=round_kernel)
        labeled, num = ndimage.label(ear_notch)
        if num > 1:
            sizes = ndimage.sum(ear_notch, labeled, range(1, num + 1))
            ear_notch = labeled == (np.argmax(sizes) + 1)

        # A real, mostly-covered ear (hair falling in front of/beside it,
        # only a sliver actually visible) produces a notch too thin to
        # read as an ear once drawn -- it just shows up as a stray narrow
        # gap of skin between the hair silhouette's outer edge and the
        # notch's own inner edge, two close-but-not-identical curves that
        # read as an unintentional doubled line rather than a visible ear
        # (confirmed directly against a render and the source photo: the
        # photo does show a bit of ear peeking past the sideburn there,
        # but not enough to draw cleanly). Below this width, it reads
        # better to just leave that area covered by hair -- an ear that
        # isn't drawn at all is a smaller loss than a confusing sliver.
        if ear_notch.any():
            notch_ys, notch_xs = np.where(ear_notch)
            notch_w = notch_xs.max() - notch_xs.min()
            if notch_w < radius * 0.35:
                continue

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
    # hair_clothes | skin instead of the raw rembg foreground cutout, AND
    # the same smoothing=1.5 as hair_outer below (was 0.3) -- the jaw/
    # cheek edge itself is now drawn from face_oval_contour (real
    # face-mesh geometry, see its docstring), not from this silhouette, so
    # this only needs to be consistent with the hair/clothes shapes it's
    # drawn alongside. Switching the mask wasn't sufficient on its own:
    # checked directly, hair_clothes and hair_only were already pixel-
    # identical near the ear (0 pixels different), yet the two drawn
    # lines still visibly diverged there, because smooth_contours' own
    # smoothing/simplification amount was different between the two
    # calls (0.3 here vs. 1.5 for hair_outer) -- identical input pixels
    # still produce two different output curves at two different
    # smoothing strengths. Matching both removes that second source of
    # divergence too.
    # samples=2000 (up from the function's default 400): smooth_contours
    # resamples each contour to a fixed point count regardless of its
    # actual perimeter, so a much longer contour (this traces the whole
    # body outline, several times the perimeter of hair_outer's hair-only
    # blob) ends up with far sparser points per unit length at the same
    # sample count. Checked directly: even with identical underlying
    # pixels and identical smoothing values, the two contours still
    # visibly diverged near the ear until the point density was also
    # matched -- spline smoothing behaves differently at different point
    # densities along the same physical edge, not just at different
    # smoothing strengths.
    silhouette_contours = smooth_contours(hair_clothes | skin, smoothing=1.5, samples=2000)
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
    # draw_hair_flow_lines disabled per direct feedback ("the hair lines
    # are bad, remove them for now") -- function kept defined, not
    # deleted, since "for now" implies this may come back in a revised
    # form rather than being rejected outright the way the two earlier
    # hair-mark approaches were.

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

        # 0.35x padding is generous enough that on a close-up photo it can
        # reach past the temple hinge into the ear itself -- confirmed
        # directly, the glasses region's x-bounds overshot landmarks
        # 127/356 (the face oval's own temple points, right where cheek
        # meets ear) by ~18px on this photo. Once that happens, the dark
        # temple arm + ear shadow + any hair overlap get morphologically
        # closed into the same blob as the actual lens/frame and traced as
        # one shape, which is what was producing a garbled scribble where
        # the ear should be instead of a clean lens outline. Clamping to
        # just inside those temple landmarks keeps the search wide enough
        # for the lens/frame (which sits well inside them) while making it
        # structurally impossible to reach the ear, regardless of how wide
        # any given photo's eyebrow-to-eyebrow span is relative to it.
        # Widening this margin further (tried 0.11 and 0.18, to keep the
        # box clear of landmark 234 -- almost the same x as landmark 127,
        # so a small margin barely moved the boundary) broke the lens
        # capture itself both times instead of just trimming the temple
        # overreach -- the lens's own edge sits close enough to that
        # boundary on this photo that there's no x-only clamp that
        # separates "lens" from "reaches too far toward the ear." Left at
        # 0.05 (known to capture the lens reliably) and the temple/hair
        # overreach is excluded directly below instead, by category
        # rather than position.
        temple_margin = (max(xs) - min(xs)) * 0.05
        gx0 = max(gx0, landmarks[127].x * w + temple_margin)
        gx1 = min(gx1, landmarks[356].x * w - temple_margin)
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
    # A glasses frame is never classified as HAIR -- excluding that
    # category directly removes any temple-arm/sideburn overreach at its
    # actual source (whatever dark pixels the segmenter itself calls
    # hair) rather than approximating "not hair" with an x-position box,
    # which broke the lens capture itself when drawn wide enough to
    # matter (see temple_margin above).
    glasses_raw = glasses_region & foreground & (gray_norm < 120) & (cat_mask != HAIR)
    if glasses and glasses_raw.any():
        # A lens glare/reflection can break the frame's brightness
        # threshold into two genuinely disconnected pieces with a real
        # gap between them (confirmed directly: bright, up to fully-white
        # pixels sitting right in the middle of the frame line). Closing
        # wide enough to bridge that (~20px) then eroding back down was
        # tried and erased the frame outright -- the frame's own raw line
        # is only a few px thick, so an erosion sized to undo a 9px
        # closing eats straight through it, not just the extra gap-filled
        # width. A small closing first (radius 3*scale, the same one that
        # fixed the over-thick bridge) keeps the frame's real thickness
        # everywhere it isn't broken. Any piece that breaks off near the
        # main frame (a lower area threshold than before, 2% instead of
        # 15%, so a genuine broken-off frame fragment isn't discarded as
        # noise the way the earlier threshold would) gets unioned back in
        # at its own true thickness, and only then does one moderate
        # closing bridge the now-much-smaller remaining gaps between
        # those real pieces -- bridging a short real gap between two
        # already-present fragments, not conjuring 20px of frame from
        # nothing the way closing the raw broken mask directly required.
        glasses_small = ndimage.binary_closing(glasses_raw, structure=disk(max(2, round(3 * scale))))
        labeled, n = ndimage.label(glasses_small)
        if n > 1:
            sizes = ndimage.sum(glasses_small, labeled, range(1, n + 1))
            keep = np.where(sizes >= sizes.max() * 0.02)[0] + 1
            glasses_small = np.isin(labeled, keep)
        glasses_closed = ndimage.binary_closing(glasses_small, structure=disk(max(2, round(8 * scale))))

        # A real glasses frame closes into one or two main blobs (the
        # lenses, joined at the bridge or not); other things this same
        # brightness threshold catches nearby -- most visibly the small
        # dark hinge screw where the frame meets the temple arm -- close
        # into their own separate, much smaller blob rather than merging
        # into the frame, and were getting traced as their own stray loop
        # right next to the lens (confirmed directly against a render:
        # a small teardrop mark hanging off the frame's outer edge, with
        # no such feature in the reference style). Keeping only
        # components at least 15% of the largest one's area drops those
        # without needing the closing kernel large enough to risk
        # re-merging with the ear/temple region next to it.
        labeled, n = ndimage.label(glasses_closed)
        if n > 0:
            sizes = ndimage.sum(glasses_closed, labeled, range(1, n + 1))
            keep = np.where(sizes >= sizes.max() * 0.15)[0] + 1
            glasses_closed = np.isin(labeled, keep)

        glasses_outer, glasses_holes = trace_glasses_contours(glasses_closed)
        # No extra bump over detail_width anymore (was +2) -- that made
        # sense back when detail_width was thin (3) and glasses needed to
        # stand out as visibly bolder, but detail_width itself is now
        # bumped up globally (5), and the extra +2 on top of that read as
        # too thick.
        glasses_width = detail_width
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
    if landmarks is not None:
        out_im = draw_smooth_open_stroke(out_im, list(face_oval_contour(landmarks, w, h)), width=detail_width)
    else:
        out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=detail_width)
    out_im = draw_dark_face_detail(
        out_im, cat_mask, gray, foreground, landmarks, detail_width=detail_width, scale=scale, glasses=glasses
    )
    out = np.array(out_im)

    # Retrace the structural layer only (silhouette/hair/clothes/jaw/
    # glasses) -- see vector_retrace's docstring for why near-duplicate-
    # line seams show up specifically there. The finer facial marks added
    # next (eyebrows, nose, mouth, ear folds) are each already a single,
    # clean supersampled stroke with a proper round cap; retracing them
    # too was actually squaring those caps off instead (vtracer's curve
    # fitting doesn't preserve small circular detail well at this scale,
    # confirmed directly against a render and not fixable by tuning its
    # corner-threshold/speckle-filter parameters), which doesn't match a
    # real reference avatar's rounded stroke ends. Adding them after the
    # retrace, on the by-then-already-clean structural base, keeps both:
    # no seams on the structural layer, real round caps on the marks.
    out = vector_retrace(out)

    if landmarks is not None:
        out = draw_face_structure_lines(out, im, landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h, detail_width=detail_width)
    else:
        print("No face detected, skipping face structure lines and dot-eye replacement")

    out, _ = crop_to_content(out, foreground)
    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def vector_retrace(out: np.ndarray) -> np.ndarray:
    """Re-trace the whole flattened composite as one clean vector (vtracer,
    color mode) and rasterize it back, instead of shipping the raster as
    each feature independently drew it.

    Every feature above is drawn by its own supersample+Lanczos call,
    which anti-aliases each stroke against the *canvas so far*, not
    against every other stroke that will eventually sit next to it. Where
    two independently-drawn regions come very close without actually
    touching (a real, recurring case: the jaw outline and the silhouette
    outline tracing two different masks -- rembg's foreground cutout vs.
    the segmenter's FACE_SKIN category -- that agree closely but not
    pixel-for-pixel), each keeps its own separate anti-aliased edge rather
    than merging into one line, which reads as an unintentional doubled
    line. Retracing the flattened raster sidesteps that structurally: by
    this point there's only one pixel grid with one set of boundaries left
    to find, so two near-coincident edges either have already merged into
    one solid region (drawn as a single clean line) or they were genuinely
    separate shapes to begin with (drawn as two) -- there's no longer a
    "was this one line or two nearly-overlapping lines" ambiguity for the
    tracer to inherit from how the raster was assembled.

    Falls back to the untraced raster if vtracer/cairosvg aren't available
    or the retrace fails for any reason, rather than blocking the whole
    composite on this final polish step."""
    import tempfile
    import os

    try:
        import vtracer
        import cairosvg
    except ImportError as e:
        print(f"vector_retrace skipped (missing dependency: {e})")
        return out

    h, w = out.shape[:2]
    fd_png, png_path = tempfile.mkstemp(suffix=".png")
    os.close(fd_png)
    fd_svg, svg_path = tempfile.mkstemp(suffix=".svg")
    os.close(fd_svg)
    try:
        Image.fromarray(out).save(png_path)
        vtracer.convert_image_to_svg_py(
            png_path, svg_path,
            colormode="color", hierarchical="stacked", mode="spline",
            filter_speckle=4, color_precision=6, layer_difference=16,
            corner_threshold=60, length_threshold=4.0, splice_threshold=45,
            path_precision=3,
        )
        png_bytes = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h, background_color="white")
        from io import BytesIO

        retraced = np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))
        if retraced.shape[:2] != (h, w):
            print(f"vector_retrace: size mismatch after retrace ({retraced.shape[:2]} vs {(h, w)}), keeping raster")
            return out
        return retraced
    except Exception as e:
        print(f"vector_retrace failed ({e}); keeping raster")
        return out
    finally:
        for p in (png_path, svg_path):
            if os.path.exists(p):
                os.unlink(p)


def face_oval_contour(landmarks, w: int, h: int):
    """The real geometric extent of the face from MediaPipe's own 3D face
    mesh (FACE_OVAL), used for the jaw/cheek outline instead of the
    segmenter's FACE_SKIN category (see face_skin_contours, kept as the
    fallback for when no landmarks are available).

    FACE_SKIN correctly excludes the ears -- they aren't face skin -- but
    using that pixel category as the entire face outline pinches the
    whole shape in sharply right at ear height, which reads as a
    narrower, more generic face than the real photo. Confirmed directly:
    even a very large closing (45px) couldn't bridge that pinch, because
    it isn't a small gap to bridge -- there's genuinely almost no
    face-skin-classified pixel immediately beside the ear, so the pinch
    is topologically real in the 2D pixel category even though it isn't
    how the actual head reads (the head isn't narrower there, the ear is
    just a different category). The landmark oval has no such artifact:
    it's not a pixel category boundary at all, just the face mesh's own
    geometric estimate of the face's extent, and a direct check against
    the real photo confirmed it tracks the actual visible face width
    smoothly past the temple instead of pinching.

    Returns an OPEN arc (temple-height down through the chin and back up
    to the other temple), not the full closed oval -- the oval's own
    upper arc crosses the forehead, and unlike FACE_SKIN (which simply
    has no pixels wherever hair actually covers skin) the landmark oval
    has no concept of hair coverage at all, so drawing the full loop drew
    a stray line straight across the forehead through the hair fill
    (confirmed directly against a render). The hairline is already drawn
    separately from the hair silhouette's own contour; this only needs
    to supply the part below that."""
    from scipy.interpolate import splev, splprep

    order = _ordered_face_oval_indices()
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in order])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    xs, ys = splev(np.linspace(0.0, 1.0, 200), tck)
    full = np.stack([xs, ys], axis=1)

    # landmark 168 sits between the eyebrows, right at brow height -- a
    # stable, already-used-elsewhere reference for "above here is
    # forehead, not cheek/temple." Taking the single longest contiguous
    # run of below-that-height points (same technique used for the nose
    # and collar arcs elsewhere) rather than a fixed index range, since
    # FACE_OVAL's point order doesn't start at a guaranteed position.
    cutoff_y = landmarks[168].y * h
    below = full[:, 1] > cutoff_y
    n = len(full)
    idx2 = np.concatenate([np.where(below)[0], np.where(below)[0] + n])
    splits = np.where(np.diff(idx2) > 1)[0]
    run_starts = np.concatenate([[0], splits + 1])
    run_ends = np.concatenate([splits, [len(idx2) - 1]])
    best = np.argmax(run_ends - run_starts)
    best_idx = idx2[run_starts[best]:run_ends[best] + 1] % n
    return full[best_idx]


def face_skin_contours(cat_mask: np.ndarray):
    """Fallback jaw/cheek outline for when no face landmarks are
    available (face_oval_contour is preferred whenever they are -- see
    its docstring for why). The segmenter's own FACE_SKIN category
    traces almost exactly the jawline (see the FACE_SKIN vs BODY_SKIN
    comparison that motivated this) -- a real semantic distinction it
    learned (face vs neck), not a blind local pixel search. Using it
    directly beats reconstructing the same boundary from a landmark
    position and gradient-snapping along a short search line, which has
    no such understanding and can grab a stronger but wrong nearby edge
    (glasses, a collar seam, a shirt pattern) instead of the real,
    sometimes subtle, jaw shadow.

    smoothing=0.5 (up from the function default of 0.15): the segmenter's
    pixel-to-pixel boundary along the jaw has real per-pixel jitter that a
    light smoothing pass leaves visible as a wobble rather than a clean
    curve -- this is the same fix already applied to the hair/clothes/
    silhouette contours below, just also needed here.

    A shadow crossing the face (glasses temple arm, a stray eyebrow-height
    crease) can split FACE_SKIN into two disconnected blobs -- the main
    face mass, and a smaller one near the temple/cheek that's still
    genuinely face skin, just cut off from the rest by a thin dark gap.
    Tracing both separately (the old behavior here) draws the smaller
    blob's own ragged edge as its own closed shape, which is what a
    stray fork/hook near the temple traced back to: not a bug in the
    tracing itself, but two real but disconnected regions both getting
    the same "confident single line" treatment a single simple region
    would. A closing bridges gaps that are actually gaps (not real
    concavities -- a real concavity is wider than a shadow line, and
    survives a small closing), then only the single largest resulting
    blob is kept, since the face is one contiguous region and anything
    else this small at this stage is noise, not a second face part."""
    face_skin = cat_mask == FACE_SKIN
    face_skin = ndimage.binary_closing(face_skin, structure=disk(6))
    labeled, num = ndimage.label(face_skin)
    if num > 1:
        sizes = ndimage.sum(face_skin, labeled, range(1, num + 1))
        face_skin = labeled == (np.argmax(sizes) + 1)
    return smooth_contours(face_skin, min_area_frac=0.01, smoothing=0.5)


def draw_face_structure_lines(out: np.ndarray, im: Image.Image, landmarks, w: int, h: int, detail_width: float = DETAIL_WIDTH) -> np.ndarray:
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
    # One constant width for every facial mark -- eyebrows, nose, mouth,
    # ear fold alike -- instead of each drawn at its own slightly
    # different offset from detail_width (-2 here, -1 there, +1
    # elsewhere) plus some of them tapered and some not. A direct
    # comparison against a reference avatar's face confirmed that's what
    # actually reads as "polished": every stroke on the face is the exact
    # same confident weight, no thin/thick disparity between features and
    # no taper anywhere, not a stylistic choice specific to any one mark.
    line_width = detail_width

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
    # A second round of direct correction against a real photo (this time
    # tracing both eyebrows individually rather than eyeballing) showed the
    # 0.02 value above still sat ~4px too low on one side while matching on
    # the other -- split the difference rather than guess further.
    brow_lift = eye_span * 0.04
    # Thin, only lightly tapered -- the confirmed reference style draws
    # eyebrows as a single simple curved stroke, not a thick filled shape
    # (an earlier version of this went bold/flat based on a different,
    # non-representative pair of avatars -- reverted).
    # Each eyebrow's inner (nose-bridge) end sits close enough to the
    # glasses frame's own rising bridge curve that a uniform brow_lift
    # wasn't enough to clear it there -- confirmed directly against a
    # render, the two independently-traced lines (eyebrow stroke, frame
    # contour) crossed at a shallow angle right at that inner corner,
    # producing a thin doubled "fork" for a few pixels rather than either
    # running clear of the other. An extra lift that only applies near
    # the bridge (decaying to nothing by the outer/temple end, which was
    # already clear) pushes just that corner up without changing the
    # eyebrow's shape or position anywhere else.
    # Lifting the inner end (tried first) reduced but didn't reliably
    # clear the fork -- with two independently-traced lines (this stroke,
    # the frame contour from a completely separate brightness threshold),
    # no amount of nudging one guarantees they won't still graze each
    # other at some point along their length, and each near-miss shows up
    # as a thin doubled seam (their anti-aliased edges sit adjacent but
    # not identical, rather than one clean overlap). Trimming the
    # eyebrow's own inner-most point instead removes the possibility
    # entirely -- it just doesn't reach far enough inward to be near the
    # frame at all, leaving a small natural gap rather than a near-miss.
    bridge_x = landmarks[168].x * w
    for conns in (connections.FACE_LANDMARKS_LEFT_EYEBROW, connections.FACE_LANDMARKS_RIGHT_EYEBROW):
        raw_pts = ordered_points(conns)
        # Whichever end (start or end of the walk) sits closer to the
        # nose bridge is the inner end -- drop its last one or two points.
        if abs(raw_pts[0][0] - bridge_x) < abs(raw_pts[-1][0] - bridge_x):
            raw_pts = raw_pts[2:]
        else:
            raw_pts = raw_pts[:-2]
        pts = [(px, py - brow_lift) for px, py in raw_pts]
        img = draw_smooth_open_stroke(img, pts, width=line_width)

    # Just the line under the nose (nostril hook to nostril hook), not the
    # full nose mesh (bridge + nostril wings + tip outline) -- matches the
    # reference avatars, which only ever mark the nose with a single simple
    # under-nose curve that hooks up at each nostril. A raw polyline through
    # these landmarks has visible straight-segment kinks at each point; a
    # light spline keeps the actual up-down-up-down nostril shape (unlike a
    # heavier smoothing, which averages it into one plain arc) while making
    # it read as one fluid stroke.
    from scipy.interpolate import splev, splprep

    # Landmark-index guessing for this line went through many rounds
    # (guessed indices sitting above the real crease, then a chain that
    # reached too far and closed into a full loop, etc.) because a fixed
    # set of point indices is only ever an approximation of where the
    # nose bottom actually falls in a specific photo. parse_face_regions()
    # + vector_trace_bottom_arc() sidestep that: they get the parser's own
    # measured "nose" pixel mask for THIS photo and vector-trace its real
    # boundary, so the curve is this face's actual nose shape, not a
    # generic point scheme's approximation of it. Falls back to the old
    # landmark spline if parsing fails for any reason (e.g. a face the
    # parser can't crop/detect well) rather than raising.
    try:
        parsing_mask, parsing_crop_box = parse_face_regions(im, landmarks, w, h)
        nose_pts = vector_trace_bottom_arc(parsing_mask, FACE_PARSING_CLASS["nose"], parsing_crop_box)
        # A light re-smoothing pass -- vtracer's own trace is already a
        # spline fit, but it's fit to the parsing mask's real boundary
        # noise (the mask itself has small pixel-level jitter), which
        # shows up as a slightly wavy rather than fluid line. A small
        # nonzero smoothing factor (not the exact-fit s=0 used elsewhere)
        # softens just that small-scale waviness without changing the
        # curve's actual position or overall shape -- confirmed against a
        # hand-drawn reference showing the nose as one smooth stroke.
        nose_tck, _ = splprep([nose_pts[:, 0], nose_pts[:, 1]], s=len(nose_pts) * 0.03, k=3)
        xs, ys = splev(np.linspace(0.0, 1.0, len(nose_pts)), nose_tck)
    except Exception as e:
        print(f"Face-parsing nose trace failed ({e}); falling back to landmark spline")
        nose_bottom_idx = [49, 129, 64, 98, 97, 2, 326, 327, 294, 358, 279]
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in nose_bottom_idx])
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
        xs, ys = splev(np.linspace(0.0, 1.0, 60), tck)

    # Only trims a sliver at the very tip (numerical noise from vtracer's
    # own extrema detection), not a real fraction of the width. A
    # hand-drawn M-shape overlay -- red desired curve vs. the yellow/blue
    # rendered one, on the actual photo -- showed the rendered line
    # undershooting at both ends compared to the reference, twice: first
    # explained as "reach nostril-wing landmark height" (trim 0.1 -> 0.02,
    # reverted as still not tall enough / wrong read), then re-explained
    # directly as "should go higher, to make this M shape, where the
    # outside lines are shorter" -- i.e. short horizontally but tall,
    # not a long gradual trace up the side of the nose. A trim sweep
    # confirmed the untrimmed arc's own natural hook (no trim at all)
    # is what matches that reference: it rises sharply right where the
    # nostril meets the cheek, short and steep, which a 0.1 or even 0.02
    # fractional trim was cutting away. xs isn't necessarily sorted (it
    # follows the traced arc's real path order), so filter by x-range
    # rather than by index; the arc is monotonic enough in x that this
    # still leaves a contiguous, sensibly-ordered run for the stroke
    # below to draw as one continuous line.
    xs, ys = np.asarray(xs), np.asarray(ys)
    x_span = xs.max() - xs.min()
    keep = (xs > xs.min() + x_span * 0.01) & (xs < xs.max() - x_span * 0.01)
    if keep.sum() >= 2:
        xs, ys = xs[keep], ys[keep]
    # Constant-width with round caps, not tapered -- a direct correction
    # said the tapered facial marks (thin-to-thick-to-thin) read wrong,
    # the lines should be a consistent circular-cap stroke throughout
    # instead. draw_smooth_open_stroke already draws round caps (an
    # ellipse at each endpoint), it just doesn't vary the width along
    # the path the way draw_tapered_stroke does.
    img = draw_smooth_open_stroke(img, list(zip(xs, ys)), width=line_width)

    # No philtrum tick -- removed per direct feedback ("that weird line
    # below the nose").

    # Mouth: a single curve through the real outer-lip landmarks (mouth
    # corners to cupid's bow). An earlier version added a second, shorter
    # "lower lip" line -- the same curve's own midsection duplicated and
    # shifted straight down -- meant to keep the mouth from reading as
    # flat/expressionless. In practice that fabricated segment created a
    # downward-pointing dip in the center that made the whole mouth read
    # as a frown on a photo that isn't one, even though the underlying
    # landmark curve itself is fine (this subject's real upper-lip curve
    # is close to level, i.e. a relaxed smile, not deeply arced either
    # way) -- removed rather than patched again, since the real fix is to
    # not draw a line that has no basis in actual landmark geometry.
    # A direct correction against a real photo measured the whole mouth
    # sitting about 5% of eye_span too high here versus its actual
    # position -- shift it down by that amount rather than just the
    # landmarks' raw position.
    # A second correction (0.09) was measured against a stale, unshifted
    # reference overlay and effectively double-counted the shift; a third
    # round, measured against the actually-corrected overlay, showed 0.09
    # overshooting by ~8px. A fourth round then showed the corrected 0.005
    # undershooting by ~6px, converging on 0.07. A fifth round then showed
    # 0.07 sitting ~4px too low, concentrated on the right side -- the raw
    # landmark corners (61, 291) aren't at the same height on this photo,
    # so the whole curve carries a rightward downward tilt the traces
    # never showed. Leveled that out (subtract the corner-to-corner linear
    # trend, re-add the flat average of both corner heights) before
    # applying the reduced shift.
    # A sixth round pointed out the actual problem underneath all of the
    # above: this was tracing the OUTER upper lip (its top edge, the
    # vermillion border), not the seam where the lips actually meet --
    # every previous "shift" was really just a rough correction trying to
    # approximate seam position from outer-lip landmarks. Switched to the
    # inner-lip contour, which sits directly on that seam, so no
    # artificial shift should be needed anymore.
    mouth_shift = 0.0
    upper_lip_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    raw_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in upper_lip_idx])
    corner_l, corner_r = raw_pts[0], raw_pts[-1]
    avg_corner_y = (corner_l[1] + corner_r[1]) / 2
    t = (raw_pts[:, 0] - corner_l[0]) / (corner_r[0] - corner_l[0])
    trend = corner_l[1] * (1 - t) + corner_r[1] * t
    leveled_y = raw_pts[:, 1] - trend + avg_corner_y
    # The same third round's annotation also showed visible waviness along
    # the seam (natural lip texture) that the leveled curve, drawn through
    # only 11 points and splined exactly, was smoothing away -- a light
    # synthetic ripple restores that texture without needing more landmarks.
    # Reduced from 0.012 -- direct comparison against the nose line right
    # above it showed the ripple's wobble was making the mouth read as
    # thin/uncertain next to the nose's confident taper, an optical effect
    # of the wobble rather than the actual stroke width. A gentler ripple
    # keeps the lip texture without undercutting the stroke's confidence.
    ripple = np.sin(t * np.pi * 3) * (eye_span * 0.006)
    upts = np.stack([raw_pts[:, 0], leveled_y + ripple + mouth_shift], axis=1)
    utck, _ = splprep([upts[:, 0], upts[:, 1]], s=0, k=3)
    uxs, uys = splev(np.linspace(0, 1, 40), utck)
    # Constant-width with round caps, not tapered -- same direct
    # correction as the nose above: a consistent circular-cap stroke,
    # not a thin-to-thick-to-thin taper.
    img = draw_smooth_open_stroke(img, list(zip(uxs, uys)), width=line_width)

    # A small interior fold line in each visible ear -- reference avatars
    # consistently mark a single curve suggesting the tragus/antihelix
    # rather than leaving the ear as a bare outline, which reads as flat
    # and empty by comparison. There's no MediaPipe ear landmark to trace
    # a real fold from, so this is a stylized accent (like the philtrum
    # tick above) sized and anchored off the nearest face-oval landmark
    # (234/454, right at the cheek/ear boundary) rather than a claim about
    # this specific ear's actual anatomy -- small and centered enough on
    # that landmark to stay inside the visible ear silhouette rather than
    # risk poking out into the hair or cheek.
    for lid, side in ((234, -1), (454, 1)):
        lm = landmarks[lid]
        ex, ey = lm.x * w, lm.y * h
        fold_len = eye_span * 0.09
        fold_bow = eye_span * 0.012
        t = np.linspace(-1.0, 1.0, 5)
        fold_pts = [(ex + side * fold_bow * (1 - tt**2), ey + tt * fold_len * 0.5) for tt in t]
        img = draw_smooth_open_stroke(img, fold_pts, width=line_width)

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
    if landmarks is not None:
        out_im = draw_smooth_open_stroke(out_im, list(face_oval_contour(landmarks, w, h)), width=detail_width)
    else:
        out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=detail_width)
    out = np.array(out_im)

    if landmarks is not None:
        out = draw_face_structure_lines(out, im, landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h, detail_width=detail_width)
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
    if landmarks is not None:
        out_im = draw_smooth_open_stroke(out_im, list(face_oval_contour(landmarks, w, h)), width=detail_width)
    else:
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
        out = draw_face_structure_lines(np.array(out_im), im, landmarks, w, h, detail_width=detail_width)
        out = draw_dot_eyes(out, landmarks, w, h, detail_width=detail_width)
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
