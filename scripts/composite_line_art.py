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


def draw_smooth_strokes(canvas: Image.Image, contours, width: int = 4, supersample: int = 4) -> Image.Image:
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


def draw_smooth_fills(canvas: Image.Image, contours, fill=(0, 0, 0), supersample: int = 4) -> Image.Image:
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
        corrected[y0:y1, x0:x1][hair_window & skin_like] = False

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

    out_im = Image.fromarray(np.full(gray.shape + (3,), 255, dtype=np.uint8))

    # Fill and stroke hair/clothes (and the outer silhouette, for the parts
    # of the edge where skin is directly visible against the background,
    # e.g. jaw/cheek) from smoothed contours throughout, not a raw pixel
    # mask -- a single fluid line that still follows this specific photo's
    # actual shape, instead of a stair-stepped edge. hair_clothes.min_area
    # is kept low since hair and clothes are frequently two disconnected
    # blobs, split by a visible neck, and both matter.
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
    hair_clothes_outer, hair_clothes_holes = smooth_contours(
        hair_clothes, min_area_frac=0.001, include_holes=True, smoothing=1.5
    )
    silhouette_contours = smooth_contours(foreground, smoothing=1.5)

    out_im = draw_smooth_fills(out_im, hair_clothes_outer)
    out_im = draw_smooth_fills(out_im, hair_clothes_holes, fill=(255, 255, 255))
    out_im = draw_smooth_strokes(out_im, silhouette_contours, width=6)
    out_im = draw_smooth_strokes(out_im, hair_clothes_outer, width=6)
    out_im = draw_smooth_strokes(out_im, hair_clothes_holes, width=6)

    return np.array(out_im), foreground, gray, hair_clothes


def composite_line_art(photo_path: Path, out_path: Path):
    print("Segmenting photo...")
    im = Image.open(photo_path).convert("RGB")
    cat_mask = segment(im)
    print("Detecting face landmarks...")
    landmarks = get_face_landmarks(im)
    out, foreground, gray, hair_clothes = _base_layers(im, cat_mask, landmarks)

    skin_mask = (cat_mask == FACE_SKIN) | (cat_mask == BODY_SKIN)
    fg_vals = gray[foreground]
    lo, hi = np.percentile(fg_vals, [2, 98]) if fg_vals.size else (0, 255)
    gray_norm = np.clip((gray - lo) / max(hi - lo, 1) * 255, 0, 255)
    # Erode away a margin near the skin's own *outer* edge before looking
    # for dark detail -- directional lighting casts a real shadow there
    # (falling off toward the side of the face away from the light) that
    # a brightness threshold can't tell apart from an actual feature.
    # Real features (eyebrows, pupils, beard texture) sit more centrally
    # on the face, so they survive the erosion; a shadow gradient hugging
    # the boundary doesn't.
    #
    # Erode a hole-filled copy, not skin_mask directly: skin_mask already
    # has internal gaps wherever something (like glasses) covers the skin,
    # and eroding it directly pulls back from those internal edges too --
    # eating exactly the glasses detail this is supposed to leave alone.
    filled_skin = ndimage.binary_fill_holes(skin_mask)
    interior_skin = ndimage.binary_erosion(filled_skin, structure=disk(18)) & skin_mask
    dark_mask = interior_skin & (gray_norm < 120)

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

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=5)
    out = np.array(out_im)

    if landmarks is not None:
        h, w = gray.shape
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping dot-eye replacement")

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
    jaw shadow."""
    return smooth_contours(cat_mask == FACE_SKIN, min_area_frac=0.01)


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

    draw_conns(connections.FACE_LANDMARKS_LEFT_EYEBROW, width=3)
    draw_conns(connections.FACE_LANDMARKS_RIGHT_EYEBROW, width=3)
    draw_conns(connections.FACE_LANDMARKS_NOSE, width=3)

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

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=5)
    out = np.array(out_im)

    if landmarks is not None:
        h, w = gray.shape
        out = draw_face_structure_lines(out, landmarks, w, h)
        out = draw_dot_eyes(out, landmarks, w, h)
    else:
        print("No face detected, skipping face structure lines")

    Image.fromarray(out).save(out_path)
    print(f"Saved to {out_path}")


def scaffold_composite(photo_path: Path, out_path: Path, mask_path: Path):
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

    out_im = Image.fromarray(out)
    out_im = draw_smooth_strokes(out_im, face_skin_contours(cat_mask), width=5)
    out = np.array(out_im)

    # The editable region is simply the segmenter's own FACE_SKIN area --
    # already exactly the visible skin, not hair-covered forehead or
    # anything outside the just-drawn outline.
    mask = cat_mask == FACE_SKIN

    Image.fromarray(out).save(out_path)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    print(f"Saved scaffold to {out_path}, mask to {mask_path}")


def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: composite_line_art.py path/to/photo.jpg [out.png] [--structure|--scaffold]"
        )

    structure_mode = "--structure" in sys.argv
    scaffold_mode = "--scaffold" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--structure", "--scaffold")]

    photo_path = Path(args[0])
    default_suffix = "_structure" if structure_mode else "_scaffold" if scaffold_mode else "_lineart"
    out_path = Path(args[1]) if len(args) > 1 else photo_path.with_name(
        photo_path.stem + default_suffix + ".png"
    )

    if scaffold_mode:
        mask_path = out_path.with_name(out_path.stem + "_mask.png")
        scaffold_composite(photo_path, out_path, mask_path)
    elif structure_mode:
        structure_composite(photo_path, out_path)
    else:
        composite_line_art(photo_path, out_path)


if __name__ == "__main__":
    main()
