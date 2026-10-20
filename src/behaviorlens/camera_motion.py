"""Global (camera) motion between consecutive sampled frames.

Pose landmarks are in image coordinates, so a pan, tilt or zoom changes a person's
apparent posture and position even when they do not move. This module estimates the
background motion between consecutive sampled frames so the rules can treat frames with
a strongly moving camera as unobservable for posture instead of guessing.

Method (per pair of consecutive sampled frames, on a downscaled grayscale copy):

1. Corners (``cv2.goodFeaturesToTrack``) are detected in the previous frame **outside** the
   primary person's landmark bounding box (padded), so subject movement is not mistaken
   for camera motion.
2. They are tracked into the current frame with pyramidal Lucas-Kanade
   (``cv2.calcOpticalFlowPyrLK``) and checked forward-backward; matches landing inside the
   current person box are dropped.
3. A RANSAC similarity transform (``cv2.estimateAffinePartial2D``) gives translation of the
   image centre, rotation and scale. The inlier ratio is kept as a reliability cue.
4. With too few features or inliers, phase correlation on the frames (person box blanked)
   gives a translation-only estimate if its peak response is strong enough.

Otherwise the frame's motion is unknown. Unknown motion never masks anything: missing
evidence about the camera is not evidence that it moved.

Raw estimates are stored with the landmarks (``PoseSequence.camera_motion``); the
static/moving decision uses ``RuleParams`` thresholds, so changing them never needs
re-extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import RuleParams
from .models import (
    MOTION_CHANNELS,
    MOTION_DX,
    MOTION_DY,
    MOTION_INLIERS,
    MOTION_LOG_SCALE,
    MOTION_METHOD,
    MOTION_ROT_DEG,
    VISIBILITY,
    X,
    Y,
    PoseSequence,
)

# Bump when the estimator changes in a way that alters stored values (part of pose cache keys).
CAMERA_MOTION_VERSION = "cm1"

# Tri-state per-frame camera flag.
STATIC, MOVING, UNKNOWN_MOTION = 0, 1, -1
# Values of the MOTION_METHOD channel.
METHOD_NONE, METHOD_FEATURES, METHOD_PHASE = 0, 1, 2


@dataclass(frozen=True)
class MotionConfig:
    """Estimator settings. Pixel values refer to the downscaled frame."""

    work_width: int = 320
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 7
    lk_win: int = 21
    lk_levels: int = 3
    fb_max_err_px: float = 1.0  # forward-backward consistency
    ransac_px: float = 1.5
    min_matches: int = 12
    min_inliers: int = 8
    person_pad: float = 0.15  # bbox padding as a fraction of its size
    landmark_visibility: float = 0.3
    phase_min_response: float = 0.15


Box = tuple[float, float, float, float]

# OpenCV's stubs require an ndarray for ``nextPts``; the real API accepts None (no initial
# guess). Typed as Any so strict mypy accepts the documented call.
_lk_flow: Any = cv2.calcOpticalFlowPyrLK


def person_box(landmarks: np.ndarray | None, pad: float, min_visibility: float) -> Box | None:
    """Padded normalized (x0, y0, x1, y1) box around a (33, 5) pose, or None."""
    if landmarks is None:
        return None
    vis = np.nan_to_num(landmarks[:, VISIBILITY], nan=0.0) >= min_visibility
    if not vis.any():
        return None
    xs, ys = landmarks[vis, X], landmarks[vis, Y]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # Landmarks cover joints, not the silhouette: pad generously (head top, hands, clothing).
    px, py = pad * max(x1 - x0, 0.05) + 0.03, pad * max(y1 - y0, 0.05) + 0.05
    return (max(0.0, x0 - px), max(0.0, y0 - py), min(1.0, x1 + px), min(1.0, y1 + py))


def _box_px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return int(x0 * w), int(y0 * h), int(np.ceil(x1 * w)), int(np.ceil(y1 * h))


class CameraMotionEstimator:
    """Stateful estimator: call ``update`` once per sampled frame, in order.

    ``update`` returns one row of ``MOTION_CHANNELS`` values describing the motion from
    the previous sampled frame to this one (all NaN and method 0 for the first frame or
    when no reliable estimate exists).
    """

    def __init__(self, cfg: MotionConfig | None = None):
        self.cfg = cfg or MotionConfig()
        self._prev: np.ndarray | None = None
        self._prev_boxes: list[Box] = []

    def reset(self) -> None:
        self._prev, self._prev_boxes = None, []

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.cfg.work_width / w)
        small = cv2.resize(frame_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def update(self, frame_bgr: np.ndarray,
               landmarks: np.ndarray | Sequence[np.ndarray] | None = None) -> np.ndarray:
        """``landmarks``: one person's (33, 5) array, or every detected person's. All people
        are excluded from the background fit, so nobody's movement reads as camera motion."""
        gray = self._prepare(frame_bgr)
        people = [] if landmarks is None else ([landmarks] if isinstance(landmarks, np.ndarray) else list(landmarks))
        boxes = [b for b in (person_box(lm, self.cfg.person_pad, self.cfg.landmark_visibility)
                             for lm in people) if b is not None]
        row = np.full(MOTION_CHANNELS, np.nan)
        row[MOTION_METHOD] = METHOD_NONE
        if self._prev is not None and self._prev.shape == gray.shape:
            row = self._estimate(self._prev, gray, self._prev_boxes, boxes)
        self._prev, self._prev_boxes = gray, boxes
        return row

    # ------------------------------------------------------------------ internals
    def _estimate(self, prev: np.ndarray, cur: np.ndarray, prev_boxes: list[Box],
                  cur_boxes: list[Box]) -> np.ndarray:
        cfg = self.cfg
        h, w = prev.shape
        row = np.full(MOTION_CHANNELS, np.nan)
        row[MOTION_METHOD] = METHOD_NONE
        mask = np.full((h, w), 255, np.uint8)
        for box in prev_boxes:
            x0, y0, x1, y1 = _box_px(box, w, h)
            mask[y0:y1, x0:x1] = 0
        pts = cv2.goodFeaturesToTrack(prev, cfg.max_corners, cfg.quality_level, cfg.min_distance, mask=mask)
        if pts is not None and len(pts) >= cfg.min_matches:
            lk: dict[str, Any] = dict(winSize=(cfg.lk_win, cfg.lk_win), maxLevel=cfg.lk_levels,
                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
            nxt, st1, _ = _lk_flow(prev, cur, pts, None, **lk)
            back, st2, _ = _lk_flow(cur, prev, nxt, None, **lk)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1)
            ok &= np.linalg.norm((back - pts).reshape(-1, 2), axis=1) <= cfg.fb_max_err_px
            q = nxt.reshape(-1, 2)
            for box in cur_boxes:
                x0, y0, x1, y1 = _box_px(box, w, h)
                ok &= ~((q[:, 0] >= x0) & (q[:, 0] < x1) & (q[:, 1] >= y0) & (q[:, 1] < y1))
            src, dst = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
            if len(src) >= cfg.min_matches:
                m, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=cfg.ransac_px,
                                                     maxIters=500, confidence=0.99)
                n_in = int(inl.sum()) if inl is not None else 0
                if m is not None and n_in >= cfg.min_inliers:
                    a, b = m[0, 0], m[1, 0]
                    centre = np.array([w / 2, h / 2])
                    moved = m[:, :2] @ centre + m[:, 2] - centre
                    row[MOTION_DX], row[MOTION_DY] = moved[0] / w, moved[1] / h
                    row[MOTION_ROT_DEG] = np.degrees(np.arctan2(b, a))
                    row[MOTION_LOG_SCALE] = np.log(max(np.hypot(a, b), 1e-6))
                    row[MOTION_INLIERS] = n_in / len(src)
                    row[MOTION_METHOD] = METHOD_FEATURES
                    return row
        return self._phase(prev, cur, prev_boxes + cur_boxes, row)

    def _phase(self, prev: np.ndarray, cur: np.ndarray, boxes: list[Box], row: np.ndarray) -> np.ndarray:
        h, w = prev.shape
        a, b = prev.astype(np.float32), cur.astype(np.float32)
        for img in (a, b):
            for box in boxes:
                x0, y0, x1, y1 = _box_px(box, w, h)
                img[y0:y1, x0:x1] = float(img.mean())
        if a.std() < 2.0 or b.std() < 2.0:  # textureless: nothing to correlate
            return row
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (sx, sy), response = cv2.phaseCorrelate(a, b, win)
        if response < self.cfg.phase_min_response:
            return row
        row[MOTION_DX], row[MOTION_DY] = sx / w, sy / h
        row[MOTION_ROT_DEG], row[MOTION_LOG_SCALE] = 0.0, 0.0
        row[MOTION_METHOD] = METHOD_PHASE
        return row


def empty_motion(n: int) -> np.ndarray:
    out = np.full((n, MOTION_CHANNELS), np.nan)
    out[:, MOTION_METHOD] = METHOD_NONE
    return out


# ---------------------------------------------------------------------- classification
def motion_rates(seq: PoseSequence) -> dict[str, np.ndarray] | None:
    """Per-frame motion rates (per second) or None if the sequence has no motion data."""
    cm = seq.camera_motion
    if cm is None or len(seq) == 0:
        return None
    dt = np.diff(seq.timestamps_s, prepend=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "translation_per_s": np.hypot(cm[:, MOTION_DX], cm[:, MOTION_DY]) / dt,
            "rotation_deg_per_s": np.abs(cm[:, MOTION_ROT_DEG]) / dt,
            "scale_change_per_s": np.abs(cm[:, MOTION_LOG_SCALE]) / dt,
        }


def camera_motion_state(seq: PoseSequence, params: RuleParams) -> np.ndarray:
    """STATIC / MOVING / UNKNOWN_MOTION per frame.

    A frame is MOVING when a reliable estimate exceeds any rate threshold, STATIC when a
    reliable estimate exceeds none, and UNKNOWN otherwise (first frame, no texture, too few
    features, low RANSAC inlier ratio, or a sequence without motion data).
    """
    state = np.full(len(seq), UNKNOWN_MOTION)
    rates = motion_rates(seq)
    if rates is None:
        return state
    cm = seq.camera_motion
    assert cm is not None  # motion_rates returned rates, so motion data exists
    method = cm[:, MOTION_METHOD]
    inliers = cm[:, MOTION_INLIERS]
    reliable = (method == METHOD_PHASE) | (
        (method == METHOD_FEATURES) & (np.nan_to_num(inliers) >= params.camera_min_inlier_ratio))
    reliable &= ~np.isnan(rates["translation_per_s"])
    with np.errstate(invalid="ignore"):
        moving = ((rates["translation_per_s"] > params.camera_max_translation_per_s)
                  | (rates["rotation_deg_per_s"] > params.camera_max_rotation_deg_per_s)
                  | (rates["scale_change_per_s"] > params.camera_max_scale_change_per_s))
    state[reliable] = STATIC
    state[reliable & moving] = MOVING
    return state


def camera_moving_mask(seq: PoseSequence, params: RuleParams) -> np.ndarray:
    """Frames to treat as unobservable for posture: MOVING frames, widened by
    ``camera_motion_pad_frames`` on each side (motion is measured between frame k-1 and k,
    and pose at the boundary frames is affected by blur and the in-between viewpoint)."""
    if not params.mask_camera_motion:
        return np.zeros(len(seq), bool)
    moving = camera_motion_state(seq, params) == MOVING
    pad = int(params.camera_motion_pad_frames)
    if pad > 0 and moving.any():
        kernel = np.ones(2 * pad + 1)
        moving = np.convolve(moving.astype(float), kernel, mode="same") > 0
    return np.asarray(moving, dtype=bool)


def camera_motion_stats(seq: PoseSequence, params: RuleParams) -> dict[str, Any]:
    """Summary used in analysis stats, warnings and the UI timeline lane."""
    n = len(seq)
    if seq.camera_motion is None or n == 0:
        return {"camera_motion_available": False, "camera_moving_ratio": None,
                "camera_motion_unknown_ratio": None, "camera_motion_intervals": []}
    state = camera_motion_state(seq, params)
    moving = state == MOVING
    intervals = []
    idx = np.flatnonzero(moving)
    if idx.size:
        # Frame k's motion happened between samples k-1 and k.
        starts = idx[np.r_[True, np.diff(idx) > 1]]
        ends = idx[np.r_[np.diff(idx) > 1, True]]
        for s, e in zip(starts, ends):
            t0 = float(seq.timestamps_s[s - 1]) if s > 0 else float(seq.timestamps_s[s])
            intervals.append([round(t0, 3), round(float(seq.timestamps_s[e]), 3)])
    return {
        "camera_motion_available": True,
        "camera_moving_ratio": round(float(moving.sum()) / n, 3),
        "camera_motion_unknown_ratio": round(float(np.sum(state == UNKNOWN_MOTION)) / n, 3),
        "camera_motion_intervals": intervals,
    }
