"""Per-frame geometric features and candidate states derived from pose landmarks.

Candidate states are tri-state: True / False when the relevant landmarks are
visible, and None ("unobservable") otherwise. Missing evidence never becomes a
forced classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_motion import camera_moving_mask
from .config import RuleParams
from .models import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    VISIBILITY,
    X,
    Y,
    PoseSequence,
)

# Tri-state per-frame values: 1 = True, 0 = False, -1 = unobservable.
TRUE, FALSE, UNKNOWN = 1, 0, -1


# Landmarks this far outside the image are extrapolated, not observed (LEARNINGS M16).
IN_FRAME_TOLERANCE = 0.01


def _visible(seq: PoseSequence, idx: int, thr: float) -> np.ndarray:
    """Confidently detected AND inside the image."""
    vis = np.nan_to_num(seq.landmarks[:, idx, VISIBILITY], nan=0.0) >= thr
    x, y = seq.landmarks[:, idx, X], seq.landmarks[:, idx, Y]
    lo, hi = -IN_FRAME_TOLERANCE, 1 + IN_FRAME_TOLERANCE
    with np.errstate(invalid="ignore"):
        inside = (x >= lo) & (x <= hi) & (y >= lo) & (y <= hi)
    return vis & inside


def _xy(seq: PoseSequence, idx: int) -> np.ndarray:
    return seq.landmarks[:, idx, [X, Y]]


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of 2D points."""
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=1) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclass
class HandRaiseFeatures:
    left: np.ndarray  # tri-state per frame
    right: np.ndarray
    # Wrist height above the same-side shoulder in head units (positive = above).
    left_margin: np.ndarray
    right_margin: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        """Raised if either side is raised; unknown only if neither side is observable."""
        out = np.full(len(self.left), FALSE)
        out[(self.left == UNKNOWN) & (self.right == UNKNOWN)] = UNKNOWN
        out[(self.left == TRUE) | (self.right == TRUE)] = TRUE
        return out


def head_unit(seq: PoseSequence, params: RuleParams) -> np.ndarray:
    """Vertical nose-to-shoulder-line distance: a body-relative scale that stays
    available when hips are hidden (e.g. seated at a table). NaN if unobservable."""
    thr = params.visibility_threshold
    ok = _visible(seq, NOSE, thr) & _visible(seq, LEFT_SHOULDER, thr) & _visible(seq, RIGHT_SHOULDER, thr)
    mid_sh_y = (seq.landmarks[:, LEFT_SHOULDER, Y] + seq.landmarks[:, RIGHT_SHOULDER, Y]) / 2
    unit = mid_sh_y - seq.landmarks[:, NOSE, Y]
    return np.where(ok & (np.nan_to_num(unit) > 1e-3), unit, np.nan)


def hand_raise_features(seq: PoseSequence, params: RuleParams) -> HandRaiseFeatures:
    thr = params.visibility_threshold
    unit = head_unit(seq, params)
    sides = {}
    for name, sh, wr in (("left", LEFT_SHOULDER, LEFT_WRIST), ("right", RIGHT_SHOULDER, RIGHT_WRIST)):
        observable = _visible(seq, sh, thr) & _visible(seq, wr, thr) & ~np.isnan(unit)
        # Image y grows downwards, so a raised wrist has a *smaller* y than the shoulder.
        with np.errstate(invalid="ignore"):
            margin = (seq.landmarks[:, sh, Y] - seq.landmarks[:, wr, Y]) / unit
        state = np.where(margin > params.hand_raise_min_height, TRUE, FALSE)
        state[~observable] = UNKNOWN
        sides[name] = (state, np.where(observable, margin, np.nan))
    return HandRaiseFeatures(
        left=sides["left"][0],
        right=sides["right"][0],
        left_margin=sides["left"][1],
        right_margin=sides["right"][1],
    )


# Posture states for standing_transition.
SEATED, STANDING, AMBIGUOUS = 1, 2, 0


@dataclass
class PostureFeatures:
    state: np.ndarray  # SEATED / STANDING / AMBIGUOUS / UNKNOWN per frame
    hip_knee_drop: np.ndarray  # (knee_y - hip_y) / torso_length
    torso_tilt_deg: np.ndarray
    knee_angle_deg: np.ndarray  # NaN when ankles are not visible
    hip_angle_deg: np.ndarray  # smaller of the two shoulder-hip-knee angles


def posture_features(seq: PoseSequence, params: RuleParams) -> PostureFeatures:
    """Classify each frame as seated-like, standing-like, ambiguous or unobservable.

    rules-v0.2 cues (thresholds chosen on the Charades tuning split; see docs/VALIDATION.md):

    * seated-like: hips flexed (shoulder-hip-knee angle <= seated_max_hip_angle_deg), OR knees
      bent (hip-knee-ankle <= seated_max_knee_angle_deg, when ankles are visible), OR both thighs
      near horizontal/foreshortened (per-leg hip-to-knee drop <= seated_max_drop, the frontal-view
      cue); and in all cases no leg hanging straight down (max drop < seated_drop_ceiling).
    * standing-like: hips extended (>= standing_min_hip_angle_deg), both thighs pointing down
      (min drop >= standing_min_drop), torso upright, knees straight when ankles are visible.

    Requires both shoulders, hips and knees to be visible and in frame; upper-body visibility
    alone never implies sitting. Frames matching both or neither description are ambiguous.
    Frames where the camera is moving (camera_motion.camera_moving_mask) are unobservable.
    """
    thr = params.visibility_threshold
    required = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]
    observable = np.all([_visible(seq, i, thr) for i in required], axis=0)

    mid_sh = (_xy(seq, LEFT_SHOULDER) + _xy(seq, RIGHT_SHOULDER)) / 2
    mid_hip = (_xy(seq, LEFT_HIP) + _xy(seq, RIGHT_HIP)) / 2
    torso = mid_hip - mid_sh
    torso_len = np.linalg.norm(torso, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Per-leg drops: both thighs must agree, so a lunge is neither seated nor standing.
        drop_l = (seq.landmarks[:, LEFT_KNEE, Y] - seq.landmarks[:, LEFT_HIP, Y]) / torso_len
        drop_r = (seq.landmarks[:, RIGHT_KNEE, Y] - seq.landmarks[:, RIGHT_HIP, Y]) / torso_len
        drop = (drop_l + drop_r) / 2
        tilt = np.degrees(np.arctan2(np.abs(torso[:, 0]), torso[:, 1]))
    observable &= np.nan_to_num(torso_len) > 1e-3

    hip_angle = np.fmin(
        _angle_deg(_xy(seq, LEFT_SHOULDER), _xy(seq, LEFT_HIP), _xy(seq, LEFT_KNEE)),
        _angle_deg(_xy(seq, RIGHT_SHOULDER), _xy(seq, RIGHT_HIP), _xy(seq, RIGHT_KNEE)),
    )
    ankles_visible = _visible(seq, LEFT_ANKLE, thr) & _visible(seq, RIGHT_ANKLE, thr)
    knee_l = _angle_deg(_xy(seq, LEFT_HIP), _xy(seq, LEFT_KNEE), _xy(seq, LEFT_ANKLE))
    knee_r = _angle_deg(_xy(seq, RIGHT_HIP), _xy(seq, RIGHT_KNEE), _xy(seq, RIGHT_ANKLE))
    knee = np.where(ankles_visible, np.minimum(knee_l, knee_r), np.nan)

    with np.errstate(invalid="ignore"):
        max_drop, min_drop = np.fmax(drop_l, drop_r), np.fmin(drop_l, drop_r)
        knees_bent = ~np.isnan(knee) & (knee <= params.seated_max_knee_angle_deg)
        seated = observable & (max_drop < params.seated_drop_ceiling) & (
            (hip_angle <= params.seated_max_hip_angle_deg) | knees_bent | (max_drop <= params.seated_max_drop))
        legs_straight = np.isnan(knee) | (knee >= params.standing_min_knee_angle_deg)
        standing = (observable & (tilt <= params.max_torso_tilt_deg) & legs_straight
                    & (hip_angle >= params.standing_min_hip_angle_deg)
                    & (min_drop >= params.standing_min_drop))

    state = np.full(len(seq), AMBIGUOUS)
    state[seated & ~standing] = SEATED
    state[standing & ~seated] = STANDING
    state[~observable] = UNKNOWN
    state[camera_moving_mask(seq, params)] = UNKNOWN
    return PostureFeatures(
        state=state,
        hip_knee_drop=np.where(observable, drop, np.nan),
        torso_tilt_deg=np.where(observable, tilt, np.nan),
        knee_angle_deg=knee,
        hip_angle_deg=np.where(observable, hip_angle, np.nan),
    )
