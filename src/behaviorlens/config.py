"""Central configuration: paths, limits and rule parameters.

All values here are engineering defaults for a research/demo prototype. They are
not clinically validated thresholds.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALGORITHM_VERSION = "rules-v0.2"

# MediaPipe Pose Landmarker model (downloaded by scripts/download_model.py).
MODEL_NAME = "pose_landmarker_lite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
# Versioned URL ("/1/", byte-identical to "/latest/" on 2026-10-20) plus a pinned checksum, so an
# upstream model update can never silently change inference: it fails the download instead.
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
MODEL_CARD_URL = (
    "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models"
)

# Upload / video limits (FR1).
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DURATION_S = 5 * 60
MIN_DURATION_S = 1.0
MIN_FPS = 1.0
MAX_FPS = 240.0
# Decoded frame size limit (4K UHD). Bounds per-frame memory (a 3840x2160 BGR frame is ~25 MB)
# and rejects crafted headers that claim enormous dimensions.
MAX_FRAME_PIXELS = 3840 * 2160
# Hard cap on frames decoded per analysis (5 min at the 240 fps maximum), independent of
# whatever frame count the container header claims.
MAX_DECODED_FRAMES = int(MAX_DURATION_S * MAX_FPS)
# Ground-truth CSV upload limits.
MAX_GROUND_TRUTH_ROWS = 10_000
# Tolerance when validating event end times against the video duration.
DURATION_TOLERANCE_S = 0.5

DEFAULT_SAMPLE_FPS = 5.0
SAMPLE_FPS_CHOICES = (5.0, 10.0)
# Pose Landmarker num_poses for analysis. People are then tracked across frames (tracking.py).
DEFAULT_MAX_PEOPLE = 4

EVENT_TYPES = ("hand_raised", "standing_transition", "sustained_standing")
MANUAL_EVENT_TYPES = EVENT_TYPES + ("other_observable",)
REVIEW_STATUSES = ("pending", "confirmed", "rejected", "edited")

DISCLAIMER = (
    "Research/demo prototype. Not for clinical decision-making. "
    "Use only synthetic or staged footage with informed consent."
)
EXPORT_WARNING = (
    "Algorithm-generated suggestions describe visually observable movement "
    "patterns only. They have not been clinically validated and do not "
    "represent diagnoses, intent, emotion, behavioral function or treatment "
    "recommendations. Human review is required."
)


def data_dir() -> Path:
    """Local, gitignored storage root. Override with BEHAVIORLENS_DATA_DIR."""
    return Path(os.environ.get("BEHAVIORLENS_DATA_DIR", PROJECT_ROOT / "data"))


def model_path() -> Path:
    override = os.environ.get("BEHAVIORLENS_MODEL_PATH")
    if override:
        return Path(override)
    return default_model_path()


def default_model_path() -> Path:
    """Where download_model puts the pinned model (config.MODEL_SHA256 applies to this file)."""
    return PROJECT_ROOT / "assets" / "models" / f"{MODEL_NAME}.task"


def sample_data_dir() -> Path:
    return PROJECT_ROOT / "sample_data"


@dataclass(frozen=True)
class RuleParams:
    """Thresholds for the interpretable event rules (FR3)."""

    # Minimum MediaPipe visibility for a landmark to count as observed.
    visibility_threshold: float = 0.5
    # hand_raised: wrist must be this far above the same-side shoulder, measured in
    # "head units" (vertical nose-to-shoulder-line distance), so it is scale-invariant.
    hand_raise_min_height: float = 0.5
    hand_raise_min_s: float = 0.6
    # Gaps (missing/negative frames) up to this long are bridged inside a segment.
    merge_gap_s: float = 0.4
    # Posture cues (rules-v0.2; thresholds chosen on the Charades tuning split, see
    # docs/VALIDATION.md). Drops are per-leg (knee_y - hip_y) / torso_length; angles in degrees.
    seated_max_drop: float = 0.35  # frontal-view cue: thighs foreshortened/horizontal
    seated_max_hip_angle_deg: float = 140.0  # shoulder-hip-knee: hips flexed
    seated_max_knee_angle_deg: float = 130.0  # hip-knee-ankle, when ankles are visible
    seated_drop_ceiling: float = 0.5  # no leg may hang straight down when seated
    standing_min_drop: float = 0.3
    standing_min_hip_angle_deg: float = 150.0
    # Torso must be within this many degrees of vertical to count as upright.
    max_torso_tilt_deg: float = 35.0
    # If ankles are visible, the knee must be at least this straight when standing.
    standing_min_knee_angle_deg: float = 140.0
    # Gap bridging for posture runs (separate from merge_gap_s so wider bridging here cannot
    # merge brief hand flicks into a sustained raise).
    posture_merge_gap_s: float = 0.4
    # Seated and standing states must each be held this long around a transition.
    # (rules-v0.2: 0.8 s / 5 s chosen on the Charades tuning split; v0.1 used 1.0 s / 3 s.)
    dwell_s: float = 0.8
    # Maximum time between the last seated frame and the first standing frame.
    max_transition_s: float = 5.0
    sustained_standing_min_s: float = 2.0
    enable_sustained_standing: bool = True
    # Camera motion (camera_motion.py). Frames where the camera moves faster than any of
    # these rates are unobservable for posture, since image-coordinate landmarks then mix
    # camera and body movement. Translation is a fraction of the frame per second; scale is
    # |log scale change| per second. Chosen on the Charades tuning split as the most sensitive
    # setting that cost no tuning-split stand-up and added no false positive (0.15 /s, or a
    # one-frame pad at 0.2 /s, or 5 deg/s already lost one); see docs/VALIDATION.md, "Camera motion".
    # Hand rules are not masked: the cue is within-frame geometry, and masking changed no
    # tuning-split clip outcome (it only split one event).
    mask_camera_motion: bool = True
    camera_max_translation_per_s: float = 0.2
    camera_max_rotation_deg_per_s: float = 10.0
    camera_max_scale_change_per_s: float = 0.1
    # Feature-based estimates with a lower RANSAC inlier ratio are unknown (never masking).
    camera_min_inlier_ratio: float = 0.5
    # Moving frames are widened by this many sampled frames on each side.
    camera_motion_pad_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuleParams:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class TrackingParams:
    """Frame-to-frame person tracking (tracking.py). Identity comes from position and size
    continuity only; there is no appearance or face model."""

    # A lost track can be re-identified for this long; after that a new track ID starts.
    # (1, 2 and 4 s tied on the tune split; the a-priori 2 s was kept.)
    max_gap_s: float = 2.0
    # Association gate, in units of the track's size (sqrt of its landmark bounding-box area).
    max_center_dist: float = 1.0
    # The gate widens by this much per second the track has been unseen.
    gate_growth_per_s: float = 1.0
    # A pair whose (predicted) boxes overlap at least this much is always a candidate match.
    min_iou: float = 0.1
    # Boxes whose areas differ by more than this factor are never the same person. Selected on the
    # Charades/UCF101 tune split from {3, 6, 100}: a real person's landmark box often jumps >3x between
    # frames (hallucinated legs, arm raises), so size works as a soft cost (size_weight), not a gate.
    max_size_ratio: float = 100.0
    # Cost = (1 - IoU) + dist_weight * centre distance + size_weight * |log area ratio|
    #        + stale_weight * seconds since the track was last seen (recent tracks win ties).
    dist_weight: float = 0.5
    size_weight: float = 0.5
    stale_weight: float = 1.0
    # Detections in one frame overlapping more than this are duplicates of one person
    # (0.7 beat 0.5 on tune-split precision).
    duplicate_iou: float = 0.7
    # Tracks seen for at least this long count as a person in the scene (warning, subject list).
    substantial_min_s: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrackingParams":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class AnalysisSettings:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    rules: RuleParams = field(default_factory=RuleParams)
    max_people: int = DEFAULT_MAX_PEOPLE
    tracking: TrackingParams = field(default_factory=TrackingParams)
    # None = automatic subject (greatest coverage x mean size); otherwise a reviewer-chosen track ID.
    subject_track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"sample_fps": self.sample_fps, "rules": self.rules.to_dict(), "max_people": self.max_people,
                "tracking": self.tracking.to_dict(), "subject_track_id": self.subject_track_id}
