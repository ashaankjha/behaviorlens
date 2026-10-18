"""Central configuration: paths, limits and rule parameters.

All values here are engineering defaults for a research/demo prototype. They are
not clinically validated thresholds.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALGORITHM_VERSION = "rules-v0.1"

# MediaPipe Pose Landmarker model (downloaded by scripts/download_model.py).
MODEL_NAME = "pose_landmarker_lite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
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
# Tolerance when validating event end times against the video duration.
DURATION_TOLERANCE_S = 0.5

DEFAULT_SAMPLE_FPS = 5.0
SAMPLE_FPS_CHOICES = (5.0, 10.0)

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
    # standing_transition: (knee_y - hip_y) / torso_length thresholds.
    seated_max_drop: float = 0.35
    standing_min_drop: float = 0.6
    # Torso must be within this many degrees of vertical to count as upright.
    max_torso_tilt_deg: float = 35.0
    # If ankles are visible, the knee must be at least this straight when standing.
    standing_min_knee_angle_deg: float = 140.0
    # Seated and standing states must each be held this long around a transition.
    dwell_s: float = 1.0
    # Maximum time between the last seated frame and the first standing frame.
    max_transition_s: float = 3.0
    sustained_standing_min_s: float = 2.0
    enable_sustained_standing: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RuleParams":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class AnalysisSettings:
    sample_fps: float = DEFAULT_SAMPLE_FPS
    rules: RuleParams = field(default_factory=RuleParams)

    def to_dict(self) -> dict:
        return {"sample_fps": self.sample_fps, "rules": self.rules.to_dict()}
