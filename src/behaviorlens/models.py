"""Typed data structures shared by the pipeline, storage and UI."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import config

# MediaPipe pose landmark indices used by the rules.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
NUM_LANDMARKS = 33
# Per-landmark channels stored in PoseSequence.landmarks.
X, Y, Z, VISIBILITY, PRESENCE = range(5)
# Per-frame channels stored in PoseSequence.camera_motion (see camera_motion.py): motion of
# the image centre from the previous sampled frame as a fraction of width/height, rotation,
# log scale change, RANSAC inlier ratio and the estimation method (0 none, 1 features, 2 phase).
MOTION_DX, MOTION_DY, MOTION_ROT_DEG, MOTION_LOG_SCALE, MOTION_INLIERS, MOTION_METHOD = range(6)
MOTION_CHANNELS = 6

DataMode = Literal["synthetic", "video_inference"]
Source = Literal["algorithm", "manual"]
ReviewStatus = Literal["pending", "confirmed", "rejected", "edited"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class PoseSequence:
    """Time-indexed pose landmarks for one primary person.

    landmarks has shape (N, 33, 5) holding normalized x, y, z, visibility and
    presence. Frames without a detected person are all-NaN and flagged in
    ``has_pose``. Image coordinates: x grows rightwards, y grows downwards.

    camera_motion, if present, has shape (N, MOTION_CHANNELS). It is None for synthetic
    sequences and for files saved before camera-motion estimation existed; rules then
    treat camera motion as unknown and mask nothing.
    """

    timestamps_s: np.ndarray
    frame_indices: np.ndarray
    landmarks: np.ndarray
    sample_fps: float
    duration_s: float
    num_people: np.ndarray | None = None
    data_mode: DataMode = "video_inference"
    camera_motion: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=float)
        self.frame_indices = np.asarray(self.frame_indices, dtype=int)
        self.landmarks = np.asarray(self.landmarks, dtype=float)
        n = len(self.timestamps_s)
        if self.landmarks.shape != (n, NUM_LANDMARKS, 5):
            raise ValueError(f"landmarks must have shape ({n}, 33, 5), got {self.landmarks.shape}")
        if n > 1 and np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not (math.isfinite(self.sample_fps) and self.sample_fps > 0):
            raise ValueError(f"sample_fps must be a positive number, got {self.sample_fps}")
        if self.data_mode not in get_args(DataMode):
            raise ValueError(f"unknown data_mode {self.data_mode!r}")
        if self.num_people is None:
            self.num_people = self.has_pose.astype(int)
        if self.camera_motion is not None:
            self.camera_motion = np.asarray(self.camera_motion, dtype=float)
            if self.camera_motion.shape != (n, MOTION_CHANNELS):
                raise ValueError(f"camera_motion must have shape ({n}, {MOTION_CHANNELS}), "
                                 f"got {self.camera_motion.shape}")

    def __len__(self) -> int:
        return len(self.timestamps_s)

    @property
    def has_pose(self) -> np.ndarray:
        has: np.ndarray = ~np.all(np.isnan(self.landmarks[:, :, X]), axis=1)
        return has

    @property
    def frame_period_s(self) -> float:
        return 1.0 / self.sample_fps

    def save_npz(self, path: str | Path) -> None:
        extra: dict[str, Any] = ({} if self.camera_motion is None
                                 else {"camera_motion": self.camera_motion.astype(np.float32)})
        np.savez_compressed(
            path,
            timestamps_s=self.timestamps_s,
            frame_indices=self.frame_indices,
            landmarks=self.landmarks.astype(np.float32),
            num_people=np.asarray(self.num_people),
            meta=np.array([self.sample_fps, self.duration_s]),
            data_mode=np.array(self.data_mode),
            **extra,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> PoseSequence:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                timestamps_s=d["timestamps_s"],
                frame_indices=d["frame_indices"],
                landmarks=d["landmarks"].astype(float),
                sample_fps=float(d["meta"][0]),
                duration_s=float(d["meta"][1]),
                num_people=d["num_people"],
                data_mode=cast(DataMode, str(d["data_mode"])),
                # Optional: files written before camera-motion estimation have no such key.
                camera_motion=d["camera_motion"].astype(float) if "camera_motion" in d.files else None,
            )


class VideoMeta(BaseModel):
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    codec: str = ""


class Event(BaseModel):
    """One candidate or manually annotated observation (see PRD section 6)."""

    model_config = ConfigDict(validate_assignment=True)

    event_id: str = Field(default_factory=new_id)
    session_id: str
    analysis_id: str | None = None
    event_type: str
    start_s: float
    end_s: float
    source: Source = "algorithm"
    review_status: ReviewStatus = "pending"
    evidence: dict[str, Any] = Field(default_factory=dict)
    algorithm_version: str | None = config.ALGORITHM_VERSION
    # Heuristic rules produce no calibrated probability, so this stays null.
    confidence: float | None = None
    review_note: str = ""
    original_proposal: dict[str, Any] | None = None
    data_mode: DataMode = "video_inference"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_interval(self) -> "Event":
        if self.start_s < 0:
            raise ValueError("start_s must be >= 0")
        if not self.start_s < self.end_s:
            raise ValueError("start_s must be strictly before end_s")
        return self

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 3)

    def check_within(self, video_duration_s: float) -> None:
        limit = video_duration_s + config.DURATION_TOLERANCE_S
        if self.end_s > limit:
            raise ValueError(
                f"end_s ({self.end_s:.2f}s) is beyond the video duration ({video_duration_s:.2f}s)"
            )


@dataclass
class AnalysisResult:
    events: list[Event]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
