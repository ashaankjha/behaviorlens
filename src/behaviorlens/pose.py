"""MediaPipe Tasks Pose Landmarker wrapper (VIDEO running mode)."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from . import config
from .models import NUM_LANDMARKS, VISIBILITY, X, Y

log = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    pass


class PoseInferenceError(RuntimeError):
    pass


class PoseEstimator(Protocol):
    model_version: str

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> tuple[np.ndarray | None, int]:
        """Return ((33, 5) landmarks or None, number of people detected)."""

    def close(self) -> None: ...


def download_model(dest: Path | None = None, url: str = config.MODEL_URL) -> Path:
    """Download the pose model (network access to Google's model bucket only)."""
    dest = dest or config.model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while block := resp.read(1 << 16):
            f.write(block)
    os.replace(tmp, dest)
    return dest


def model_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}@{digest}"


class MediaPipePoseEstimator:
    """Primary-person pose estimation. Detects up to two people so that a second
    person can be flagged (attribution is not reliable in that case)."""

    def __init__(self, model_file: Path | None = None, max_people: int = 2,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        path = Path(model_file or config.model_path())
        if not path.is_file():
            raise ModelNotFoundError(
                f"Pose model not found at {path.name}. Run `python scripts/download_model.py` "
                "(downloads ~6 MB from Google's MediaPipe model bucket)."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except Exception as exc:  # pragma: no cover - depends on environment
            raise PoseInferenceError(
                f"MediaPipe could not be imported ({exc}). On Linux, install libegl1 and libgles2."
            ) from exc
        self._mp = mp
        self.model_version = model_version(path)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise PoseInferenceError(f"Could not load the pose model: {exc}") from exc
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> tuple[np.ndarray | None, int]:
        # VIDEO mode requires strictly increasing integer millisecond timestamps.
        ts_ms = max(int(round(timestamp_s * 1000)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        poses = result.pose_landmarks or []
        if not poses:
            return None, 0
        arrays = [_to_array(p) for p in poses]
        primary = max(arrays, key=_bbox_area)
        return primary, len(arrays)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _to_array(landmarks) -> np.ndarray:
    out = np.full((NUM_LANDMARKS, 5), np.nan)
    for i, lm in enumerate(landmarks[:NUM_LANDMARKS]):
        out[i] = (lm.x, lm.y, lm.z,
                  lm.visibility if lm.visibility is not None else np.nan,
                  lm.presence if lm.presence is not None else np.nan)
    return out


def _bbox_area(arr: np.ndarray) -> float:
    vis = np.nan_to_num(arr[:, VISIBILITY], nan=0.0) > 0.5
    if not vis.any():
        return 0.0
    xs, ys = arr[vis, X], arr[vis, Y]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

