"""MediaPipe Tasks Pose Landmarker wrapper (VIDEO running mode)."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Protocol

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


class MultiPoseEstimator(PoseEstimator, Protocol):
    def detect_all(self, frame_bgr: np.ndarray, timestamp_s: float) -> list[np.ndarray]:
        """Return every detected person's (33, 5) landmarks (possibly empty)."""


class ModelIntegrityError(RuntimeError):
    pass


MAX_MODEL_BYTES = 64 * 1024 * 1024


def download_model(dest: Path | None = None, url: str = config.MODEL_URL,
                   expected_sha256: str | None = config.MODEL_SHA256) -> Path:
    """Download the pose model (network access to Google's model bucket only).

    The file is written to a temporary name and only moved into place once its
    SHA-256 matches ``expected_sha256`` (pass None to skip, e.g. for other model sizes).
    """
    if not url.startswith("https://"):
        raise ValueError("model URL must use https")
    dest = dest or config.model_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
            total = 0
            while block := resp.read(1 << 16):
                total += len(block)
                if total > MAX_MODEL_BYTES:
                    raise ModelIntegrityError("model download is larger than expected; aborted")
                h.update(block)
                f.write(block)
        if expected_sha256 and h.hexdigest() != expected_sha256:
            raise ModelIntegrityError(
                f"downloaded model checksum {h.hexdigest()[:12]}… does not match the pinned "
                f"{expected_sha256[:12]}…; the file was discarded. If Google published a new model, "
                "it must be re-validated (docs/VALIDATION.md) before MODEL_SHA256 is updated.")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def verify_model(path: Path | None = None, expected_sha256: str | None = config.MODEL_SHA256) -> bool:
    """True when the model file exists and matches the pinned checksum."""
    path = path or config.model_path()
    if not path.is_file():
        return False
    return expected_sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def model_version(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}@{digest}"


class MediaPipePoseEstimator:
    """Pose estimation for up to ``max_people`` people per frame.

    ``detect`` returns the largest person per frame (legacy single-person path);
    ``detect_all`` returns every detected person, for tracking (tracking.py).
    """

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
        if path.resolve() == config.default_model_path().resolve() and not verify_model(path):
            raise ModelIntegrityError(
                "The pose model file does not match the pinned checksum (it may be corrupted or replaced). "
                "Run `python scripts/download_model.py --force`.")
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
        arrays = self.detect_all(frame_bgr, timestamp_s)
        if not arrays:
            return None, 0
        return select_primary(arrays), len(arrays)

    def detect_all(self, frame_bgr: np.ndarray, timestamp_s: float) -> list[np.ndarray]:
        # VIDEO mode requires strictly increasing integer millisecond timestamps.
        ts_ms = max(int(round(timestamp_s * 1000)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        return [_to_array(p) for p in result.pose_landmarks or []]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _to_array(landmarks: Any) -> np.ndarray:
    out = np.full((NUM_LANDMARKS, 5), np.nan)
    for i, lm in enumerate(landmarks[:NUM_LANDMARKS]):
        out[i] = (lm.x, lm.y, lm.z,
                  lm.visibility if lm.visibility is not None else np.nan,
                  lm.presence if lm.presence is not None else np.nan)
    return out


def select_primary(arrays: list[np.ndarray]) -> np.ndarray:
    """The person whose visible landmarks span the largest bounding box (usually the nearest)."""
    return max(arrays, key=_bbox_area)


def _bbox_area(arr: np.ndarray) -> float:
    vis = np.nan_to_num(arr[:, VISIBILITY], nan=0.0) > 0.5
    if not vis.any():
        return 0.0
    xs, ys = arr[vis, X], arr[vis, Y]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

