from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from behaviorlens import config
from behaviorlens.demo_generator import _keypoints
from behaviorlens.models import NUM_LANDMARKS, PoseSequence
from behaviorlens.storage import Store

LEG_IDS = set(range(23, 33))


def make_seq(duration: float = 10.0, fps: float = 5.0, *,
             stand: Callable[[float], float] = lambda t: 1.0,
             left_up: Callable[[float], float] = lambda t: 0.0,
             right_up: Callable[[float], float] = lambda t: 0.0,
             hidden: Callable[[float], set[int]] = lambda t: set(),
             absent: Callable[[float], bool] = lambda t: False,
             noise: float = 0.0, seed: int = 0, people: int = 1) -> PoseSequence:
    """Synthetic landmark sequence built from the demo skeleton."""
    rng = np.random.default_rng(seed)
    times = np.round(np.arange(0, duration, 1 / fps), 4)
    lms = np.full((len(times), NUM_LANDMARKS, 5), np.nan)
    n = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        if absent(t):
            continue
        n[i] = people
        pts = _keypoints(stand(t), left_up(t), right_up(t))
        hid = hidden(t)
        for j in range(NUM_LANDMARKS):
            x, y = pts[j]
            lms[i, j] = (x + rng.normal(0, noise), y + rng.normal(0, noise), 0, 0.05 if j in hid else 0.95, 0.99)
    return PoseSequence(timestamps_s=times, frame_indices=np.arange(len(times)), landmarks=lms,
                        sample_fps=fps, duration_s=duration, num_people=n, data_mode="synthetic")


def window(a: float, b: float, value: float = 1.0) -> Callable[[float], float]:
    return lambda t: value if a <= t < b else 0.0


@pytest.fixture
def store(tmp_path, monkeypatch) -> Store:
    monkeypatch.setenv("BEHAVIORLENS_DATA_DIR", str(tmp_path / "data"))
    return Store(tmp_path / "data")


def write_video(path: Path, seconds: float, fps: float = 15.0, size=(320, 240),
                fourcc: str = "mp4v") -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    for k in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.circle(frame, (20 + k % (size[0] - 40), size[1] // 2), 10, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


requires_model = pytest.mark.skipif(not config.model_path().is_file(),
                                    reason="pose model not downloaded (python scripts/download_model.py)")
