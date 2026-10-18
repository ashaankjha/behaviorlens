"""Clearly-labelled SYNTHETIC demo data.

Generates a programmatic pose sequence (no real person, no inference) that
exercises the event rules and review UI, the matching ground truth, and a
stick-figure MP4 for timeline playback. A rendered stick figure does not yield
reliable MediaPipe landmarks, so the demo session bypasses pose inference and
must never be presented as validating real pose extraction.

Scripted timeline (40 s):
  0-5 s     seated, hands down
  5-6.2 s   stands up                      -> standing_transition
  8-10 s    right hand raised              -> hand_raised (right)
  12.6 s    0.25 s left-hand flick         (below 0.6 s minimum: no event)
  13.5-14.7 sits down                      (stand->sit: not a standing_transition)
  16.5-18.5 both hands raised while seated -> hand_raised (both)
  20-21.2 s stands up                      -> standing_transition
  23-25 s   legs hidden behind a desk      (posture unobservable)
  26-28 s   right hand raised, 1 dropped frame of wrist visibility -> one hand_raised
  30-32 s   person out of frame            (no_pose)
  32-37.2 s seated behind desk, stands up while legs hidden
            (a real transition the rules cannot observe: expected miss)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import config
from .models import NUM_LANDMARKS, PoseSequence

DEMO_DURATION_S = 40.0
DEMO_VIDEO = "demo_synthetic.mp4"
DEMO_POSE = "demo_synthetic_pose.npz"
DEMO_TRUTH = "demo_ground_truth.json"
SEED = 7

_STAND_KP = [(0, 0), (5, 0), (6.2, 1), (13.5, 1), (14.7, 0), (20, 0), (21.2, 1), (30, 1),
             (32, 0), (36, 0), (37.2, 1), (40, 1)]
_RIGHT_UP_KP = [(0, 0), (7.85, 0), (8.15, 1), (9.85, 1), (10.15, 0), (16.35, 0), (16.65, 1),
                (18.35, 1), (18.65, 0), (25.85, 0), (26.15, 1), (27.85, 1), (28.15, 0), (40, 0)]
_LEFT_UP_KP = [(0, 0), (12.55, 0), (12.65, 1), (12.75, 1), (12.85, 0), (16.35, 0), (16.65, 1),
               (18.35, 1), (18.65, 0), (40, 0)]
_LEGS_HIDDEN = [(23.0, 25.0), (32.0, 37.2)]
_ABSENT = [(30.0, 32.0)]
_WRIST_DROPOUT = [(26.9, 27.1)]

GROUND_TRUTH = [
    {"event_type": "standing_transition", "start_s": 5.0, "end_s": 6.2},
    {"event_type": "hand_raised", "start_s": 8.0, "end_s": 10.0, "side": "right"},
    {"event_type": "hand_raised", "start_s": 16.5, "end_s": 18.5, "side": "both"},
    {"event_type": "standing_transition", "start_s": 20.0, "end_s": 21.2},
    {"event_type": "hand_raised", "start_s": 26.0, "end_s": 28.0, "side": "right"},
    {"event_type": "standing_transition", "start_s": 36.0, "end_s": 37.2,
     "note": "legs hidden: not observable by the rules (expected miss)"},
    {"event_type": "sustained_standing", "start_s": 6.2, "end_s": 13.5},
    {"event_type": "sustained_standing", "start_s": 25.0, "end_s": 30.0},
    {"event_type": "sustained_standing", "start_s": 37.2, "end_s": 40.0,
     "note": "legs visible again after 37.2 s"},
]


def _interp(kp: list[tuple[float, float]], t: float) -> float:
    xs, ys = zip(*kp)
    return float(np.interp(t, xs, ys))


def _in(windows: list[tuple[float, float]], t: float) -> bool:
    return any(a <= t < b for a, b in windows)


def _keypoints(stand: float, left_up: float, right_up: float) -> dict[int, tuple[float, float]]:
    """Frontal-view skeleton in normalized image coordinates (y down).

    MediaPipe's "left" is the person's left, which appears on the image right.
    """
    lerp = lambda a, b: a + (b - a) * stand  # noqa: E731
    sh_y = lerp(0.38, 0.28)
    hip_y = lerp(0.62, 0.52)
    pts: dict[int, tuple[float, float]] = {0: (0.5, sh_y - 0.10)}
    for sign, sh, el, wr, hip, knee, ank, up in (
        (+1, 11, 13, 15, 23, 25, 27, left_up),
        (-1, 12, 14, 16, 24, 26, 28, right_up),
    ):
        sx = 0.5 + sign * 0.06
        pts[sh] = (sx, sh_y)
        down_el, up_el = (sx + sign * 0.02, sh_y + 0.11), (sx + sign * 0.03, sh_y - 0.09)
        down_wr, up_wr = (sx + sign * 0.03, sh_y + 0.21), (sx + sign * 0.04, sh_y - 0.17)
        pts[el] = tuple(np.add(down_el, np.multiply(up, np.subtract(up_el, down_el))))
        pts[wr] = tuple(np.add(down_wr, np.multiply(up, np.subtract(up_wr, down_wr))))
        pts[hip] = (0.5 + sign * 0.04, hip_y)
        pts[knee] = (0.5 + sign * lerp(0.06, 0.045), lerp(hip_y + 0.04, hip_y + 0.18))
        pts[ank] = (0.5 + sign * lerp(0.06, 0.05), lerp(0.86, 0.88))
    nx, ny = pts[0]
    face = {1: (0.01, -0.02), 2: (0.015, -0.02), 3: (0.02, -0.02), 4: (-0.01, -0.02),
            5: (-0.015, -0.02), 6: (-0.02, -0.02), 7: (0.04, -0.01), 8: (-0.04, -0.01),
            9: (0.01, 0.03), 10: (-0.01, 0.03)}
    for i, (dx, dy) in face.items():
        pts[i] = (nx + dx, ny + dy)
    for wr, ids, sign in ((15, (17, 19, 21), +1), (16, (18, 20, 22), -1)):
        wx, wy = pts[wr]
        for k, i in enumerate(ids):
            pts[i] = (wx + sign * 0.01 * (k - 1), wy + 0.02)
    for ank, heel, toe in ((27, 29, 31), (28, 30, 32)):
        ax, ay = pts[ank]
        pts[heel], pts[toe] = (ax, ay + 0.015), (ax, ay + 0.03)
    return pts


def pose_at(t: float) -> tuple[dict[int, tuple[float, float]], set[int], bool]:
    """Keypoints, hidden landmark ids and presence flag at time t."""
    pts = _keypoints(_interp(_STAND_KP, t), _interp(_LEFT_UP_KP, t), _interp(_RIGHT_UP_KP, t))
    hidden: set[int] = set()
    if _in(_LEGS_HIDDEN, t):
        hidden |= {23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
    if _in(_WRIST_DROPOUT, t):
        hidden |= {16, 18, 20, 22}
    return pts, hidden, not _in(_ABSENT, t)


def generate_pose_sequence(sample_fps: float = config.DEFAULT_SAMPLE_FPS, seed: int = SEED,
                           noise: float = 0.004) -> PoseSequence:
    rng = np.random.default_rng(seed)
    times = np.round(np.arange(0, DEMO_DURATION_S, 1.0 / sample_fps), 4)
    lms = np.full((len(times), NUM_LANDMARKS, 5), np.nan)
    people = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        pts, hidden, present = pose_at(t)
        if not present:
            continue
        people[i] = 1
        for j in range(NUM_LANDMARKS):
            x, y = pts[j]
            vis = 0.08 if j in hidden else float(np.clip(rng.normal(0.95, 0.02), 0.85, 1.0))
            lms[i, j] = (x + rng.normal(0, noise), y + rng.normal(0, noise), 0.0, vis, 0.99)
    return PoseSequence(timestamps_s=times, frame_indices=np.arange(len(times)), landmarks=lms,
                        sample_fps=sample_fps, duration_s=DEMO_DURATION_S, num_people=people,
                        data_mode="synthetic")


_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
          (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def render_frame(t: float, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), (46, 38, 34), dtype=np.uint8)
    cv2.putText(frame, "SYNTHETIC DEMO - not real footage", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (90, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"t = {t:5.1f}s", (width - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 220), 1, cv2.LINE_AA)
    pts, hidden, present = pose_at(t)
    if present:
        px = {i: (int(x * width), int(y * height)) for i, (x, y) in pts.items()}
        if _interp(_STAND_KP, t) < 0.5 and not _in(_LEGS_HIDDEN, t):
            seat_y = int((pts[23][1] + 0.03) * height)
            cv2.rectangle(frame, (int(0.36 * width), seat_y), (int(0.64 * width), seat_y + 10),
                          (60, 110, 150), -1)
            for x in (0.38, 0.62):
                cv2.line(frame, (int(x * width), seat_y), (int(x * width), int(0.9 * height)),
                         (60, 110, 150), 6)
        for a, b in _BONES:
            cv2.line(frame, px[a], px[b], (235, 235, 235), 4, cv2.LINE_AA)
        cv2.circle(frame, px[0], 22, (235, 235, 235), 3, cv2.LINE_AA)
        for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            cv2.circle(frame, px[i], 5, (120, 200, 120), -1, cv2.LINE_AA)
        if _in(_LEGS_HIDDEN, t):
            top = int((min(pts[23][1], pts[24][1]) - 0.02) * height)
            cv2.rectangle(frame, (int(0.25 * width), top), (int(0.75 * width), height), (70, 90, 120), -1)
            cv2.putText(frame, "desk (legs hidden)", (int(0.33 * width), top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "(person out of frame)", (int(0.32 * width), height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def render_video(path: Path, fps: float = 15.0) -> Path:
    from .video_io import FrameWriter

    with FrameWriter(path, 640, 480, fps) as writer:
        for t in np.arange(0, DEMO_DURATION_S, 1.0 / fps):
            writer.write(render_frame(float(t)))
    return path


def ensure_demo_assets(directory: Path | None = None, render: bool = True) -> dict[str, Path]:
    """Create the synthetic fixture, ground truth and (optionally) video if missing."""
    directory = Path(directory or config.sample_data_dir())
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"pose": directory / DEMO_POSE, "truth": directory / DEMO_TRUTH, "video": directory / DEMO_VIDEO}
    if not paths["pose"].is_file():
        generate_pose_sequence().save_npz(paths["pose"])
    if not paths["truth"].is_file():
        paths["truth"].write_text(json.dumps({
            "description": "Ground truth for the SYNTHETIC demo fixture (scripted, not human-annotated footage).",
            "events": GROUND_TRUTH,
        }, indent=2))
    if render and not paths["video"].is_file():
        render_video(paths["video"])
    return paths
