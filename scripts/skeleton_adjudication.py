"""Operational hand_raised ground truth from the datasets' own Kinect skeletons.

Action labels are not operational definitions (LEARNINGS M18): PKU-MMD "drink water" or
"put on a hat" raise a hand above the shoulder without a "hand waving" label, and UTKinect
labels only one segment per action. Both datasets ship Kinect skeletons recorded by a separate
sensor (depth-based tracking, independent of MediaPipe). This script derives windows where the
PRD definition holds in 3D and scores a validate_real_video.py --dataset JSON against them.

Definition (mirrors RuleParams): a frame is "raised" if either wrist is at least
``hand_raise_min_height`` (0.5) head units above the same-side shoulder, where a head unit is the
vertical head-to-shoulder-centre distance (the video's median, because bending collapses the
per-frame value and would make low hands count as raised; seen when viewing frames);
runs of >= ``hand_raise_min_s`` (0.6 s), with gaps up
to ``merge_gap_s`` (0.4 s) bridged. Missing skeleton frames are gaps (never "raised").
Only single-subject videos are scored (the rules follow one subject track).

Skeleton files (raw downloads, gitignored, never committed):
* utkinect: data/raw/utkinect/joints/joints/joints_<seq>.txt (Kinect v1, 20 joints, frame
  numbers at 60/s; see fetch_fixed_camera_datasets.py)
* pkummd1 / pkummd2: data/raw/<phase>/skel/**/<id>-M.txt (Kinect v2, 25 joints x 2 people,
  one row per 30 fps frame)

Usage:
    python scripts/skeleton_adjudication.py results.json [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets  # noqa: E402

from behaviorlens import config  # noqa: E402
from behaviorlens.config import RuleParams  # noqa: E402

# (head, shoulder centre, left shoulder, left wrist, right shoulder, right wrist), 0-based
JOINTS = {"v1": (3, 2, 4, 6, 8, 10), "v2": (3, 20, 4, 6, 8, 10)}


def raised_windows(t: np.ndarray, joints: np.ndarray, kind: str, params: RuleParams) -> list[tuple[float, float]]:
    """Windows where the hand_raised definition holds. ``joints``: (frames, n_joints, 3), y up,
    rows of NaN for frames without a skeleton."""
    head, centre, ls, lw, rs, rw = JOINTS[kind]
    y = joints[:, :, 1]
    per_frame = y[:, head] - y[:, centre]
    positive = per_frame[np.isfinite(per_frame) & (per_frame > 0)]
    if not len(positive):
        return []
    unit = float(np.median(positive))
    with np.errstate(invalid="ignore"):
        up = (((y[:, lw] - y[:, ls]) >= params.hand_raise_min_height * unit)
                           | ((y[:, rw] - y[:, rs]) >= params.hand_raise_min_height * unit))
    runs: list[tuple[float, float]] = []
    start = last = None
    for ti, u in zip(t, up):
        if not u:
            continue
        if start is None or last is None or ti - last > params.merge_gap_s:
            if start is not None and last is not None:
                runs.append((start, last))
            start = ti
        last = ti
    if start is not None and last is not None:
        runs.append((start, last))
    return [(float(a), float(b)) for a, b in runs if b - a >= params.hand_raise_min_s]


def load_utkinect(seq: str, raw: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(raw / "utkinect" / "joints" / "joints" / f"joints_{seq}.txt")
    frames, idx = np.unique(rows[:, 0], return_index=True)  # ~2% duplicate rows: keep the first
    return (frames - frames[0]) / 60.0, rows[idx, 1:].reshape(-1, 20, 3)


def load_pku(name: str, vid: str, raw: Path) -> tuple[np.ndarray, np.ndarray]:
    path = next((raw / name / "skel").rglob(f"{vid}-M.txt"))
    rows = np.loadtxt(path, ndmin=2)[:, :75].reshape(-1, 25, 3)  # first body
    rows[np.all(rows == 0, axis=(1, 2))] = np.nan
    return np.arange(len(rows)) / 30.0, rows


def windows_for(name: str, vid: str, raw: Path, params: RuleParams) -> list[tuple[float, float]] | None:
    try:
        if name == "utkinect":
            t, j = load_utkinect(vid, raw)
            return raised_windows(t, j, "v1", params)
        if name.startswith("pkummd"):
            t, j = load_pku(name, vid, raw)
            return raised_windows(t, j, "v2", params)
    except (OSError, StopIteration):
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="validate_real_video.py --dataset ... --json output")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text())
    params = RuleParams.from_dict(res["params"])
    raw = config.data_dir() / "raw"
    out: dict[str, dict] = {}
    pooled = []
    for name, d in res["datasets"].items():
        if name == "pooled":
            continue
        rows = []
        for c in d["per_clip"]:
            if c.get("subjects") != 1:
                continue
            w = windows_for(name, c["clip"], raw, params)
            if w is None:
                continue
            rec = {"events": [{"type": "hand_raised", "start_s": s, "end_s": e, "source_label": "skeleton"}
                              for s, e in w]}
            rows.append(datasets.match_video(rec, [tuple(x) for x in c["hand_detections"]], "hand_raised", a.tol))
        if rows:
            out[name] = datasets.score_events(rows)
            pooled += rows
    if len(out) > 1:
        out["pooled"] = datasets.score_events(pooled)
    for s in out.values():
        s.pop("fp_overlapping_labels", None)
        s.pop("confused_with_sitting_down", None)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
