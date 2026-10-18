"""Validate the event rules on real moving-person video with real MediaPipe inference.

Run scripts/fetch_validation_clips.py first. Pose landmarks are cached per clip in
data/validation/cache/, so re-running with different rule parameters is fast.

Usage:
    python scripts/validate_real_video.py [--params '{"dwell_s": 0.8}'] [--split all|tune|holdout]
                                          [--report docs/VALIDATION.md] [--json out.json]

Metrics:
* Charades standing_transition (event level). A detection is correct if it overlaps a
  label-154 "standing up" window widened by --tol seconds. Recall counts windows with
  at least one detection. Detections inside label-151 "sitting down" windows are reported
  separately as confusions.
* UCF101 hand_raised (clip level). Arms-overhead classes should yield >=1 event; the
  other classes should yield none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402
from behaviorlens.config import RuleParams  # noqa: E402
from behaviorlens.models import PoseSequence  # noqa: E402
from behaviorlens.pipeline import analyze_pose, extract_pose  # noqa: E402
from behaviorlens.pose import MediaPipePoseEstimator, model_version  # noqa: E402
from behaviorlens.video_io import VideoValidationError, probe_video  # noqa: E402

STAND_UP, SIT_DOWN = 154, 151
FPS = 5.0


def vdir() -> Path:
    return config.data_dir() / "validation"


def split_of(clip_id: str) -> str:
    """Deterministic 50/50 split so thresholds can be tuned on one half and checked on the other."""
    return "tune" if int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def pose_for(path_str: str) -> tuple[str, str | None]:
    path = Path(path_str)
    cache = vdir() / "cache" / f"{path.parent.name}_{path.stem}_{FPS:g}fps.npz"
    if cache.is_file():
        return path_str, str(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = probe_video(path)
        with MediaPipePoseEstimator() as est:
            seq, _ = extract_pose(path, meta, est, FPS)
        seq.save_npz(cache)
        return path_str, str(cache)
    except VideoValidationError as exc:
        return path_str, f"ERROR: {exc}"


def ensure_poses(paths: list[Path], workers: int) -> dict[str, str]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(pose_for, [str(p) for p in paths]))


def overlaps(a: tuple[float, float], b: tuple[float, float], tol: float = 0.0) -> bool:
    return a[0] < b[1] + tol and b[0] - tol < a[1]


def eval_charades(params: RuleParams, poses: dict[str, str], split: str, tol: float) -> dict:
    ann = json.loads((vdir() / "charades" / "annotations.json").read_text())
    clips, windows, hits, n_det, det_ok, det_sitdown = [], 0, 0, 0, 0, 0
    for a in ann:
        if split != "all" and split_of(a["video_id"]) != split:
            continue
        cache = poses.get(str(vdir() / "charades" / "videos" / f"{a['video_id']}.mp4"))
        if not cache or cache.startswith("ERROR"):
            continue
        seq = PoseSequence.load_npz(cache)
        res = analyze_pose(seq, params, a["video_id"])
        det = [(e.start_s, e.end_s) for e in res.events if e.event_type == "standing_transition"]
        ups = [tuple(t) for lab, t in zip(a["labels"], a["action_timings"]) if lab == STAND_UP]
        downs = [tuple(t) for lab, t in zip(a["labels"], a["action_timings"]) if lab == SIT_DOWN]
        found = sum(any(overlaps(d, w, tol) for d in det) for w in ups)
        good = sum(any(overlaps(d, w, tol) for w in ups) for d in det)
        confused = sum(1 for d in det if not any(overlaps(d, w, tol) for w in ups)
                       and any(overlaps(d, w, tol) for w in downs))
        windows += len(ups)
        hits += found
        n_det += len(det)
        det_ok += good
        det_sitdown += confused
        clips.append({"clip": a["video_id"], "scene": a["scene"], "stand_up_windows": len(ups),
                      "found": found, "detections": len(det), "correct_detections": good,
                      "posture_observable": res.stats["posture_rule_observable_ratio"],
                      "pose_detected": res.stats["pose_detected_ratio"]})
    observable = [c for c in clips if c["stand_up_windows"] and c["posture_observable"] >= 0.3]
    return {
        "clips": len(clips),
        "stand_up_windows": windows,
        "recall": round(hits / windows, 3) if windows else None,
        "detections": n_det,
        "precision": round(det_ok / n_det, 3) if n_det else None,
        "detections_during_sit_down": det_sitdown,
        "recall_when_legs_observable": (
            round(sum(c["found"] for c in observable) / max(1, sum(c["stand_up_windows"] for c in observable)), 3)
            if observable else None),
        "clips_with_stand_up_and_legs_observable": len(observable),
        "median_posture_observable": sorted(c["posture_observable"] for c in clips)[len(clips) // 2] if clips else None,
        "per_clip": clips,
    }


def eval_ucf(params: RuleParams, poses: dict[str, str], split: str) -> dict:
    labels = json.loads((vdir() / "ucf101" / "labels.json").read_text())
    rows = []
    for name, lab in labels.items():
        if split != "all" and split_of(name) != split:
            continue
        cache = poses.get(str(vdir() / "ucf101" / name))
        if not cache or cache.startswith("ERROR"):
            continue
        res = analyze_pose(PoseSequence.load_npz(cache), params, name)
        n = sum(e.event_type == "hand_raised" for e in res.events)
        rows.append({"clip": name, "class": lab["class"], "arms_overhead": lab["arms_overhead"],
                     "hand_raised_events": n, "predicted": n > 0,
                     "hand_observable": res.stats["hand_rule_observable_ratio"]})
    tp = sum(r["predicted"] and r["arms_overhead"] for r in rows)
    fp = sum(r["predicted"] and not r["arms_overhead"] for r in rows)
    fn = sum(not r["predicted"] and r["arms_overhead"] for r in rows)
    tn = sum(not r["predicted"] and not r["arms_overhead"] for r in rows)
    per_class = {}
    for r in rows:
        c = per_class.setdefault(r["class"], {"clips": 0, "with_hand_raised": 0})
        c["clips"] += 1
        c["with_hand_raised"] += r["predicted"]
    return {"clips": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "accuracy": round((tp + tn) / len(rows), 3) if rows else None,
            "per_class": per_class, "per_clip": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="{}", help="JSON overrides for RuleParams")
    ap.add_argument("--split", default="all", choices=["all", "tune", "holdout"])
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--json", help="write full results JSON here")
    a = ap.parse_args()
    params = RuleParams.from_dict({**RuleParams().to_dict(), **json.loads(a.params)})
    paths = sorted((vdir() / "charades" / "videos").glob("*.mp4")) + sorted((vdir() / "ucf101").glob("*.avi"))
    if not paths:
        sys.exit("no clips: run scripts/fetch_validation_clips.py first")
    t0 = time.time()
    poses = ensure_poses(paths, a.workers)
    errors = {Path(k).name: v for k, v in poses.items() if v.startswith("ERROR")}
    result = {
        "model_version": model_version(config.model_path()),
        "algorithm_version": config.ALGORITHM_VERSION,
        "sample_fps": FPS,
        "params": params.to_dict(),
        "split": a.split,
        "tolerance_s": a.tol,
        "charades_standing_transition": eval_charades(params, poses, a.split, a.tol),
        "ucf101_hand_raised": eval_ucf(params, poses, a.split),
        "clip_errors": errors,
        "seconds": round(time.time() - t0, 1),
    }
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k not in ("charades_standing_transition", "ucf101_hand_raised")}
    summary["charades"] = {k: v for k, v in result["charades_standing_transition"].items() if k != "per_clip"}
    summary["ucf101"] = {k: v for k, v in result["ucf101_hand_raised"].items() if k != "per_clip"}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
