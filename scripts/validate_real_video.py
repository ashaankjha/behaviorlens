"""Validate the event rules on real moving-person video with real MediaPipe inference.

Run scripts/fetch_validation_clips.py first. Pose landmarks are cached per clip in
data/validation/cache/, so re-running with different rule parameters is fast.

Usage:
    python scripts/validate_real_video.py [--params '{"dwell_s": 0.8}'] [--split all|tune|holdout]
                                          [--fps 5] [--json out.json]

The pose model is config.model_path() (override with BEHAVIORLENS_MODEL_PATH). The landmark
cache is keyed by clip, model version (name + content hash), fps and camera-motion estimator
version, so switching models or sampling rates never silently reuses another model's landmarks
(LEARNINGS M21), and caches without camera-motion data are never mistaken for current ones.

Metrics:
* Charades standing_transition (event level). A detection is correct if it overlaps a
  label-154 "standing up" window widened by --tol seconds. Recall counts windows with
  at least one detection. Detections inside label-151 "sitting down" windows are reported
  separately as confusions.
* Charades per-frame posture. Frames inside a seated activity (and no standing activity or
  stand-up/sit-down window) are labelled seated, and vice versa. Reports, over frames the
  posture rule observes, the share labelled seated that the rule calls standing and the reverse.
* UCF101 hand_raised (clip level). Arms-overhead classes should yield >=1 event; the
  other classes should yield none.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402
from behaviorlens.camera_motion import CAMERA_MOTION_VERSION  # noqa: E402
from behaviorlens.config import RuleParams  # noqa: E402
from behaviorlens.models import PoseSequence  # noqa: E402
from behaviorlens.pipeline import analyze_pose, extract_pose  # noqa: E402
from behaviorlens.pose import MediaPipePoseEstimator, model_version  # noqa: E402
from behaviorlens.video_io import VideoValidationError, probe_video  # noqa: E402

STAND_UP, SIT_DOWN = 154, 151
# Charades activities that imply a posture for per-frame checks (class names from the dataset card).
SEATED_ACTIVITIES = {11, 59, 123}  # sitting at a table, in a chair, on a sofa/couch
STANDING_ACTIVITIES = {6, 8, 97, 104, 105}  # closing/opening a door, doorway, light on/off


def vdir() -> Path:
    return config.data_dir() / "validation"


def split_of(clip_id: str) -> str:
    """Deterministic 50/50 split so thresholds can be tuned on one half and checked on the other."""
    return "tune" if int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def cache_path(clip: Path, model_ver: str, fps: float) -> Path:
    """Landmark cache file for one clip. Model version, fps and the camera-motion estimator
    version are part of the key."""
    safe_model = model_ver.replace("@", "-")
    return vdir() / "cache" / f"{clip.parent.name}_{clip.stem}_{safe_model}_{fps:g}fps_{CAMERA_MOTION_VERSION}.npz"


def pose_for(path_str: str, model_file: str, model_ver: str, fps: float) -> tuple[str, str]:
    path = Path(path_str)
    cache = cache_path(path, model_ver, fps)
    if cache.is_file():
        return path_str, str(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = probe_video(path)
        with MediaPipePoseEstimator(Path(model_file)) as est:
            seq, _ = extract_pose(path, meta, est, fps)
        seq.save_npz(cache)
        return path_str, str(cache)
    except VideoValidationError as exc:
        return path_str, f"ERROR: {exc}"


def ensure_poses(paths: list[Path], workers: int, model_file: Path, fps: float) -> tuple[dict[str, str], dict]:
    """Compute missing landmark caches. Returns the cache map and the extraction cost."""
    model_ver = model_version(model_file)
    todo = sum(not cache_path(p, model_ver, fps).is_file() for p in paths)
    t0, cpu0 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    job = partial(pose_for, model_file=str(model_file), model_ver=model_ver, fps=fps)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        poses = dict(ex.map(job, [str(p) for p in paths]))
    cpu1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cost = {"clips_computed": todo, "workers": workers, "wall_s": round(time.time() - t0, 1),
            "cpu_s": round((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime), 1)}
    return poses, cost


def parse_params(text: str) -> tuple[RuleParams, dict]:
    """RuleParams from JSON overrides. Unknown keys are an error, so a typo can't silently
    score the defaults (AUDIT D5)."""
    overrides = json.loads(text)
    unknown = set(overrides) - set(RuleParams().to_dict())
    if unknown:
        raise SystemExit(f"unknown --params keys: {', '.join(sorted(unknown))}")
    return RuleParams.from_dict({**RuleParams().to_dict(), **overrides}), overrides


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
                      "detection_times": [[round(s, 2), round(e, 2)] for s, e in det],
                      "posture_observable": res.stats["posture_rule_observable_ratio"],
                      "camera_moving": res.stats["camera_moving_ratio"],
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


def eval_charades_frames(params: RuleParams, poses: dict[str, str], split: str) -> dict:
    from behaviorlens.features import SEATED, STANDING, UNKNOWN, posture_features

    ann = json.loads((vdir() / "charades" / "annotations.json").read_text())
    labels, states = [], []
    for a in ann:
        if split != "all" and split_of(a["video_id"]) != split:
            continue
        cache = poses.get(str(vdir() / "charades" / "videos" / f"{a['video_id']}.mp4"))
        if not cache or cache.startswith("ERROR"):
            continue
        seq = PoseSequence.load_npz(cache)
        t = seq.timestamps_s

        def inside(ids: set[int], t=t, a=a):
            m = np.zeros(len(t), bool)
            for lab, (s, e) in zip(a["labels"], a["action_timings"]):
                if lab in ids:
                    m |= (t >= s) & (t <= e)
            return m

        seat, stand, trans = inside(SEATED_ACTIVITIES), inside(STANDING_ACTIVITIES), inside({STAND_UP, SIT_DOWN})
        lab = np.zeros(len(t), int)
        lab[seat & ~stand & ~trans] = SEATED
        lab[stand & ~seat & ~trans] = STANDING
        labels.append(lab)
        states.append(posture_features(seq, params).state)
    if not labels:
        return {}
    lab, st = np.concatenate(labels), np.concatenate(states)
    obs = st != UNKNOWN
    n_seat, n_stand = int(np.sum((lab == SEATED) & obs)), int(np.sum((lab == STANDING) & obs))
    rate = lambda num, den: round(float(num) / den, 4) if den else None  # noqa: E731
    acc_seat = rate(np.sum((lab == SEATED) & (st == SEATED)), n_seat)
    acc_stand = rate(np.sum((lab == STANDING) & (st == STANDING)), n_stand)
    return {
        "labelled_frames_seated": int(np.sum(lab == SEATED)), "labelled_frames_standing": int(np.sum(lab == STANDING)),
        "observed_seated": n_seat, "observed_standing": n_stand,
        "balanced_accuracy": round((acc_seat + acc_stand) / 2, 3) if acc_seat is not None and acc_stand is not None
        else None,
        "seated_called_standing": rate(np.sum((lab == SEATED) & (st == STANDING)), n_seat),
        "standing_called_seated": rate(np.sum((lab == STANDING) & (st == SEATED)), n_stand),
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
                     "hand_observable": res.stats["hand_rule_observable_ratio"],
                     "camera_moving": res.stats["camera_moving_ratio"]})
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
    ap.add_argument("--fps", type=float, default=config.DEFAULT_SAMPLE_FPS, help="pose sampling rate")
    ap.add_argument("--json", help="write full results JSON here")
    a = ap.parse_args()
    params, overrides = parse_params(a.params)
    paths = sorted((vdir() / "charades" / "videos").glob("*.mp4")) + sorted((vdir() / "ucf101").glob("*.avi"))
    if not paths:
        sys.exit("no clips: run scripts/fetch_validation_clips.py first")
    t0 = time.time()
    model_file = config.model_path()
    poses, cost = ensure_poses(paths, a.workers, model_file, a.fps)
    errors = {Path(k).name: v for k, v in poses.items() if v.startswith("ERROR")}
    frames = sum(len(PoseSequence.load_npz(v).timestamps_s) for v in poses.values() if not v.startswith("ERROR"))
    result = {
        "model_version": model_version(model_file),
        "algorithm_version": config.ALGORITHM_VERSION,
        "sample_fps": a.fps,
        "pose_extraction": {**cost, "sampled_frames_all_clips": frames},
        "params": params.to_dict(),
        # Overridden runs are NOT the shipped rules version (AUDIT D5); label them as such.
        "params_override": overrides or None,
        "split": a.split,
        "tolerance_s": a.tol,
        "charades_standing_transition": eval_charades(params, poses, a.split, a.tol),
        "charades_posture_frames": eval_charades_frames(params, poses, a.split),
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
