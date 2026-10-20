"""Score a grid of RuleParams overrides on shared-format datasets in one pass.

Tracking does not depend on RuleParams, so each clip's subject track is computed once and only
the rules (pipeline.analyze_pose, the production rule path) are re-run per setting. Landmark
caches must already exist (run scripts/validate_real_video.py --dataset ... first).

Usage:
    python scripts/tune_grid.py --dataset charades,utkinect,pkummd1,pkummd2 --split tune \\
        --grid '{"dwell_s": [0.4, 0.6, 0.8, 1.0], "max_transition_s": [3, 5, 7]}' --json grid.json \\
        [--tracking-params '{"merge_fragments": true}']

Output: one row per setting with pooled and per-dataset standing_transition / hand_raised
scores (scripts/datasets.score_events).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets  # noqa: E402
import validate_real_video as vrv  # noqa: E402

from behaviorlens import config  # noqa: E402
from behaviorlens.config import RuleParams, TrackingParams  # noqa: E402
from behaviorlens.pipeline import analyze_pose  # noqa: E402
from behaviorlens.pose import model_version  # noqa: E402

TYPES = ("standing_transition", "hand_raised")


def settings(grid: dict[str, list]) -> list[dict]:
    keys = sorted(grid)
    unknown = set(keys) - set(RuleParams().to_dict())
    if unknown:
        raise SystemExit(f"unknown grid keys: {', '.join(sorted(unknown))}")
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def clip_detections(job: tuple[str, str, str, list[dict], dict]) -> tuple[str, list[dict[str, list]]]:
    """Detections per setting for one clip (subject track computed once)."""
    cache, clip_id, attribution, overrides, tracking = job
    base = RuleParams().to_dict()
    seq = vrv.Analyzer(attribution, RuleParams(), TrackingParams.from_dict(tracking)).sequence(cache, clip_id)
    out = []
    for o in overrides:
        res = analyze_pose(seq, RuleParams.from_dict({**base, **o}), clip_id)
        out.append({t: [(e.start_s, e.end_s) for e in res.events if e.event_type == t] for t in TYPES})
    return cache, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="tune", choices=["all", "tune", "holdout"])
    ap.add_argument("--grid", required=True, help="JSON {param: [values]}")
    ap.add_argument("--attribution", default="tracked", choices=["tracked", "largest", "legacy"])
    ap.add_argument("--max-people", type=int, default=config.DEFAULT_MAX_PEOPLE)
    ap.add_argument("--fps", type=float, default=config.DEFAULT_SAMPLE_FPS)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tracking-params", default="{}", help="JSON overrides for TrackingParams")
    ap.add_argument("--json")
    a = ap.parse_args()
    tracking = vrv.parse_tracking(a.tracking_params).to_dict()
    grid = settings(json.loads(a.grid))
    mver = model_version(config.model_path())
    max_people = None if a.attribution == "legacy" else a.max_people
    clips = []  # (dataset, record, cache)
    for name in a.dataset.split(","):
        for rec in datasets.select(datasets.load_dataset(name, vrv.vdir()), name, a.split):
            cache = vrv.cache_path(datasets.video_path(name, rec, vrv.vdir()), mver, a.fps, max_people)
            if cache.is_file():
                clips.append((name, rec, str(cache)))
            else:
                print(f"no cache for {name}:{rec['video_id']} (run validate_real_video.py first)", file=sys.stderr)
    jobs = [(c, r["video_id"], a.attribution, grid, tracking) for _, r, c in clips]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        dets = dict(ex.map(clip_detections, jobs, chunksize=4))
    rows = []
    for i, o in enumerate(grid):
        per: dict[str, dict[str, list]] = {}
        for name, rec, cache in clips:
            d = dets[cache][i]
            for t in TYPES:
                per.setdefault(name, {}).setdefault(t, []).append(datasets.match_video(rec, d[t], t, a.tol))
        row: dict = {"params": o, "pooled": {}, "datasets": {}}
        for t in TYPES:
            row["pooled"][t] = datasets.score_events(m for p in per.values() for m in p[t])
        for name, p in per.items():
            row["datasets"][name] = {t: datasets.score_events(p[t]) for t in TYPES}
        rows.append(row)
    out = {"model_version": mver, "algorithm_version": config.ALGORITHM_VERSION, "split": a.split,
           "attribution": a.attribution, "tracking_params": tracking, "tolerance_s": a.tol,
           "clips": len(clips), "rows": rows}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    for r in rows:
        st, hr = r["pooled"]["standing_transition"], r["pooled"]["hand_raised"]
        print(json.dumps(r["params"]), f"ST f1={st['f1']} {st['found']}/{st['windows']}",
              f"{st['correct']}/{st['detections']} HR f1={hr['f1']}")


if __name__ == "__main__":
    main()
