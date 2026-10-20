"""End to end: recording kit -> shared dataset format -> real-video evaluator.

SYNTHETIC clips from ``ingest_staged_recordings --synthetic-example`` are ingested, loaded by
``scripts/datasets.py`` and scored by ``validate_real_video.eval_datasets`` with the real rules.
MediaPipe finds no pose in stick figures, so the landmark caches are the demo generator's
synthetic PoseSequence for the same timeline (what MediaPipe would ideally return).
"""

from __future__ import annotations

import importlib.util
import json
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from behaviorlens.config import RuleParams, TrackingParams
from behaviorlens.demo_generator import generate_pose_sequence
from behaviorlens.models import PoseSequence

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kit = _load("ingest_staged_recordings")
ds = _load("datasets")
vrv = _load("validate_real_video")

# make_synthetic_example renders demo seconds [0, 20) and [20, 40) as two takes.
TAKES = ((0.0, 20.0), (20.0, 40.0))


def _window(seq: PoseSequence, t0: float, t1: float) -> PoseSequence:
    keep = (seq.timestamps_s >= t0) & (seq.timestamps_s < t1)
    return PoseSequence(timestamps_s=seq.timestamps_s[keep] - t0, frame_indices=np.arange(int(keep.sum())),
                        landmarks=seq.landmarks[keep], sample_fps=seq.sample_fps, duration_s=t1 - t0,
                        num_people=np.asarray(seq.num_people)[keep], data_mode="synthetic")


@pytest.fixture
def staged(tmp_path, monkeypatch) -> list[dict]:
    monkeypatch.setenv("BEHAVIORLENS_DATA_DIR", str(tmp_path / "data"))
    raw = tmp_path / "raw"
    kit.make_synthetic_example(raw)
    kit.ingest(raw, out=io.StringIO())
    return ds.load_dataset("staged", vrv.vdir())


def test_ingested_recordings_load_in_the_shared_format(staged):
    assert len(staged) == 2
    for rec in staged:
        assert rec["camera"] == "fixed" and rec["subjects"] == 1
        assert ds.video_path("staged", rec, vrv.vdir()).is_file()
        assert "synthetic" not in rec["video_id"]  # opaque id, not the source filename
    counts = ds.event_counts(staged)
    assert counts["standing_transition"] >= 1 and counts["hand_raised"] >= 1
    # The split is keyed by dataset so staged ids can't collide with other datasets' splits.
    assert {ds.split_key("staged", r["video_id"]) for r in staged} == {f"staged:{r['video_id']}" for r in staged}


def test_evaluator_scores_ingested_recordings_with_the_real_rules(staged, tmp_path):
    full = generate_pose_sequence()
    # annotations.json order is not guaranteed: take 2 is the one with the scripted occlusion.
    takes = sorted(staged, key=lambda r: min((e["start_s"] for e in r["events"]
                                               if e["source_label"] == "occlusion"), default=-1.0))
    poses = {}
    for rec, (t0, t1) in zip(takes, TAKES):
        cache = tmp_path / f"{rec['video_id']}.npz"
        _window(full, t0, t1).save_npz(cache)
        poses[str(ds.video_path("staged", rec, vrv.vdir()))] = str(cache)
        # The evaluator's landmark cache name is prefixed by the dataset, never the source name.
        assert vrv.cache_path(ds.video_path("staged", rec, vrv.vdir()), "m", 5.0, 4).name.startswith("staged_")

    analyze = vrv.Analyzer("legacy", RuleParams(), TrackingParams())
    out = vrv.eval_datasets(analyze, poses, {"staged": staged}, "all", 1.0)
    st, hr = out["staged"]["standing_transition"], out["staged"]["hand_raised"]

    assert out["staged"]["videos"] == 2
    assert st["windows"] == ds.event_counts(staged)["standing_transition"]
    assert hr["windows"] == ds.event_counts(staged)["hand_raised"]
    assert st["precision"] == 1.0 and hr["recall"] == 1.0
    # Take 2 starts mid stand-up (demo 20.0 s), so no seated dwell precedes it and the rule must
    # not report it: unobservable onset, never a guessed transition. Take 1's stand-up is found.
    assert (st["found"], st["windows"]) == (1, 2)
    clips = {c["clip"]: c for c in out["staged"]["per_clip"]}
    assert clips[takes[0]["video_id"]]["standing"]["found"] == 1
    assert clips[takes[1]["video_id"]]["standing"]["found"] == 0
    # The one hand_raised "false positive" is the scripted raise annotator B missed: ingest leaves
    # unadjudicated disagreements out of the reference set, so a correct detection scores as wrong.
    # Adjudicate (or ingest with --strict) before trusting precision on staged recordings.
    assert (hr["detections"], hr["correct"]) == (3, 2)
    take1 = takes[0]
    report = json.loads((vrv.vdir() / "staged" / "ingest_report.json").read_text())
    pending = [(p["start_s"], p["end_s"]) for p in report["videos"][take1["video_id"]]["disagreements"]
               if p["label"] == "hand_raised"]
    wrong = [tuple(d) for d in clips[take1["video_id"]]["hand_detections"]
             if not any(ds.overlaps(tuple(d), w, 1.0) for w in ds.windows(take1, "hand_raised"))]
    assert len(wrong) == 1 and any(ds.overlaps(wrong[0], p) for p in pending)
    assert out["staged"]["camera_moving_ratio"]["max"] is None  # synthetic caches carry no motion
