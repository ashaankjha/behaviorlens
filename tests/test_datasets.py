"""Shared real-video dataset format (scripts/datasets.py), the --dataset evaluation in
scripts/validate_real_video.py and the fixed-camera fetch helpers. No video or network."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ds = _load("datasets")


def ev(etype, s, e, label="x"):
    return {"type": etype, "start_s": s, "end_s": e, "source_label": label}


def rec(vid, events, **kw):
    return {"video_id": vid, "video": f"videos/{vid}.mp4", "fps": 30.0, "duration_s": 60.0,
            "camera": "fixed", "subjects": 1, "events": events, **kw}


GOOD = [rec("a", [ev("standing_transition", 10, 12, "standUp"), ev("sitting_down", 4, 6, "sitDown"),
                  ev("hand_raised", 20, 24, "waveHands"), ev("other", 30, 32, "throw")]),
        rec("b", [], subjects=None, camera="unknown")]


def test_valid_records_pass_and_counts():
    ds.validate_records(GOOD, "t")
    assert ds.event_counts(GOOD) == {"standing_transition": 1, "sitting_down": 1, "hand_raised": 1, "other": 1}


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r[0].pop("fps"), "missing"),
    (lambda r: r[0].update(extra=1), "unknown keys"),
    (lambda r: r[1].update(video_id="a"), "duplicate"),
    (lambda r: r[0].update(video="/abs/a.mp4"), "relative path"),
    (lambda r: r[0].update(video="videos/../../x.mp4"), "relative path"),
    (lambda r: r[0].update(camera="drone"), "camera"),
    (lambda r: r[0].update(fps=0), "fps"),
    (lambda r: r[0].update(duration_s=float("nan")), "duration_s"),
    (lambda r: r[0].update(subjects=True), "subjects"),
    (lambda r: r[0]["events"].append(ev("waving", 1, 2)), "type"),
    (lambda r: r[0]["events"].append(ev("other", 5, 4)), "start_s <= end_s"),
    (lambda r: r[0]["events"].append({"type": "other", "start_s": 1, "end_s": 2}), "exactly"),
])
def test_schema_violations_are_reported(mutate, msg):
    bad = copy.deepcopy(GOOD)
    mutate(bad)
    with pytest.raises(ds.DatasetError, match=msg):
        ds.validate_records(bad, "t")


def test_non_list_is_rejected():
    with pytest.raises(ds.DatasetError):
        ds.validate_records({"video_id": "a"}, "t")


def test_split_is_deterministic_balanced_and_dataset_scoped():
    ids = [f"v{i}" for i in range(400)]
    a = [ds.split_of(ds.split_key("utkinect", i)) for i in ids]
    assert a == [ds.split_of(ds.split_key("utkinect", i)) for i in ids]
    assert 160 < a.count("tune") < 240
    # same ids in another dataset are split independently
    assert a != [ds.split_of(ds.split_key("pkummd2", i)) for i in ids]
    # Charades keeps the bare-id key of rules-v0.1..v0.3, identical to validate_real_video.split_of
    v = _load("validate_real_video")
    assert all(ds.split_of(ds.split_key("charades", i)) == v.split_of(i) for i in ids)
    recs = [rec(i, []) for i in ids[:50]]
    tune, hold = ds.select(recs, "x", "tune"), ds.select(recs, "x", "holdout")
    assert len(tune) + len(hold) == 50 and not {r["video_id"] for r in tune} & {r["video_id"] for r in hold}


def test_load_dataset_converts_legacy_charades(tmp_path):
    d = tmp_path / "charades"
    d.mkdir()
    (d / "annotations.json").write_text(json.dumps(
        [{"video_id": "ABCDE", "length": 30.0, "scene": "Kitchen", "labels": [154, 151, 11],
          "action_timings": [[3.0, 6.0], [10.0, 12.5], [0.0, 30.0]]}]))
    recs = ds.load_dataset("charades", tmp_path)
    assert [e["type"] for e in recs[0]["events"]] == ["standing_transition", "sitting_down", "other"]
    assert recs[0]["camera"] == "handheld" and recs[0]["subjects"] is None
    with pytest.raises(ds.DatasetError, match="not found"):
        ds.load_dataset("nope", tmp_path)


def test_scoring_on_a_tiny_synthetic_dataset():
    r = GOOD[0]
    # one hit (within tol), one confusion with sitting down, one unrelated FP
    m = ds.match_video(r, [(12.5, 13.0), (5.0, 5.5), (40.0, 41.0)], "standing_transition", tol=1.0)
    assert (m["windows"], m["found"], m["detections"], m["correct"], m["confused"]) == (1, 1, 3, 1, 1)
    assert m["fp_labels"] == [["sitDown"], ["(unlabelled)"]]
    miss = ds.match_video(GOOD[1], [], "standing_transition", 1.0)
    s = ds.score_events([m, miss])
    assert s["recall"] == 1.0 and s["precision"] == pytest.approx(0.333, abs=1e-3)
    assert s["false_positives"] == 2 and s["confused_with_sitting_down"] == 1
    assert s["f1"] == pytest.approx(0.5)
    assert s["fp_overlapping_labels"] == {"sitDown": 1, "(unlabelled)": 1}
    # a detection overlapping only an "other" action is attributed to its source label
    h = ds.match_video(r, [(21, 23), (30.5, 31.5)], "hand_raised", 0.0)
    assert (h["found"], h["correct"], h["fp_labels"]) == (1, 1, [["throw"]])
    assert ds.score_events([])["recall"] is None


def test_wilson_interval():
    assert ds.wilson(0, 0) is None
    lo, hi = ds.wilson(2, 10)
    assert lo == pytest.approx(0.057, abs=1e-3) and hi == pytest.approx(0.510, abs=1e-3)


def test_eval_datasets_pools_and_scores_with_a_stub_analyzer(tmp_path, monkeypatch):
    v = _load("validate_real_video")
    monkeypatch.setenv("BEHAVIORLENS_DATA_DIR", str(tmp_path))
    loaded = {"d1": [GOOD[0]], "d2": [rec("c", [ev("standing_transition", 5, 6)])]}
    poses = {str(ds.video_path(n, r, v.vdir())): f"cache-{n}-{r['video_id']}"
             for n, recs in loaded.items() for r in recs}
    stats = {"posture_rule_observable_ratio": 0.9, "hand_rule_observable_ratio": 0.8,
             "pose_detected_ratio": 1.0, "camera_moving_ratio": 0.0}
    outputs = {"cache-d1-a": [("standing_transition", 10.5, 11.0), ("hand_raised", 21.0, 22.0)],
               "cache-d2-c": [("standing_transition", 20.0, 21.0)]}

    def analyze(cache, clip_id):
        return SimpleNamespace(events=[SimpleNamespace(event_type=t, start_s=s, end_s=e)
                                       for t, s, e in outputs[cache]], stats=stats)

    out = v.eval_datasets(analyze, poses, loaded, "all", 1.0)
    assert set(out) == {"pooled", "d1", "d2"}
    st = out["pooled"]["standing_transition"]
    assert (st["windows"], st["found"], st["detections"], st["correct"]) == (2, 1, 2, 1)
    assert out["d1"]["hand_raised"]["recall"] == 1.0 and out["d2"]["standing_transition"]["recall"] == 0.0
    assert out["pooled"]["camera_moving_ratio"]["max"] == 0.0
    assert "_rows" not in out["pooled"]


def test_cache_key_is_dataset_scoped_but_charades_unchanged(tmp_path):
    v = _load("validate_real_video")
    k = lambda p: v.cache_path(tmp_path / p, "m@1", 5.0, 4).name  # noqa: E731
    assert k("utkinect/videos/x.mp4") != k("pkummd2/videos/x.mp4")
    assert k("charades/videos/ABCDE.mp4").startswith("videos_ABCDE_")


def test_fixed_camera_fetch_helpers():
    f = _load("fetch_fixed_camera_datasets")
    labels = f.parse_utk_labels("s01_e01\nwalk: 252 390\nsitDown: 572 686\ncarry: NaN NaN\n")
    assert labels == {"s01_e01": {"walk": (252, 390), "sitDown": (572, 686), "carry": None}}
    # frame numbers advance at 60/s (30 fps, +2 per frame): gaps are held, not dropped
    assert f.utk_frame_durations([100, 102, 110]) == pytest.approx([2 / 60, 8 / 60, 2 / 60])
    events, subjects = f.parse_pku_label("34,30,90,2\n13,120,180,1\n14,200,260,2\n")
    assert [(e["type"], e["start_s"], e["end_s"]) for e in events] == [
        ("standing_transition", 1.0, 3.0), ("hand_raised", 4.0, 6.0), ("other", 6.667, 8.667)]
    assert subjects == 2 and events[2]["source_label"] == "handshaking"
