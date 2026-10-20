import io
import json
import os
import urllib.request

import cv2
import numpy as np
import pytest

from behaviorlens import config, export, metrics, pipeline
from behaviorlens.config import AnalysisSettings
from behaviorlens.demo_generator import GROUND_TRUTH, generate_pose_sequence
from behaviorlens.pose import MediaPipePoseEstimator, ModelNotFoundError
from behaviorlens.video_io import (
    VideoValidationError,
    probe_video,
    sample_frame_indices,
    sanitize_filename,
    validate_upload,
)

from conftest import make_multi, make_seq, person, requires_model, window, write_video


def upload(store, path, name=None):
    data = path.read_bytes()
    return pipeline.create_upload_session(store, name or path.name, io.BytesIO(data), len(data))


class FakeEstimator:
    """Returns landmarks from a synthetic sequence by timestamp (no inference)."""

    model_version = "fake@1"

    def __init__(self, seq):
        self.seq = seq
        self.calls = 0

    def detect(self, frame, t):
        self.calls += 1
        k = int(np.argmin(np.abs(self.seq.timestamps_s - t)))
        row = self.seq.landmarks[k]
        return (None, 0) if np.isnan(row[0, 0]) else (row, 1)

    def close(self):
        pass


class FakeMultiEstimator:
    """Returns every person of a synthetic MultiPoseSequence by timestamp (no inference)."""

    model_version = "fakemulti@1"

    def __init__(self, multi, **kwargs):
        self.multi, self.kwargs, self.calls = multi, kwargs, 0

    def detect_all(self, frame, t):
        self.calls += 1
        return self.multi.detections(int(np.argmin(np.abs(self.multi.timestamps_s - t))))

    def detect(self, frame, t):  # pragma: no cover - the pipeline prefers detect_all
        raise AssertionError("detect_all should be used")

    def close(self):
        pass


def two_people(t):
    """A large seated person throughout, and a smaller person who stands up at 3 s."""
    return [person(0.3, 1.0, stand=0.0), person(0.75, 0.5, stand=1.0 if t >= 3 else 0.0)]


# ------------------------------------------------------------ demo (synthetic) end-to-end
def test_demo_session_end_to_end(store):
    sid = pipeline.create_demo_session(store)
    result = pipeline.run_analysis(store, sid)
    events = store.list_events(sid)
    assert all(e.data_mode == "synthetic" for e in events)
    types = [e.event_type for e in events]
    assert types.count("hand_raised") == 3 and types.count("standing_transition") == 2
    assert result["stats"]["pose_detected_ratio"] < 1.0  # the out-of-frame segment
    m = metrics.match_events([e.model_dump() for e in events], GROUND_TRUTH, tolerance_s=1.0)
    assert m["per_type"]["hand_raised"]["recall"] == 1.0 and m["per_type"]["hand_raised"]["precision"] == 1.0
    # the transition that happens while legs are hidden is an expected, documented miss
    assert m["per_type"]["standing_transition"]["fn"] == 1
    assert store.get_session(sid)["status"] == "review_ready"
    assert json.loads(export.to_json(store, sid))["data_mode"] == "synthetic"


def test_demo_fixture_is_deterministic():
    a, b = generate_pose_sequence(), generate_pose_sequence()
    assert np.array_equal(np.nan_to_num(a.landmarks), np.nan_to_num(b.landmarks))


# ------------------------------------------------------------ video validation
def test_validate_upload_rejects_bad_inputs():
    good_head = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4
    assert validate_upload("a.MP4", 10, good_head) == ".mp4"
    with pytest.raises(VideoValidationError, match="Unsupported"):
        validate_upload("notes.txt", 10, good_head)
    with pytest.raises(VideoValidationError, match="limit"):
        validate_upload("a.mp4", config.MAX_UPLOAD_BYTES + 1, good_head)
    with pytest.raises(VideoValidationError, match="empty"):
        validate_upload("a.mp4", 0, good_head)
    with pytest.raises(VideoValidationError, match="does not look like"):
        validate_upload("a.mp4", 10, b"<html>not a video")  # MIME/extension spoofing
    with pytest.raises(VideoValidationError, match="AVI"):
        validate_upload("a.avi", 10, good_head)


def test_sanitize_filename():
    assert sanitize_filename("../../etc/pass wd?.MP4") == "pass_wd.mp4"
    assert sanitize_filename("C:\\Users\\x\\clip.mov") == "clip.mov"


def test_upload_valid_video_creates_session(store, tmp_path):
    sid = upload(store, write_video(tmp_path / "clip.mp4", 3.0))
    s = store.get_session(sid)
    assert s["status"] == "ready" and s["duration_s"] == pytest.approx(3.0, abs=0.1)
    preview = store.path(s["preview_relpath"])
    assert preview.is_file()
    assert probe_video(preview).codec in ("avc1", "h264")  # transcoded for browser playback


def test_corrupted_video_rejected_and_cleaned_up(store, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(5000))
    with pytest.raises(VideoValidationError):
        upload(store, bad)
    assert store.list_sessions() == []
    assert not any((store.root / "sessions" / "_incoming").iterdir())


def test_short_clip_rejected(store, tmp_path):
    with pytest.raises(VideoValidationError, match="only"):
        upload(store, write_video(tmp_path / "short.mp4", 0.4))


def test_bad_fps_rejected(store, tmp_path):
    with pytest.raises(VideoValidationError, match="frame rate"):
        upload(store, write_video(tmp_path / "slow.avi", 6, fps=0.5, fourcc="MJPG"))


def test_too_large_upload_rejected(store, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    with pytest.raises(VideoValidationError, match="limit"):
        upload(store, write_video(tmp_path / "clip.mp4", 2))


def test_sample_frame_indices_use_source_fps():
    from behaviorlens.models import VideoMeta

    meta = VideoMeta(fps=30, frame_count=90, width=10, height=10, duration_s=3.0)
    idx = sample_frame_indices(meta, 5)
    assert list(idx[:4]) == [0, 6, 12, 18] and len(idx) == 15
    meta2 = VideoMeta(fps=4, frame_count=12, width=10, height=10, duration_s=3.0)
    assert len(sample_frame_indices(meta2, 5)) == 12  # never above source fps


# ------------------------------------------------------------ video pipeline with fake estimator
def test_video_pipeline_with_fake_estimator_and_cache(store, tmp_path):
    seq = make_seq(6, right_up=window(2, 4))
    sid = upload(store, write_video(tmp_path / "clip.mp4", 6.0))
    fake = FakeEstimator(seq)
    res = pipeline.run_analysis(store, sid, AnalysisSettings(sample_fps=5), estimator_factory=lambda: fake)
    assert res["added"] >= 1
    (hand,) = [e for e in store.list_events(sid) if e.event_type == "hand_raised"]
    assert hand.start_s == pytest.approx(2.0, abs=0.25) and hand.data_mode == "video_inference"
    analysis = store.latest_analysis(sid)
    assert analysis["model_version"] == "fake@1"
    landmarks = store.load_landmarks(analysis)
    assert np.all(np.diff(landmarks.timestamps_s) > 0)

    calls = fake.calls
    pipeline.run_analysis(store, sid, AnalysisSettings(sample_fps=5), estimator_factory=lambda: fake)
    assert fake.calls == calls  # second run served from the pose cache


def test_multi_person_video_is_tracked_and_subject_is_selectable(store, tmp_path):
    multi = make_multi(6, people=two_people)
    sid = upload(store, write_video(tmp_path / "two.mp4", 6.0))
    fake = FakeMultiEstimator(multi)
    res = pipeline.run_analysis(store, sid, AnalysisSettings(sample_fps=5), estimator_factory=lambda: fake)
    t = res["stats"]["tracking"]
    assert t["tracks"] == 2 and t["substantial_tracks"] == 2 and t["subject_selection"] == "auto"
    assert t["max_people"] == 2 and len(t["per_track"]) == 2
    auto = t["subject_track_id"]
    assert any("2 people were tracked" in w for w in res["warnings"])
    # the automatic subject (larger, seated person) never stands up
    assert not any(e.event_type == "standing_transition" for e in store.list_events(sid))
    analysis = store.latest_analysis(sid)
    assert analysis["settings"]["subject_track_id"] is None and analysis["settings"]["max_people"] == 4
    assert pipeline.tracks_path(store, sid, analysis["analysis_id"]).is_file()

    # reviewer picks the other person: served from the cache, events computed on that track only
    other = next(p["track_id"] for p in t["per_track"] if p["track_id"] != auto)
    calls = fake.calls
    res2 = pipeline.run_analysis(store, sid, AnalysisSettings(sample_fps=5, subject_track_id=other),
                                 estimator_factory=lambda: fake)
    assert fake.calls == calls
    assert res2["stats"]["tracking"]["subject_track_id"] == other
    assert res2["stats"]["tracking"]["subject_selection"] == "reviewer"
    (st,) = [e for e in store.list_events(sid) if e.event_type == "standing_transition"]
    assert st.evidence["tracking"]["subject_track_id"] == other
    assert st.evidence["tracking"]["tracked_people"] == 2


def test_pose_cache_key_includes_max_people(store, tmp_path):
    multi = make_multi(3, people=lambda t: [person(0.5, 0.8)])
    sid = upload(store, write_video(tmp_path / "one.mp4", 3.0))
    fake = FakeMultiEstimator(multi)
    pipeline.run_analysis(store, sid, AnalysisSettings(max_people=4), estimator_factory=lambda: fake)
    calls = fake.calls
    pipeline.run_analysis(store, sid, AnalysisSettings(max_people=2), estimator_factory=lambda: fake)
    assert fake.calls == 2 * calls  # different num_poses -> different landmarks -> new extraction
    keys = {pipeline._pose_cache_path(store, "c", "m@1", 5.0, n) for n in (None, 2, 4)}
    keys |= {pipeline._pose_cache_path(store, "c", "m@1", 10.0, 4)}
    assert len(keys) == 4


def test_default_estimator_gets_max_people(store, tmp_path, monkeypatch):
    made = []

    def factory(**kwargs):
        made.append(kwargs)
        return FakeMultiEstimator(make_multi(2, people=lambda t: [person()]), **kwargs)

    monkeypatch.setattr(pipeline, "MediaPipePoseEstimator", factory)
    sid = upload(store, write_video(tmp_path / "one.mp4", 2.0))
    pipeline.run_analysis(store, sid, AnalysisSettings(max_people=3))
    assert made == [{"max_people": 3}]


def test_overlay_labels_tracks_and_is_invalidated_by_reanalysis(store, tmp_path):
    multi = make_multi(4, people=two_people)
    sid = upload(store, write_video(tmp_path / "two.mp4", 4.0))
    run = lambda **kw: pipeline.run_analysis(store, sid, AnalysisSettings(**kw),
                                             estimator_factory=lambda: FakeMultiEstimator(multi))
    run()
    out = pipeline.render_overlay(store, sid)
    assert out.is_file() and probe_video(out).frame_count > 0
    run(subject_track_id=2)
    assert not out.exists() and store.get_session(sid)["overlay_relpath"] is None


def test_missing_model_is_actionable(store, tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIORLENS_MODEL_PATH", str(tmp_path / "nope.task"))
    with pytest.raises(ModelNotFoundError, match="download_model"):
        MediaPipePoseEstimator()
    sid = upload(store, write_video(tmp_path / "clip.mp4", 2.0))
    with pytest.raises(ModelNotFoundError):
        pipeline.run_analysis(store, sid)
    s = store.get_session(sid)
    assert s["status"] == "failed" and "download_model" in s["error"]


# ------------------------------------------------------------ real MediaPipe inference
@requires_model
@pytest.mark.inference
def test_real_inference_without_person_yields_no_pose(store, tmp_path):
    """Real MediaPipe on a video with no person: explicit no_pose, no events, no crash."""
    sid = upload(store, write_video(tmp_path / "empty.mp4", 3.0))
    res = pipeline.run_analysis(store, sid)
    assert res["stats"]["pose_detected_ratio"] == 0.0
    assert any("No person" in w for w in res["warnings"])
    assert store.list_events(sid) == []
    assert store.get_session(sid)["status"] == "review_ready"


ASSET_BASE = "https://storage.googleapis.com/mediapipe-assets/"


def still_video(tmp_path, image_name: str, frames: int = 25):
    """Video of a public MediaPipe sample photo of an adult (downloaded to a temp dir, never committed)."""
    img_path = tmp_path / image_name
    try:
        urllib.request.urlretrieve(ASSET_BASE + image_name, img_path)
    except Exception:
        pytest.skip("sample person image not reachable (offline)")
    img = cv2.imread(str(img_path))
    if img is None:
        pytest.skip("sample person image could not be read")
    h = 480
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]) // 2 * 2, h))
    video = tmp_path / f"{img_path.stem}.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (img.shape[1], h))
    for _ in range(frames):
        writer.write(img)
    writer.release()
    return video


@requires_model
@pytest.mark.inference
def test_real_inference_standing_person(store, tmp_path):
    """Real MediaPipe inference on a full-height standing adult: upright posture, no raised hand."""
    sid = upload(store, still_video(tmp_path, "male_full_height_hands.jpg"))
    res = pipeline.run_analysis(store, sid)
    assert res["stats"]["pose_detected_ratio"] >= 0.9
    types = {e.event_type for e in store.list_events(sid)}
    assert types == {"sustained_standing"}
    seq = store.load_landmarks(store.latest_analysis(sid))
    assert seq.data_mode == "video_inference"
    out = pipeline.render_overlay(store, sid)
    assert out.is_file() and probe_video(out).frame_count > 0


@requires_model
@pytest.mark.inference
def test_real_inference_lunge_with_level_arms_has_no_events(store, tmp_path):
    """Regression: horizontal arms are not 'hand_raised' and a lunge is not 'seated'."""
    sid = upload(store, still_video(tmp_path, "pose.jpg"))
    res = pipeline.run_analysis(store, sid)
    assert res["stats"]["pose_detected_ratio"] >= 0.9
    assert store.list_events(sid) == []


def test_event_thumbnails_are_local_and_deleted_with_session(store):
    sid = pipeline.create_demo_session(store)
    thumbs = pipeline.event_thumbnails(store, sid, 5.2, 6.2)
    assert [t for t, _ in thumbs] == [4.7, 6.7]
    for _, path in thumbs:
        img = cv2.imread(str(path))
        assert img is not None and img.shape[0] == 200
        assert store.session_dir(sid) in path.parents
    # clamped to the video bounds
    assert pipeline.event_thumbnails(store, sid, 0.1, 39.9)[0][0] == 0.0
    store.delete_session(sid)
    assert not any(p.exists() for _, p in thumbs)


def test_validation_harness_helpers_are_deterministic():
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "validate_real_video", _P(__file__).resolve().parents[1] / "scripts" / "validate_real_video.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    ids = [f"clip{i}" for i in range(200)]
    splits = [v.split_of(i) for i in ids]
    assert splits == [v.split_of(i) for i in ids]  # stable across calls/processes (sha1, not hash())
    assert 70 < splits.count("tune") < 130
    assert v.overlaps((1, 2), (2.5, 3), tol=1.0) and not v.overlaps((1, 2), (3.5, 4), tol=1.0)


def test_validation_cache_key_includes_model_and_fps(tmp_path, monkeypatch):
    # LEARNINGS M21: a clip+fps-only key silently reused lite landmarks for other models.
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "validate_real_video", _P(__file__).resolve().parents[1] / "scripts" / "validate_real_video.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    monkeypatch.setenv("BEHAVIORLENS_DATA_DIR", str(tmp_path))
    clip = tmp_path / "validation" / "ucf101" / "v_PullUps_g01_c01.avi"
    lite = v.cache_path(clip, "pose_landmarker_lite@aaaaaaaaaaaa", 5.0)
    keys = {lite,
            v.cache_path(clip, "pose_landmarker_full@bbbbbbbbbbbb", 5.0),
            v.cache_path(clip, "pose_landmarker_lite@cccccccccccc", 5.0),  # same name, new weights
            v.cache_path(clip, "pose_landmarker_lite@aaaaaaaaaaaa", 10.0),
            # all-people caches: num_poses changes MediaPipe's output, so it is part of the key
            v.cache_path(clip, "pose_landmarker_lite@aaaaaaaaaaaa", 5.0, max_people=4),
            v.cache_path(clip, "pose_landmarker_lite@aaaaaaaaaaaa", 5.0, max_people=2)}
    assert len(keys) == 6
    assert lite == v.cache_path(clip, "pose_landmarker_lite@aaaaaaaaaaaa", 5.0)
    assert lite.parent == tmp_path / "validation" / "cache"


def test_validation_params_reject_typos():
    """AUDIT D5: RuleParams.from_dict drops unknown keys, so the script must reject them."""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "validate_real_video", _P(__file__).resolve().parents[1] / "scripts" / "validate_real_video.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    params, overrides = v.parse_params('{"dwell_s": 1.0}')
    assert params.dwell_s == 1.0 and overrides == {"dwell_s": 1.0}
    with pytest.raises(SystemExit, match="unknown --params keys: dwel_s"):
        v.parse_params('{"dwel_s": 1.0}')
    assert v.parse_tracking('{"max_gap_s": 1.0}').max_gap_s == 1.0
    with pytest.raises(SystemExit, match="unknown --tracking-params keys: max_gap"):
        v.parse_tracking('{"max_gap": 1.0}')
