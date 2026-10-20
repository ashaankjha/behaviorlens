"""Input limits, structured logging without PHI/paths, and model supply-chain checks."""

import hashlib
import io
import json
import logging

import numpy as np
import pytest
from conftest import make_seq, write_video

from behaviorlens import config, metrics, pipeline, pose
from behaviorlens.logging_setup import configure_logging, redact
from behaviorlens.models import NUM_LANDMARKS, PoseSequence
from behaviorlens.video_io import VideoValidationError, probe_video


class FakeEstimator:
    model_version = "fake@1"

    def detect(self, frame, t):
        return make_seq(duration=1, fps=1).landmarks[0], 1

    def close(self):
        pass


def test_logs_are_json_and_contain_no_paths_or_filenames(store, tmp_path):
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    try:
        video = write_video(tmp_path / "Jane_Doe_private.mp4", 2.0)
        with open(video, "rb") as f:
            sid = pipeline.create_upload_session(store, "Jane_Doe_private.mp4", f, video.stat().st_size)
        pipeline.run_analysis(store, sid, estimator_factory=FakeEstimator)
        logging.getLogger("behaviorlens.test").error("failed reading %s", str(tmp_path / "x.mp4"), exc_info=True)
    finally:
        configure_logging("WARNING")
    text = buf.getvalue()
    records = [json.loads(line) for line in text.splitlines()]
    events = [r["event"] for r in records]
    assert "session_created" in events and "analysis_started" in events and "analysis_completed" in events
    assert str(tmp_path) not in text and str(store.root) not in text
    assert "Jane_Doe" not in text and ".mp4" not in text.replace("<redacted>", "")
    done = next(r for r in records if r["event"] == "analysis_completed")
    assert done["session_id"] == sid and done["frames"] > 0 and "elapsed_s" in done


def test_failed_analysis_logs_exception_type_only(store, tmp_path):
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    try:
        video = write_video(tmp_path / "clip.mp4", 2.0)
        with open(video, "rb") as f:
            sid = pipeline.create_upload_session(store, "clip.mp4", f, video.stat().st_size)
        (store.root / store.get_session(sid)["video_relpath"]).unlink()
        with pytest.raises(VideoValidationError):
            pipeline.run_analysis(store, sid, estimator_factory=FakeEstimator)
    finally:
        configure_logging("WARNING")
    failed = [json.loads(x) for x in buf.getvalue().splitlines() if '"analysis_failed"' in x]
    assert failed and failed[0]["exc_type"] == "VideoValidationError" and "Traceback" not in buf.getvalue()


def test_redact_removes_paths():
    assert "/home" not in redact("could not open /home/user/videos/Jane Doe.mp4 now")
    assert "C:\\" not in redact(r"bad C:\Users\jane\clip.mp4")
    assert redact("rules-v0.2 at 5 fps") == "rules-v0.2 at 5 fps"


def test_upload_copy_is_bounded(store, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    data = io.BytesIO(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 5000)
    with pytest.raises(VideoValidationError, match="upload limit"):
        pipeline.create_upload_session(store, "big.mp4", data, 500)  # client under-reports its size
    assert list((store.root / "sessions" / "_incoming").iterdir()) == []


def test_oversized_resolution_rejected(tmp_path, monkeypatch):
    video = write_video(tmp_path / "v.mp4", 1.5, size=(320, 240))
    monkeypatch.setattr(config, "MAX_FRAME_PIXELS", 320 * 240 - 1)
    with pytest.raises(VideoValidationError, match="resolution"):
        probe_video(video)


def test_pose_sequence_rejects_invalid_metadata():
    lms = np.zeros((2, NUM_LANDMARKS, 5))
    with pytest.raises(ValueError, match="sample_fps"):
        PoseSequence(timestamps_s=[0, 1], frame_indices=[0, 1], landmarks=lms, sample_fps=float("nan"), duration_s=2)
    with pytest.raises(ValueError, match="data_mode"):
        PoseSequence(timestamps_s=[0, 1], frame_indices=[0, 1], landmarks=lms, sample_fps=1, duration_s=2,
                     data_mode="other")  # type: ignore[arg-type]


@pytest.mark.parametrize("text, message", [
    ("event_type,start_s,end_s\nhand_raised,0,inf\n", "finite"),
    ("event_type,start_s,end_s\n,0,1\n", "event_type"),
])
def test_ground_truth_rejects_non_finite_and_blank(text, message):
    with pytest.raises(ValueError, match=message):
        metrics.parse_ground_truth_csv(text)


def test_ground_truth_row_cap(monkeypatch):
    monkeypatch.setattr(config, "MAX_GROUND_TRUTH_ROWS", 3)
    rows = "".join(f"hand_raised,{i},{i + 1}\n" for i in range(4))
    with pytest.raises(ValueError, match="more than 3 rows"):
        metrics.parse_ground_truth_csv("event_type,start_s,end_s\n" + rows)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_model_download_verifies_checksum(tmp_path, monkeypatch):
    payload = b"model-bytes"
    monkeypatch.setattr(pose.urllib.request, "urlopen", lambda url, timeout: _FakeResponse(payload))
    dest = tmp_path / "m.task"
    with pytest.raises(pose.ModelIntegrityError, match="does not match"):
        pose.download_model(dest, url="https://example.invalid/m.task", expected_sha256="0" * 64)
    assert not dest.exists() and not dest.with_suffix(".part").exists()
    good = hashlib.sha256(payload).hexdigest()
    assert pose.download_model(dest, url="https://example.invalid/m.task", expected_sha256=good) == dest
    assert pose.verify_model(dest, good) and not pose.verify_model(dest, "0" * 64)
    with pytest.raises(ValueError, match="https"):
        pose.download_model(dest, url="http://example.invalid/m.task")


def test_pinned_model_matches_downloaded_file():
    default = config.default_model_path()
    if config.model_path() != default or not default.is_file():
        pytest.skip("default pose model not downloaded")
    assert pose.verify_model()


def test_overlay_writer_cleans_up_on_failure(tmp_path):
    from behaviorlens.video_io import FrameWriter

    out = tmp_path / "o.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        with FrameWriter(out, 64, 48, 5) as w:
            w.write(np.zeros((48, 64, 3), np.uint8))
            raise RuntimeError("boom")
    assert not out.exists()


def test_decode_is_capped_independently_of_header(tmp_path, monkeypatch):
    from behaviorlens.video_io import iter_sampled_frames

    video = write_video(tmp_path / "v.mp4", 3.0, fps=10)
    meta = probe_video(video)
    monkeypatch.setattr(config, "MAX_DECODED_FRAMES", 12)
    idx = [i for i, _, _ in iter_sampled_frames(video, meta, 10.0)]
    assert idx and max(idx) < 12


def test_mid_stream_oversized_frame_is_refused(tmp_path, monkeypatch):
    from behaviorlens.video_io import iter_sampled_frames

    video = write_video(tmp_path / "v.mp4", 1.5)
    meta = probe_video(video)
    monkeypatch.setattr(config, "MAX_FRAME_PIXELS", 100)
    with pytest.raises(VideoValidationError, match="4K"):
        list(iter_sampled_frames(video, meta, 5.0))


def test_tampered_default_model_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "pose_landmarker_lite.task"
    fake.write_bytes(b"not the pinned model")
    monkeypatch.setattr(config, "default_model_path", lambda: fake)
    with pytest.raises(pose.ModelIntegrityError, match="pinned checksum"):
        pose.MediaPipePoseEstimator(fake)


def test_dockerfile_keeps_privacy_and_least_privilege_settings():
    """Container: non-root, checksum-verified model, no checkip.amazonaws.com call, loopback publish documented."""
    text = (config.PROJECT_ROOT / "Dockerfile").read_text()
    assert "USER behaviorlens" in text and "scripts/download_model.py" in text
    assert "--logger.hideWelcomeMessage=true" in text  # wildcard address would otherwise trigger the lookup
    assert "--browser.gatherUsageStats=false" in text
    assert "-p 127.0.0.1:8501:8501" in text
    ignore = (config.PROJECT_ROOT / ".dockerignore").read_text().split()
    assert {"data/", "assets/models/*.task", ".venv"} <= set(ignore)
