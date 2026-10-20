"""Smoke test of the Streamlit UI using Streamlit's headless AppTest harness."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(store):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_with_empty_state(app):
    assert not app.exception
    assert app.title[0].value == "BehaviorLens"
    assert any("Not for clinical decision-making" in w.value for w in app.warning)


def test_demo_review_flow(app, store):
    app.button[0].click().run()  # "Load synthetic demo session"
    assert not app.exception
    (session,) = store.list_sessions()
    assert session["data_mode"] == "synthetic"
    assert any("SYNTHETIC FIXTURE" in i.value for i in app.info)
    events = store.list_events(session["session_id"])
    assert len(events) >= 5

    # select first pending event via "Next pending", then confirm it
    next(b for b in app.button if b.label == "Next pending").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "✓ Confirm").click().run()
    assert not app.exception
    statuses = [e.review_status for e in store.list_events(session["session_id"])]
    assert statuses.count("confirmed") == 1


def test_server_binds_localhost_only():
    """AUDIT D1: without server.address Streamlit binds 0.0.0.0 and calls checkip.amazonaws.com."""
    import tomllib

    cfg = tomllib.loads((Path(APP).parent / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["address"] == "localhost"
    assert cfg["browser"]["gatherUsageStats"] is False


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_rerun_requires_explicit_confirmation(app, store):
    """FR5: re-running analysis with unreviewed suggestions needs a ticked confirmation box."""
    app.button[0].click().run()  # load the demo (analysis runs once)
    assert store.pending_algorithm_count(store.list_sessions()[0]["session_id"]) > 0
    assert _button(app, "Re-run analysis").disabled
    box = next(c for c in app.checkbox if c.label.startswith("Replace"))
    box.check().run()
    assert not _button(app, "Re-run analysis").disabled


def test_ground_truth_metrics_only_when_ground_truth_exists(app, store):
    """FR7: precision/recall are hidden unless ground truth was supplied."""
    app.button[0].click().run()  # demo session carries scripted ground truth
    assert any("Algorithm proposals" in m.value for m in app.markdown)
    sid = store.create_session(source_filename="no_truth.mp4", checksum="x", data_mode="video_inference",
                               meta={"duration_s": 10, "fps": 30})
    app.run()  # refresh the selectbox options so the new session is listed
    app.selectbox(key="session_select").set_value(sid).run()
    assert not app.exception
    assert any("No ground truth supplied" in c.value for c in app.caption)
    assert not any("Algorithm proposals" in m.value for m in app.markdown)


def test_reviewer_can_pick_the_subject_track(app, store, tmp_path, monkeypatch):
    """Multiple tracked people: a subject selectbox with unique labels (M2) re-runs on that track."""
    import io

    import numpy as np

    from behaviorlens import pipeline
    from conftest import make_multi, person, write_video

    multi = make_multi(6, people=lambda t: [person(0.3, 1.0, stand=0.0),
                                            person(0.75, 0.5, stand=1.0 if t >= 3 else 0.0)])

    class Fake:
        model_version = "fakemulti@1"

        def __init__(self, **kwargs):
            pass

        def detect_all(self, frame, t):
            return multi.detections(int(np.argmin(np.abs(multi.timestamps_s - t))))

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "MediaPipePoseEstimator", Fake)
    data = write_video(tmp_path / "two.mp4", 6.0).read_bytes()
    sid = pipeline.create_upload_session(store, "two.mp4", io.BytesIO(data), len(data))
    pipeline.run_analysis(store, sid)
    auto = store.latest_analysis(sid)["stats"]["tracking"]["subject_track_id"]
    app.run()
    app.selectbox(key="session_select").set_value(sid).run()
    subject = app.selectbox(key=f"subject_{sid}")
    labels = list(subject.options)  # AppTest exposes the formatted labels
    assert len(labels) == 3 and len(set(labels)) == 3  # automatic + two tracks, all distinct
    assert labels[0].startswith(f"Automatic: track #{auto}")
    other = next(t for t in (1, 2) if t != auto)
    subject.set_value(other).run()
    if store.pending_algorithm_count(sid):
        next(c for c in app.checkbox if c.label.startswith("Replace")).check().run()
    _button(app, "Re-run analysis").click().run()
    assert not app.exception
    t = store.latest_analysis(sid)["stats"]["tracking"]
    assert t["subject_track_id"] == other and t["subject_selection"] == "reviewer"
    assert any("standing_transition" == e.event_type for e in store.list_events(sid))
