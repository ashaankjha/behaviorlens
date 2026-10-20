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


def test_camera_motion_lane_and_details(app, store):
    """Camera motion is shown as a timeline lane and summarized in Analysis details."""
    from behaviorlens.models import Event

    sid = store.create_session(source_filename="pan.mp4", checksum="y", data_mode="video_inference",
                               meta={"duration_s": 10, "fps": 30})
    stats = {"camera_motion_available": True, "camera_moving_ratio": 0.25, "camera_motion_unknown_ratio": 0.05,
             "camera_motion_intervals": [[2.0, 4.4]]}
    warning = "The camera moved (pan, tilt, zoom or shake) in 25% of sampled frames."
    aid = store.add_analysis(sid, settings={}, stats=stats, warnings=[warning], model_version="fake@1",
                             data_mode="video_inference")
    store.save_suggestions(sid, aid, [Event(session_id=sid, event_type="hand_raised", start_s=5.0, end_s=6.0)])
    app.run()
    app.selectbox(key="session_select").set_value(sid).run()
    assert not app.exception
    assert any(w.value == warning for w in app.warning)
    assert any("Camera motion: moving in 25%" in c.value for c in app.caption)
