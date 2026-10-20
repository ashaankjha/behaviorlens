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


def test_audit_trail_status_is_shown_and_flags_tampering(app, store):
    """The review-history panel reports the hash-chain check, and a failure when data was edited directly."""
    import sqlite3

    app.button[0].click().run()
    next(b for b in app.button if b.label == "Next pending").click().run()
    next(b for b in app.button if b.label == "✓ Confirm").click().run()
    assert any("Audit trail verified: 1 hash-chained record" in c.value for c in app.caption)

    with sqlite3.connect(store.db_path) as con:
        con.execute("UPDATE events SET review_status = 'rejected' WHERE review_status = 'confirmed'")
    app.run()
    assert not app.exception
    assert any("integrity check FAILED" in e.value for e in app.error)
