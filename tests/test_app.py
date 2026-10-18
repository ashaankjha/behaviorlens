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
