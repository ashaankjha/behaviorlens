import pytest
from pydantic import ValidationError

from behaviorlens.models import Event
from behaviorlens.storage import NotFoundError, Store


def new_session(store, duration=30.0, mode="video_inference"):
    return store.create_session(source_filename="clip.mp4", checksum="abc", data_mode=mode,
                                meta={"fps": 30, "width": 640, "height": 480, "frame_count": 900,
                                      "duration_s": duration})


def suggest(store, sid, *spans, etype="hand_raised"):
    aid = store.add_analysis(sid, settings={}, stats={}, warnings=[], model_version="m", data_mode="video_inference")
    evs = [Event(session_id=sid, event_type=etype, start_s=a, end_s=b, evidence={"rule": "r"}) for a, b in spans]
    counts = store.save_suggestions(sid, aid, evs)
    store.update_session(sid, status="review_ready")
    return aid, counts


def test_session_roundtrip(store):
    sid = new_session(store)
    s = store.get_session(sid)
    assert s["status"] == "ready" and s["duration_s"] == 30.0
    assert [x["session_id"] for x in store.list_sessions()] == [sid]


def test_review_actions_persist_across_restart(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2), (5, 6), (8, 9))
    e1, e2, e3 = store.list_events(sid)
    store.set_status(e1.event_id, "confirmed", note="clear")
    store.set_status(e2.event_id, "rejected")
    store.edit_event(e3.event_id, start_s=7.5, end_s=9.5, note="adjusted")
    manual = store.add_manual_event(sid, "other_observable", 12, 13, "walked to door")

    reopened = Store(store.root)  # simulates an app restart
    got = {e.event_id: e for e in reopened.list_events(sid)}
    assert got[e1.event_id].review_status == "confirmed" and got[e1.event_id].review_note == "clear"
    assert got[e2.event_id].review_status == "rejected"
    assert got[e3.event_id].review_status == "edited"
    assert (got[e3.event_id].start_s, got[e3.event_id].end_s) == (7.5, 9.5)
    assert got[e3.event_id].original_proposal == {"event_type": "hand_raised", "start_s": 8.0, "end_s": 9.0}
    assert got[manual.event_id].source == "manual" and got[manual.event_id].review_status == "confirmed"
    assert reopened.get_session(sid)["status"] == "reviewed"
    assert [h["action"] for h in reopened.review_history(sid)] == ["confirmed", "rejected", "edit", "manual_add"]


def test_original_proposal_kept_after_second_edit(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2))
    (e,) = store.list_events(sid)
    store.edit_event(e.event_id, start_s=1.5)
    store.edit_event(e.event_id, end_s=3.0, event_type="other_observable")
    got = store.get_event(e.event_id)
    assert got.original_proposal == {"event_type": "hand_raised", "start_s": 1.0, "end_s": 2.0}
    assert got.event_id == e.event_id  # IDs are stable across edits


def test_duplicate_request_ids_are_ignored(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2))
    (e,) = store.list_events(sid)
    store.set_status(e.event_id, "confirmed", request_id="req-1")
    store.set_status(e.event_id, "rejected", request_id="req-1")  # replayed rerun
    assert store.get_event(e.event_id).review_status == "confirmed"
    store.add_manual_event(sid, "hand_raised", 3, 4, request_id="req-2")
    store.add_manual_event(sid, "hand_raised", 3, 4, request_id="req-2")
    assert sum(ev.source == "manual" for ev in store.list_events(sid)) == 1
    assert len(store.review_history(sid)) == 2


def test_rerun_replaces_only_pending_suggestions(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2), (5, 6), (8, 9))
    e1, e2, _ = store.list_events(sid)
    store.set_status(e1.event_id, "confirmed")
    store.set_status(e2.event_id, "rejected")
    store.add_manual_event(sid, "hand_raised", 20, 21)

    _, counts = suggest(store, sid, (1.1, 2.1), (5, 6), (14, 15))
    assert counts == {"added": 1, "replaced_pending": 1, "skipped_already_reviewed": 2}
    events = store.list_events(sid)
    statuses = sorted((e.start_s, e.review_status, e.source) for e in events)
    assert statuses == [(1.0, "confirmed", "algorithm"), (5.0, "rejected", "algorithm"),
                        (14.0, "pending", "algorithm"), (20.0, "confirmed", "manual")]
    assert len(store.list_analyses(sid)) == 2


def test_event_validation(store):
    sid = new_session(store, duration=10)
    with pytest.raises(ValidationError):
        Event(session_id=sid, event_type="hand_raised", start_s=3, end_s=3)
    with pytest.raises(ValidationError):
        Event(session_id=sid, event_type="hand_raised", start_s=-1, end_s=3)
    with pytest.raises(ValueError):
        store.add_manual_event(sid, "hand_raised", 5, 30)  # beyond duration
    with pytest.raises(ValueError):
        store.add_manual_event(sid, "not_a_type", 1, 2)
    suggest(store, sid, (1, 2))
    (e,) = store.list_events(sid)
    with pytest.raises(ValueError):
        store.edit_event(e.event_id, start_s=4, end_s=2)
    assert store.get_event(e.event_id).review_status == "pending"  # failed edit changes nothing


def test_delete_session_removes_records_and_files(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2))
    f = store.session_dir(sid) / "source.mp4"
    f.write_bytes(b"x")
    cache = store.cache_dir() / "abc_model_5fps.npz"
    cache.write_bytes(b"x")
    store.delete_session(sid)
    with pytest.raises(NotFoundError):
        store.get_session(sid)
    assert store.list_events(sid) == [] and store.review_history(sid) == []
    assert not f.exists() and not cache.exists()


def test_status_moves_back_to_review_ready_when_pending_exists(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2), (3, 4))
    a, b = store.list_events(sid)
    store.set_status(a.event_id, "confirmed")
    assert store.get_session(sid)["status"] == "review_ready"
    store.set_status(b.event_id, "rejected")
    assert store.get_session(sid)["status"] == "reviewed"


def test_export_status_only_after_full_review(store):
    sid = new_session(store)
    suggest(store, sid, (1, 2))
    (e,) = store.list_events(sid)
    assert store.mark_exported(sid) == "review_ready"
    store.set_status(e.event_id, "confirmed")
    assert store.mark_exported(sid) == "exported"
    assert store.get_session(sid)["status"] == "exported"
