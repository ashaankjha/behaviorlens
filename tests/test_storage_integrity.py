"""Schema versioning, migrations and the tamper-evident review history (production readiness)."""

import json
import sqlite3

import pytest

from behaviorlens import export
from behaviorlens.models import Event, new_id
from behaviorlens.storage import SCHEMA_V1, SCHEMA_VERSION, SchemaVersionError, Store, history_row_hash


def new_session(store, duration=30.0):
    return store.create_session(source_filename="clip.mp4", checksum="abc", data_mode="video_inference",
                                meta={"fps": 30, "width": 640, "height": 480, "frame_count": 900,
                                      "duration_s": duration})


def reviewed_session(store):
    sid = new_session(store)
    aid = store.add_analysis(sid, settings={}, stats={}, warnings=[], model_version="m", data_mode="video_inference")
    store.save_suggestions(sid, aid, [Event(session_id=sid, event_type="hand_raised", start_s=a, end_s=a + 1)
                                      for a in (1, 4, 8)])
    store.update_session(sid, status="review_ready")
    e1, e2, e3 = store.list_events(sid)
    store.set_status(e1.event_id, "confirmed", note="ok")
    store.set_status(e2.event_id, "rejected")
    store.edit_event(e3.event_id, start_s=7.5)
    store.add_manual_event(sid, "other_observable", 12, 13, "manual")
    return sid


def raw(store):
    con = sqlite3.connect(store.db_path)
    con.row_factory = sqlite3.Row
    return con


def test_new_database_is_at_latest_schema_with_wal_and_foreign_keys(store):
    assert store.schema_version() == SCHEMA_VERSION
    with raw(store) as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        versions = [r[0] for r in con.execute("SELECT version FROM schema_version ORDER BY version")]
    assert versions == list(range(1, SCHEMA_VERSION + 1))
    with store._conn() as con:
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_legacy_v1_database_is_migrated_without_data_loss(tmp_path):
    """A database created before schema versioning (v1) upgrades in place, keeps its data,
    gets a backup, and its existing review history becomes a valid hash chain."""
    root = tmp_path / "data"
    root.mkdir()
    sid, eid = new_id(), new_id()
    con = sqlite3.connect(root / "behaviorlens.db")
    for stmt in SCHEMA_V1:
        con.execute(stmt)
    con.execute("INSERT INTO sessions (session_id, source_filename, checksum, created_at, duration_s, status, "
                "data_mode) VALUES (?,?,?,?,?,?,?)", (sid, "old.mp4", "c", "2026-10-01T00:00:00+00:00", 20.0,
                                                       "reviewed", "video_inference"))
    after = Event(event_id=eid, session_id=sid, event_type="hand_raised", start_s=1, end_s=2,
                  review_status="confirmed", created_at="2026-10-01T00:00:00+00:00",
                  updated_at="2026-10-01T00:01:00+00:00")
    before = after.model_copy(update={"review_status": "pending"})
    con.execute("INSERT INTO events (event_id, session_id, event_type, start_s, end_s, source, review_status, "
                "evidence_json, algorithm_version, data_mode, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (eid, sid, "hand_raised", 1.0, 2.0, "algorithm", "confirmed", "{}", after.algorithm_version,
                 "video_inference", after.created_at, after.updated_at))
    con.execute("INSERT INTO review_history (event_id, session_id, action, before_json, after_json, request_id, at) "
                "VALUES (?,?,?,?,?,?,?)", (eid, sid, "confirmed", before.model_dump_json(),
                                           after.model_dump_json(), "r1", after.updated_at))
    con.commit()
    con.close()

    store = Store(root)
    assert store.schema_version() == SCHEMA_VERSION
    assert (root / f"behaviorlens.db.pre-v{SCHEMA_VERSION}-backup").is_file()
    assert store.get_event(eid).review_status == "confirmed"
    assert [h["action"] for h in store.review_history(sid)] == ["confirmed"]
    assert store.verify_integrity() == []
    # New actions extend the migrated chain.
    store.set_status(eid, "rejected")
    assert store.verify_integrity() == [] and store.review_chain_head(sid)[0] == 2
    # Re-opening is a no-op (no second migration, no error).
    assert Store(root).schema_version() == SCHEMA_VERSION


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    from behaviorlens import storage

    root = tmp_path / "data"
    root.mkdir()
    con = sqlite3.connect(root / "behaviorlens.db")
    for stmt in SCHEMA_V1:
        con.execute(stmt)
    con.commit()
    con.close()

    def broken(con):
        con.execute("ALTER TABLE review_history ADD COLUMN prev_hash TEXT")
        raise RuntimeError("simulated crash mid-migration")

    monkeypatch.setattr(storage, "MIGRATIONS", [storage.MIGRATIONS[0], (2, "broken", broken)])
    with pytest.raises(RuntimeError, match="simulated"):
        Store(root)
    with sqlite3.connect(root / "behaviorlens.db") as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(review_history)")]
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "prev_hash" not in cols and "schema_version" not in tables  # untouched v1 database
    monkeypatch.undo()
    assert Store(root).schema_version() == SCHEMA_VERSION


def test_database_from_newer_release_is_refused(store):
    with raw(store) as con:
        con.execute("INSERT INTO schema_version VALUES (?,?,?)", (SCHEMA_VERSION + 1, "future", "x"))
    with pytest.raises(SchemaVersionError, match="Upgrade BehaviorLens"):
        Store(store.root)


def test_review_history_is_hash_chained_and_verifies(store):
    sid = reviewed_session(store)
    assert store.verify_integrity() == [] and store.verify_integrity(sid) == []
    with raw(store) as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM review_history ORDER BY history_id")]
    assert len(rows) == 4 and rows[0]["prev_hash"] == "0" * 64
    for prev, row in zip(rows, rows[1:]):
        assert row["prev_hash"] == prev["row_hash"]
    assert all(r["row_hash"] == history_row_hash(r["prev_hash"], r) for r in rows)
    assert store.review_chain_head(sid) == (4, rows[-1]["row_hash"])


def test_review_history_is_append_only(store):
    sid = reviewed_session(store)
    with raw(store) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE review_history SET action = 'confirmed' WHERE action = 'rejected'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("DELETE FROM review_history WHERE session_id = ?", (sid,))


def _drop_triggers(con):
    con.execute("DROP TRIGGER review_history_no_update")
    con.execute("DROP TRIGGER review_history_no_delete")


def test_tampering_with_history_is_detected(store):
    sid = reviewed_session(store)
    with raw(store) as con:
        _drop_triggers(con)  # an attacker with file access can do this too; the chain still catches it
        con.execute("UPDATE review_history SET after_json = replace(after_json, 'rejected', 'confirmed') "
                    "WHERE action = 'rejected'")
    problems = store.verify_integrity(sid)
    assert any("content does not match its hash" in p for p in problems)

    with raw(store) as con:
        con.execute("DELETE FROM review_history WHERE action = 'confirmed'")
    assert any("chain broken" in p for p in store.verify_integrity(sid))


def test_editing_events_outside_the_audit_trail_is_detected(store):
    sid = reviewed_session(store)
    with raw(store) as con:
        con.execute("UPDATE events SET review_status = 'confirmed' WHERE review_status = 'rejected'")
    assert any("differs from its last audit record (review_status" in p for p in store.verify_integrity(sid))


def test_reviewed_event_without_audit_record_is_detected(store):
    sid = new_session(store)
    e = Event(session_id=sid, event_type="hand_raised", start_s=1, end_s=2, review_status="confirmed")
    with store._conn(write=True) as con:
        store._insert_event(con, e)
    assert any("no audit record" in p for p in store.verify_integrity(sid))


def test_delete_session_is_logged_and_keeps_chain_valid(store):
    keep, gone = reviewed_session(store), reviewed_session(store)
    rows, head = store.review_chain_head(gone)
    store.delete_session(gone)
    assert store.review_history(gone) == []
    (entry,) = store.deletion_log()
    assert entry["session_id"] == gone and entry["history_rows"] == rows and entry["history_chain_head"] == head
    assert store.verify_integrity() == [] and store.review_chain_head(keep)[0] == 4


def test_json_export_carries_chain_head(store):
    sid = reviewed_session(store)
    audit = json.loads(export.to_json(store, sid))["audit"]
    assert audit == {"review_history_rows": 4, "review_history_chain_head_sha256": store.review_chain_head(sid)[1],
                     "integrity_problems": 0}


def test_stored_paths_cannot_escape_the_data_dir(store):
    sid = new_session(store)
    store.update_session(sid, video_relpath="../../etc/passwd")
    with pytest.raises(ValueError, match="outside the data directory"):
        store.path(store.get_session(sid)["video_relpath"])
    with pytest.raises(ValueError, match="invalid identifier"):
        store.session_dir("../escape")
    with pytest.raises(ValueError, match="invalid identifier"):
        store.create_session(source_filename="x.mp4", checksum="c", data_mode="synthetic", meta={},
                             session_id="a/b")


def test_concurrent_duplicate_request_is_applied_once(store):
    """Two Store instances (e.g. two browser tabs) replaying the same request_id."""
    sid = reviewed_session(store)
    other = Store(store.root)
    e = store.list_events(sid)[1]
    store.set_status(e.event_id, "confirmed", request_id="same")
    other.set_status(e.event_id, "pending", request_id="same")
    assert store.get_event(e.event_id).review_status == "confirmed"
    assert store.verify_integrity() == []


def test_replayed_manual_add_returns_the_stored_event(store):
    sid = new_session(store)
    first = store.add_manual_event(sid, "hand_raised", 1, 2, request_id="m1")
    again = store.add_manual_event(sid, "hand_raised", 1, 2, request_id="m1")
    assert again.event_id == first.event_id and len(store.list_events(sid)) == 1
