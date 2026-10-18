import csv
import io
import json

from behaviorlens import config, export
from behaviorlens.export import CSV_FIELDS
from behaviorlens.models import Event

REQUIRED_CSV = ["session_id", "event_id", "event_type", "start_s", "end_s", "duration_s", "source",
                "review_status", "review_note", "algorithm_version", "created_at", "updated_at"]


def seeded(store):
    sid = store.create_session(source_filename="Jane_Doe_session.mp4", checksum="deadbeef",
                               data_mode="video_inference",
                               meta={"fps": 30, "width": 640, "height": 480, "frame_count": 600, "duration_s": 20})
    video = store.session_dir(sid) / "source.mp4"
    video.write_bytes(b"\x00" * 10)
    store.update_session(sid, video_relpath=store.relpath(video), status="review_ready")
    aid = store.add_analysis(sid, settings={"sample_fps": 5}, stats={"frames": 100}, warnings=["w"],
                             model_version="pose_landmarker_lite@123", data_mode="video_inference")
    store.save_suggestions(sid, aid, [
        Event(session_id=sid, event_type="hand_raised", start_s=1, end_s=2.5),
        Event(session_id=sid, event_type="standing_transition", start_s=4, end_s=5),
        Event(session_id=sid, event_type="hand_raised", start_s=8, end_s=9),
    ])
    a, b, c = store.list_events(sid)
    store.set_status(a.event_id, "confirmed", note="ok, with comma")
    store.set_status(b.event_id, "rejected")
    store.edit_event(c.event_id, start_s=7.5)
    store.add_manual_event(sid, "other_observable", 10, 11, "manual")
    return sid


def test_csv_fields_and_statuses(store):
    sid = seeded(store)
    rows = list(csv.DictReader(io.StringIO(export.to_csv(store, sid))))
    assert set(REQUIRED_CSV) <= set(rows[0]) and list(rows[0]) == CSV_FIELDS
    assert [r["review_status"] for r in rows] == ["confirmed", "rejected", "edited", "confirmed"]
    assert rows[0]["review_note"] == "ok, with comma"
    assert float(rows[0]["duration_s"]) == 1.5
    assert rows[2]["original_start_s"] == "8.0" and rows[2]["start_s"] == "7.5"
    assert rows[3]["source"] == "manual" and rows[3]["algorithm_version"] == ""


def test_status_filter_keeps_rejected_unless_filtered(store):
    sid = seeded(store)
    assert len(list(csv.DictReader(io.StringIO(export.to_csv(store, sid))))) == 4
    only = list(csv.DictReader(io.StringIO(export.to_csv(store, sid, ["confirmed", "edited"]))))
    assert {r["review_status"] for r in only} == {"confirmed", "edited"}


def test_json_structure(store):
    sid = seeded(store)
    data = json.loads(export.to_json(store, sid))
    assert data["warning"] == config.EXPORT_WARNING
    assert data["session"]["session_id"] == sid and data["session"]["duration_s"] == 20
    assert data["analysis"]["settings"] == {"sample_fps": 5}
    assert data["analysis"]["model_version"] == "pose_landmarker_lite@123"
    assert len(data["events"]) == 4
    edited = next(e for e in data["events"] if e["review_status"] == "edited")
    assert edited["original_proposal"]["start_s"] == 8.0
    assert data["data_mode"] == "video_inference"


def test_exports_contain_no_paths_filenames_or_video(store):
    sid = seeded(store)
    for text in (export.to_csv(store, sid), export.to_json(store, sid)):
        assert str(store.root) not in text
        assert "Jane_Doe" not in text
        assert "source.mp4" not in text and "sessions/" not in text


def test_synthetic_sessions_are_labelled(store):
    sid = store.create_session(source_filename="demo", checksum="x", data_mode="synthetic",
                               meta={"duration_s": 10})
    data = export.to_json_dict(store, sid)
    assert data["data_mode"] == "synthetic" and "SYNTHETIC" in data["data_mode_note"]
