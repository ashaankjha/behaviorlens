"""CSV/JSON export of reviewed events (FR6).

Exports never contain video bytes, filesystem paths, source filenames (which
may contain names) or image crops.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from . import config
from .models import utc_now
from .storage import Store

CSV_FIELDS = [
    "session_id", "event_id", "event_type", "start_s", "end_s", "duration_s", "source",
    "review_status", "review_note", "algorithm_version", "created_at", "updated_at",
    "analysis_id", "data_mode", "original_event_type", "original_start_s", "original_end_s",
]


def _event_rows(store: Store, session_id: str, statuses: list[str] | None) -> list[dict[str, Any]]:
    rows = []
    for e in store.list_events(session_id, statuses):
        orig = e.original_proposal or {}
        rows.append({
            "session_id": e.session_id,
            "event_id": e.event_id,
            "event_type": e.event_type,
            "start_s": e.start_s,
            "end_s": e.end_s,
            "duration_s": e.duration_s,
            "source": e.source,
            "review_status": e.review_status,
            "review_note": e.review_note,
            "algorithm_version": e.algorithm_version or "",
            "created_at": e.created_at,
            "updated_at": e.updated_at,
            "analysis_id": e.analysis_id or "",
            "data_mode": e.data_mode,
            "original_event_type": orig.get("event_type", ""),
            "original_start_s": orig.get("start_s", ""),
            "original_end_s": orig.get("end_s", ""),
        })
    return rows


def to_csv(store: Store, session_id: str, statuses: list[str] | None = None) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_event_rows(store, session_id, statuses))
    return buf.getvalue()


def to_json_dict(store: Store, session_id: str, statuses: list[str] | None = None) -> dict[str, Any]:
    s = store.get_session(session_id)
    analysis = store.latest_analysis(session_id)
    events = []
    for e in store.list_events(session_id, statuses):
        d = e.model_dump()
        d["duration_s"] = e.duration_s
        events.append(d)
    return {
        "export_version": 1,
        "exported_at": utc_now(),
        "warning": config.EXPORT_WARNING,
        "data_mode": s["data_mode"],
        "data_mode_note": (
            "SYNTHETIC FIXTURE: landmarks were generated programmatically; no pose inference was run."
            if s["data_mode"] == "synthetic"
            else "Landmarks produced by local MediaPipe pose inference on the uploaded video."
        ),
        "session": {
            "session_id": s["session_id"],
            "content_sha256": s["checksum"],
            "created_at": s["created_at"],
            "fps": s["fps"],
            "width": s["width"],
            "height": s["height"],
            "frame_count": s["frame_count"],
            "duration_s": s["duration_s"],
            "status": s["status"],
        },
        "analysis": None if analysis is None else {
            "analysis_id": analysis["analysis_id"],
            "created_at": analysis["created_at"],
            "algorithm_version": analysis["algorithm_version"],
            "model_version": analysis["model_version"],
            "settings": analysis["settings"],
            "stats": analysis["stats"],
            "warnings": analysis["warnings"],
        },
        "status_filter": statuses or "all",
        "events": events,
    }


def to_json(store: Store, session_id: str, statuses: list[str] | None = None) -> str:
    return json.dumps(to_json_dict(store, session_id, statuses), indent=2)
