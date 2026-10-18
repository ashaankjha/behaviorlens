"""SQLite persistence for sessions, analyses, events and review history (FR5).

Large derived artifacts (videos, landmark arrays, overlay previews) live as
files under the data directory and are referenced by paths *relative* to it.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config
from .models import Event, PoseSequence, new_id, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    video_relpath TEXT,
    preview_relpath TEXT,
    overlay_relpath TEXT,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    fps REAL, width INTEGER, height INTEGER, frame_count INTEGER, duration_s REAL,
    status TEXT NOT NULL,
    data_mode TEXT NOT NULL,
    error TEXT,
    ground_truth_json TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    model_version TEXT,
    settings_json TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    landmarks_relpath TEXT,
    data_mode TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    analysis_id TEXT,
    event_type TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    source TEXT NOT NULL,
    review_status TEXT NOT NULL,
    review_note TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    algorithm_version TEXT,
    confidence REAL,
    original_proposal_json TEXT,
    data_mode TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, start_s);
CREATE TABLE IF NOT EXISTS review_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    request_id TEXT UNIQUE,
    at TEXT NOT NULL
);
"""

SESSION_STATUSES = ("ready", "processing", "review_ready", "reviewed", "exported", "failed")


class NotFoundError(KeyError):
    pass


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.data_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "behaviorlens.db"
        with self._conn() as con:
            con.executescript(SCHEMA)

    # -- infrastructure -------------------------------------------------
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def path(self, relpath: str | None) -> Path | None:
        return self.root / relpath if relpath else None

    def relpath(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def session_dir(self, session_id: str) -> Path:
        d = self.root / "sessions" / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- sessions -------------------------------------------------------
    def create_session(self, *, source_filename: str, checksum: str, data_mode: str,
                       meta: dict[str, Any], session_id: str | None = None,
                       video_relpath: str | None = None, preview_relpath: str | None = None,
                       ground_truth: list[dict] | None = None) -> str:
        session_id = session_id or new_id()
        with self._conn() as con:
            con.execute(
                """INSERT INTO sessions (session_id, source_filename, video_relpath, preview_relpath,
                   checksum, created_at, fps, width, height, frame_count, duration_s, status,
                   data_mode, ground_truth_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, source_filename, video_relpath, preview_relpath, checksum, utc_now(),
                 meta.get("fps"), meta.get("width"), meta.get("height"), meta.get("frame_count"),
                 meta.get("duration_s"), "ready", data_mode,
                 json.dumps(ground_truth) if ground_truth is not None else None),
            )
        return session_id

    def update_session(self, session_id: str, **fields: Any) -> None:
        allowed = {"video_relpath", "preview_relpath", "overlay_relpath", "status", "error"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update {bad}")
        if "status" in fields and fields["status"] not in SESSION_STATUSES:
            raise ValueError(f"invalid status {fields['status']}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as con:
            con.execute(f"UPDATE sessions SET {cols} WHERE session_id = ?", (*fields.values(), session_id))

    def set_ground_truth(self, session_id: str, events: list[dict[str, Any]] | None) -> None:
        with self._conn() as con:
            con.execute("UPDATE sessions SET ground_truth_json = ? WHERE session_id = ?",
                        (json.dumps(events) if events is not None else None, session_id))

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise NotFoundError(session_id)
        out = dict(row)
        out["ground_truth"] = json.loads(out.pop("ground_truth_json") or "null")
        return out

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT session_id, source_filename, created_at, status, data_mode, duration_s "
                "FROM sessions ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def sessions_with_checksum(self, checksum: str) -> list[str]:
        with self._conn() as con:
            return [r[0] for r in con.execute(
                "SELECT session_id FROM sessions WHERE checksum = ?", (checksum,))]

    def delete_session(self, session_id: str) -> None:
        """Delete DB records and every local file derived from this session."""
        checksum = self.get_session(session_id)["checksum"]
        with self._conn() as con:
            con.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        shutil.rmtree(self.root / "sessions" / session_id, ignore_errors=True)
        if not self.sessions_with_checksum(checksum):
            for cached in self.cache_dir().glob(f"{checksum}_*.npz"):
                cached.unlink(missing_ok=True)

    def cache_dir(self) -> Path:
        d = self.root / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def mark_exported(self, session_id: str) -> str:
        """Advance a fully reviewed session to 'exported'; sessions with pending events stay as they are."""
        status = self.refresh_review_status(session_id)
        if status == "reviewed":
            self.update_session(session_id, status="exported")
            return "exported"
        return status

    def refresh_review_status(self, session_id: str) -> str:
        """Move a session between review_ready and reviewed based on pending events."""
        session = self.get_session(session_id)
        if session["status"] not in ("review_ready", "reviewed", "exported"):
            return session["status"]
        events = self.list_events(session_id)
        pending = sum(1 for e in events if e.review_status == "pending")
        status = session["status"]
        if pending or not events:
            status = "review_ready"
        elif status == "review_ready":
            status = "reviewed"
        if status != session["status"]:
            self.update_session(session_id, status=status)
        return status

    # -- analyses -------------------------------------------------------
    def add_analysis(self, session_id: str, *, settings: dict, stats: dict, warnings: list[str],
                     model_version: str | None, data_mode: str, pose: PoseSequence | None = None,
                     algorithm_version: str = config.ALGORITHM_VERSION) -> str:
        analysis_id = new_id()
        rel = None
        if pose is not None:
            p = self.session_dir(session_id) / f"landmarks_{analysis_id}.npz"
            pose.save_npz(p)
            rel = self.relpath(p)
        with self._conn() as con:
            con.execute(
                """INSERT INTO analyses (analysis_id, session_id, created_at, algorithm_version,
                   model_version, settings_json, stats_json, warnings_json, landmarks_relpath, data_mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (analysis_id, session_id, utc_now(), algorithm_version, model_version,
                 json.dumps(settings), json.dumps(stats), json.dumps(warnings), rel, data_mode),
            )
        return analysis_id

    def list_analyses(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM analyses WHERE session_id = ? ORDER BY created_at, rowid", (session_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("settings", "stats", "warnings"):
                d[k] = json.loads(d.pop(f"{k}_json"))
            out.append(d)
        return out

    def latest_analysis(self, session_id: str) -> dict[str, Any] | None:
        analyses = self.list_analyses(session_id)
        return analyses[-1] if analyses else None

    def load_landmarks(self, analysis: dict[str, Any]) -> PoseSequence | None:
        p = self.path(analysis.get("landmarks_relpath"))
        return PoseSequence.load_npz(p) if p and p.is_file() else None

    # -- events ---------------------------------------------------------
    def _row_to_event(self, row: sqlite3.Row) -> Event:
        d = dict(row)
        d["evidence"] = json.loads(d.pop("evidence_json") or "{}")
        d["original_proposal"] = json.loads(d.pop("original_proposal_json") or "null")
        return Event(**d)

    def _insert_event(self, con: sqlite3.Connection, e: Event) -> None:
        con.execute(
            """INSERT INTO events (event_id, session_id, analysis_id, event_type, start_s, end_s,
               source, review_status, review_note, evidence_json, algorithm_version, confidence,
               original_proposal_json, data_mode, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (e.event_id, e.session_id, e.analysis_id, e.event_type, e.start_s, e.end_s, e.source,
             e.review_status, e.review_note, json.dumps(e.evidence), e.algorithm_version,
             e.confidence, json.dumps(e.original_proposal) if e.original_proposal else None,
             e.data_mode, e.created_at, e.updated_at),
        )

    def list_events(self, session_id: str, statuses: list[str] | None = None) -> list[Event]:
        q = "SELECT * FROM events WHERE session_id = ?"
        args: list[Any] = [session_id]
        if statuses:
            q += f" AND review_status IN ({','.join('?' * len(statuses))})"
            args += statuses
        q += " ORDER BY start_s, event_type, created_at"
        with self._conn() as con:
            return [self._row_to_event(r) for r in con.execute(q, args)]

    def get_event(self, event_id: str) -> Event:
        with self._conn() as con:
            row = con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise NotFoundError(event_id)
        return self._row_to_event(row)

    def pending_algorithm_count(self, session_id: str) -> int:
        with self._conn() as con:
            return con.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ? AND source = 'algorithm' "
                "AND review_status = 'pending'", (session_id,)).fetchone()[0]

    def save_suggestions(self, session_id: str, analysis_id: str, events: list[Event]) -> dict[str, int]:
        """Store a new analysis's suggestions without touching reviewed work.

        Unreviewed (pending) algorithm suggestions from earlier analyses are
        replaced. Reviewed events (confirmed/rejected/edited) and manual events
        are preserved, and a new suggestion overlapping an already-reviewed
        event of the same type is skipped instead of duplicated.
        """
        with self._conn() as con:
            replaced = con.execute(
                "DELETE FROM events WHERE session_id = ? AND source = 'algorithm' "
                "AND review_status = 'pending'", (session_id,)).rowcount
            reviewed = [self._row_to_event(r) for r in con.execute(
                "SELECT * FROM events WHERE session_id = ?", (session_id,))]
            added = skipped = 0
            for e in events:
                if any(r.event_type == e.event_type and _overlap(r, e) > 0.5 for r in reviewed):
                    skipped += 1
                    continue
                self._insert_event(con, e.model_copy(update={"analysis_id": analysis_id,
                                                              "session_id": session_id}))
                added += 1
        return {"added": added, "replaced_pending": replaced, "skipped_already_reviewed": skipped}

    def _apply_review(self, event_id: str, action: str, request_id: str | None,
                      mutate) -> Event:
        with self._conn() as con:
            if request_id and con.execute(
                    "SELECT 1 FROM review_history WHERE request_id = ?", (request_id,)).fetchone():
                row = con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                return self._row_to_event(row)  # duplicate submit (e.g. Streamlit rerun): no-op
            row = con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:
                raise NotFoundError(event_id)
            before = self._row_to_event(row)
            after = mutate(before.model_copy(deep=True))
            after.updated_at = utc_now()
            con.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
            self._insert_event(con, after)
            con.execute(
                "INSERT INTO review_history (event_id, session_id, action, before_json, after_json, "
                "request_id, at) VALUES (?,?,?,?,?,?,?)",
                (event_id, before.session_id, action, before.model_dump_json(), after.model_dump_json(),
                 request_id, after.updated_at),
            )
        self.refresh_review_status(after.session_id)
        return after

    def set_status(self, event_id: str, status: str, note: str | None = None,
                   request_id: str | None = None) -> Event:
        if status not in ("confirmed", "rejected", "pending"):
            raise ValueError(f"invalid review action status {status}")

        def mutate(e: Event) -> Event:
            e.review_status = status
            if note is not None:
                e.review_note = note
            return e

        return self._apply_review(event_id, status if status != "pending" else "reset", request_id, mutate)

    def edit_event(self, event_id: str, *, event_type: str | None = None, start_s: float | None = None,
                   end_s: float | None = None, note: str | None = None,
                   request_id: str | None = None) -> Event:
        duration = self.get_session(self.get_event(event_id).session_id)["duration_s"] or float("inf")

        def mutate(e: Event) -> Event:
            if e.original_proposal is None and e.source == "algorithm":
                e.original_proposal = {"event_type": e.event_type, "start_s": e.start_s, "end_s": e.end_s}
            new_type = event_type or e.event_type
            if new_type not in config.MANUAL_EVENT_TYPES:
                raise ValueError(f"unknown event type {new_type}")
            updated = Event(**{**e.model_dump(),
                               "event_type": new_type,
                               "start_s": e.start_s if start_s is None else round(float(start_s), 3),
                               "end_s": e.end_s if end_s is None else round(float(end_s), 3),
                               "review_note": e.review_note if note is None else note,
                               "review_status": "edited"})
            updated.check_within(duration)
            return updated

        return self._apply_review(event_id, "edit", request_id, mutate)

    def add_manual_event(self, session_id: str, event_type: str, start_s: float, end_s: float,
                         note: str = "", request_id: str | None = None) -> Event:
        if event_type not in config.MANUAL_EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type}")
        session = self.get_session(session_id)
        e = Event(session_id=session_id, event_type=event_type, start_s=round(float(start_s), 3),
                  end_s=round(float(end_s), 3), source="manual", review_status="confirmed",
                  review_note=note, algorithm_version=None, data_mode=session["data_mode"],
                  evidence={"rule": "manual_annotation"})
        e.check_within(session["duration_s"] or float("inf"))
        with self._conn() as con:
            if request_id and con.execute(
                    "SELECT 1 FROM review_history WHERE request_id = ?", (request_id,)).fetchone():
                return e
            self._insert_event(con, e)
            con.execute(
                "INSERT INTO review_history (event_id, session_id, action, before_json, after_json, "
                "request_id, at) VALUES (?,?,?,?,?,?,?)",
                (e.event_id, session_id, "manual_add", None, e.model_dump_json(), request_id, e.created_at),
            )
        self.refresh_review_status(session_id)
        return e

    def review_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT event_id, action, before_json, after_json, at FROM review_history "
                "WHERE session_id = ? ORDER BY history_id", (session_id,)).fetchall()
        return [dict(r) for r in rows]


def _overlap(a: Event, b: Event) -> float:
    """Intersection over the shorter interval."""
    inter = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    shorter = min(a.end_s - a.start_s, b.end_s - b.start_s)
    return max(0.0, inter) / shorter if shorter > 0 else 0.0
