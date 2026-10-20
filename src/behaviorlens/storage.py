"""SQLite persistence for sessions, analyses, events and review history (FR5).

Large derived artifacts (videos, landmark arrays, overlay previews) live as
files under the data directory and are referenced by paths *relative* to it.

Integrity measures (docs/regulatory/risk_management.md):

- The schema is versioned (``schema_version`` table) and upgraded by ordered,
  transactional migrations. Existing databases are backed up before an upgrade,
  and a database written by a newer release is refused rather than modified.
- WAL journaling, ``synchronous=FULL`` and foreign keys on every connection.
  Writes run in ``BEGIN IMMEDIATE`` transactions so concurrent writers serialize.
- ``review_history`` is append-only (triggers) and hash-chained per session, and
  ``verify_integrity`` recomputes the chain and checks that every reviewed event
  matches its last audit record. The chain is tamper-*evident*, not tamper-proof:
  someone with write access to the file can rebuild it, which is why exports carry
  the chain head so it can be anchored outside the database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from . import config
from .models import Event, PoseSequence, ReviewStatus, new_id, utc_now

log = logging.getLogger(__name__)

# Version 1: the original (pre-versioning) schema. Kept verbatim so legacy databases
# are recognised as version 1 and upgraded by the later migrations.
SCHEMA_V1 = [
    """CREATE TABLE IF NOT EXISTS sessions (
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
)""",
    """CREATE TABLE IF NOT EXISTS analyses (
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
)""",
    """CREATE TABLE IF NOT EXISTS events (
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
)""",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, start_s)",
    """CREATE TABLE IF NOT EXISTS review_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    request_id TEXT UNIQUE,
    at TEXT NOT NULL
)""",
]

# Fields covered by each review_history row hash, in a fixed order.
HISTORY_HASH_FIELDS = ("event_id", "session_id", "action", "before_json", "after_json", "request_id", "at")
GENESIS_HASH = "0" * 64


def history_row_hash(prev_hash: str | None, row: dict[str, Any]) -> str:
    """SHA-256 over the previous row's hash and this row's canonical JSON."""
    payload = json.dumps([row[k] for k in HISTORY_HASH_FIELDS], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(f"{prev_hash or GENESIS_HASH}\n{payload}".encode()).hexdigest()


def _migrate_v2(con: sqlite3.Connection) -> None:
    """Hash-chain review_history, make it append-only, and log session deletions."""
    con.execute("ALTER TABLE review_history ADD COLUMN prev_hash TEXT")
    con.execute("ALTER TABLE review_history ADD COLUMN row_hash TEXT")
    heads: dict[str, str] = {}
    rows = con.execute(f"SELECT history_id, {', '.join(HISTORY_HASH_FIELDS)} FROM review_history "
                       "ORDER BY history_id").fetchall()
    for r in rows:
        prev = heads.get(r["session_id"], GENESIS_HASH)
        h = history_row_hash(prev, dict(r))
        con.execute("UPDATE review_history SET prev_hash = ?, row_hash = ? WHERE history_id = ?",
                    (prev, h, r["history_id"]))
        heads[r["session_id"]] = h
    con.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON review_history(session_id, history_id)")
    con.execute("""CREATE TRIGGER review_history_no_update BEFORE UPDATE ON review_history
        BEGIN SELECT RAISE(ABORT, 'review_history is append-only'); END""")
    # Rows may only disappear through the ON DELETE CASCADE of a deleted session.
    con.execute("""CREATE TRIGGER review_history_no_delete BEFORE DELETE ON review_history
        WHEN EXISTS (SELECT 1 FROM sessions WHERE session_id = OLD.session_id)
        BEGIN SELECT RAISE(ABORT, 'review_history is append-only'); END""")
    con.execute("""CREATE TABLE deletion_log (
        deletion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        deleted_at TEXT NOT NULL,
        history_rows INTEGER NOT NULL,
        history_chain_head TEXT
    )""")


def _migrate_v1(con: sqlite3.Connection) -> None:
    for stmt in SCHEMA_V1:
        con.execute(stmt)


# Ordered, append-only list. Never edit a released migration; add a new one.
MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "baseline schema", _migrate_v1),
    (2, "hash-chained append-only review_history; deletion_log", _migrate_v2),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")

SESSION_STATUSES = ("ready", "processing", "review_ready", "reviewed", "exported", "failed")


class NotFoundError(KeyError):
    pass


class SchemaVersionError(RuntimeError):
    """The database was written by a newer release; refuse to touch it."""


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.data_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "behaviorlens.db"
        self._migrate()

    # -- infrastructure -------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transactions are explicit (see _conn), so migrations can
        # run DDL atomically and writers take the lock before reading the chain head.
        con = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA synchronous = FULL")
        # Overwrite deleted content so Delete session doesn't leave video metadata or notes in free pages.
        con.execute("PRAGMA secure_delete = ON")
        return con

    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")
        finally:
            con.close()

    def schema_version(self) -> int:
        with self._conn() as con:
            return _current_version(con)

    def _migrate(self) -> None:
        con = self._connect()
        try:
            con.execute("PRAGMA journal_mode = WAL")
            current = _current_version(con)
            if current > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"The database uses schema version {current}, but this release only knows up to "
                    f"{SCHEMA_VERSION}. Upgrade BehaviorLens; the database was not modified.")
            if current == SCHEMA_VERSION:
                return
            if current > 0:
                backup = self.root / f"behaviorlens.db.pre-v{SCHEMA_VERSION}-backup"
                with sqlite3.connect(backup) as dst:
                    con.backup(dst)
                dst.close()
                log.info("schema_backup_created", extra={"fields": {"from_version": current}})
            con.execute("BEGIN IMMEDIATE")
            try:
                con.execute("CREATE TABLE IF NOT EXISTS schema_version ("
                            "version INTEGER PRIMARY KEY, description TEXT NOT NULL, applied_at TEXT NOT NULL)")
                for version, description, apply in MIGRATIONS:
                    if version <= current:
                        continue
                    apply(con)
                    con.execute("INSERT INTO schema_version VALUES (?,?,?)", (version, description, utc_now()))
                problems = [tuple(r) for r in con.execute("PRAGMA foreign_key_check")]
                if problems:
                    raise sqlite3.IntegrityError(f"foreign key violations after migration: {problems[:5]}")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            log.info("schema_migrated", extra={"fields": {"from_version": current, "to_version": SCHEMA_VERSION}})
        finally:
            con.close()

    def path(self, relpath: str | None) -> Path | None:
        """Absolute path for a stored relative path; refuses paths that escape the data dir."""
        if not relpath:
            return None
        p = self.root / relpath
        if not p.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("stored path points outside the data directory")
        return p

    def relpath(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    def session_dir(self, session_id: str) -> Path:
        d = self.root / "sessions" / _check_id(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- sessions -------------------------------------------------------
    def create_session(self, *, source_filename: str, checksum: str, data_mode: str,
                       meta: dict[str, Any], session_id: str | None = None,
                       video_relpath: str | None = None, preview_relpath: str | None = None,
                       ground_truth: list[dict[str, Any]] | None = None) -> str:
        session_id = _check_id(session_id or new_id())
        with self._conn(write=True) as con:
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
        cols = ", ".join(f"{k} = ?" for k in fields)  # keys are whitelisted above
        with self._conn(write=True) as con:
            con.execute(f"UPDATE sessions SET {cols} WHERE session_id = ?", (*fields.values(), session_id))

    def set_ground_truth(self, session_id: str, events: list[dict[str, Any]] | None) -> None:
        with self._conn(write=True) as con:
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
        with self._conn(write=True) as con:
            rows, head = _chain_head(con, session_id)
            con.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            # The deletion itself is recorded (IDs and counts only, no content) so a missing
            # session's audit trail is explained rather than silently gone.
            con.execute("INSERT INTO deletion_log (session_id, deleted_at, history_rows, history_chain_head) "
                        "VALUES (?,?,?,?)", (session_id, utc_now(), rows, head))
        con = self._connect()
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # don't keep deleted rows in the WAL file
        finally:
            con.close()
        shutil.rmtree(self.root / "sessions" / _check_id(session_id), ignore_errors=True)
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
        status: str = session["status"]
        if status not in ("review_ready", "reviewed", "exported"):
            return status
        events = self.list_events(session_id)
        pending = sum(1 for e in events if e.review_status == "pending")
        if pending or not events:
            status = "review_ready"
        elif status == "review_ready":
            status = "reviewed"
        if status != session["status"]:
            self.update_session(session_id, status=status)
        return status

    # -- analyses -------------------------------------------------------
    def add_analysis(self, session_id: str, *, settings: dict[str, Any], stats: dict[str, Any], warnings: list[str],
                     model_version: str | None, data_mode: str, pose: PoseSequence | None = None,
                     algorithm_version: str = config.ALGORITHM_VERSION) -> str:
        analysis_id = new_id()
        rel = None
        if pose is not None:
            p = self.session_dir(session_id) / f"landmarks_{analysis_id}.npz"
            pose.save_npz(p)
            rel = self.relpath(p)
        with self._conn(write=True) as con:
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
        return Event(**_event_dict(row))

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
            return int(con.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ? AND source = 'algorithm' "
                "AND review_status = 'pending'", (session_id,)).fetchone()[0])

    def save_suggestions(self, session_id: str, analysis_id: str, events: list[Event]) -> dict[str, int]:
        """Store a new analysis's suggestions without touching reviewed work.

        Unreviewed (pending) algorithm suggestions from earlier analyses are
        replaced. Reviewed events (confirmed/rejected/edited) and manual events
        are preserved, and a new suggestion overlapping an already-reviewed
        event of the same type is skipped instead of duplicated.
        """
        with self._conn(write=True) as con:
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
                      mutate: Callable[[Event], Event]) -> Event:
        with self._conn(write=True) as con:
            if request_id and con.execute(
                    "SELECT 1 FROM review_history WHERE request_id = ?", (request_id,)).fetchone():
                row = con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
                if row is None:
                    raise NotFoundError(event_id)
                return self._row_to_event(row)  # duplicate submit (e.g. Streamlit rerun): no-op
            row = con.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
            if row is None:
                raise NotFoundError(event_id)
            before = self._row_to_event(row)
            after = mutate(before.model_copy(deep=True))
            after.updated_at = utc_now()
            con.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
            self._insert_event(con, after)
            _append_history(con, event_id=event_id, session_id=before.session_id, action=action,
                            before_json=before.model_dump_json(), after_json=after.model_dump_json(),
                            request_id=request_id, at=after.updated_at)
        log.info("review_action", extra={"fields": {"session_id": after.session_id, "event_id": event_id,
                                                    "action": action}})
        self.refresh_review_status(after.session_id)
        return after

    def set_status(self, event_id: str, status: str, note: str | None = None,
                   request_id: str | None = None) -> Event:
        if status not in ("confirmed", "rejected", "pending"):
            raise ValueError(f"invalid review action status {status}")
        new_status = cast(ReviewStatus, status)

        def mutate(e: Event) -> Event:
            e.review_status = new_status
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
        with self._conn(write=True) as con:
            dup = con.execute("SELECT event_id FROM review_history WHERE request_id = ?",
                              (request_id,)).fetchone() if request_id else None
            if dup:
                row = con.execute("SELECT * FROM events WHERE event_id = ?", (dup[0],)).fetchone()
                if row is None:
                    raise NotFoundError(dup[0])
                return self._row_to_event(row)  # replayed submit: return the event already stored
            self._insert_event(con, e)
            _append_history(con, event_id=e.event_id, session_id=session_id, action="manual_add",
                            before_json=None, after_json=e.model_dump_json(), request_id=request_id,
                            at=e.created_at)
        log.info("review_action", extra={"fields": {"session_id": session_id, "event_id": e.event_id,
                                                    "action": "manual_add"}})
        self.refresh_review_status(session_id)
        return e

    def review_history(self, session_id: str) -> list[dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT event_id, action, before_json, after_json, at FROM review_history "
                "WHERE session_id = ? ORDER BY history_id", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def review_chain_head(self, session_id: str) -> tuple[int, str | None]:
        """(number of review_history rows, hash of the last row) for a session."""
        with self._conn() as con:
            return _chain_head(con, session_id)

    def deletion_log(self) -> list[dict[str, Any]]:
        with self._conn() as con:
            return [dict(r) for r in con.execute("SELECT * FROM deletion_log ORDER BY deletion_id")]

    def verify_integrity(self, session_id: str | None = None) -> list[str]:
        """Check the audit trail; returns a list of problems (empty means intact).

        - SQLite ``integrity_check`` and ``foreign_key_check`` pass (whole database).
        - Each session's review_history hash chain recomputes and links up.
        - Every event that has audit records matches its last ``after_json``, and every
          event without one is still an unreviewed algorithm suggestion, so a status or
          time changed directly in the database (bypassing the audit trail) is flagged.

        Truncating the *newest* rows of a chain is only detectable against a chain head
        recorded elsewhere (exports include it).
        """
        problems: list[str] = []
        with self._conn() as con:
            if session_id is None:
                problems += [f"sqlite: {r[0]}" for r in con.execute("PRAGMA integrity_check") if r[0] != "ok"]
                problems += [f"foreign key violation in {r[0]} (rowid {r[1]})"
                             for r in con.execute("PRAGMA foreign_key_check")]
                sessions = [r[0] for r in con.execute("SELECT session_id FROM sessions")]
            else:
                sessions = [session_id]
            for sid in sessions:
                problems += _verify_session(con, sid)
        return problems


def _check_id(value: str) -> str:
    """IDs become directory names, so restrict them to a safe character set."""
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError("invalid identifier")
    return value


def _current_version(con: sqlite3.Connection) -> int:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "schema_version" in tables:
        return int(con.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0])
    # Databases created before schema versioning have the v1 tables but no version table.
    return 1 if "sessions" in tables else 0


def _chain_head(con: sqlite3.Connection, session_id: str) -> tuple[int, str | None]:
    row = con.execute("SELECT COUNT(*), (SELECT row_hash FROM review_history WHERE session_id = ? "
                      "ORDER BY history_id DESC LIMIT 1) FROM review_history WHERE session_id = ?",
                      (session_id, session_id)).fetchone()
    return int(row[0]), row[1]


def _append_history(con: sqlite3.Connection, **row: Any) -> None:
    """Insert one hash-chained review_history row. Call inside a write transaction."""
    _, prev = _chain_head(con, row["session_id"])
    prev = prev or GENESIS_HASH
    con.execute(
        "INSERT INTO review_history (event_id, session_id, action, before_json, after_json, request_id, at, "
        "prev_hash, row_hash) VALUES (?,?,?,?,?,?,?,?,?)",
        (*(row[k] for k in HISTORY_HASH_FIELDS), prev, history_row_hash(prev, row)))


def _verify_session(con: sqlite3.Connection, session_id: str) -> list[str]:
    problems: list[str] = []
    prev = GENESIS_HASH
    last_after: dict[str, str | None] = {}
    for r in con.execute("SELECT * FROM review_history WHERE session_id = ? ORDER BY history_id", (session_id,)):
        row = dict(r)
        where = f"session {session_id[:8]} history row {row['history_id']}"
        if row["prev_hash"] != prev:
            problems.append(f"{where}: chain broken (a row was removed, reordered or inserted)")
        if row["row_hash"] != history_row_hash(row["prev_hash"], row):
            problems.append(f"{where}: content does not match its hash (row modified)")
        prev = row["row_hash"] or ""
        last_after[row["event_id"]] = row["after_json"]
    for r in con.execute("SELECT * FROM events WHERE session_id = ?", (session_id,)):
        current = Event.model_validate(_event_dict(r)).model_dump(mode="json")
        eid = current["event_id"]
        if eid not in last_after:
            if current["source"] != "algorithm" or current["review_status"] != "pending":
                problems.append(f"event {eid[:8]}: reviewed or manual event has no audit record")
            continue
        recorded = json.loads(last_after[eid] or "null") or {}
        # Compare only fields the audit record has, so fields added by later releases don't false-alarm.
        diff = sorted(k for k in recorded if _norm(recorded[k]) != _norm(current.get(k)))
        if diff:
            problems.append(f"event {eid[:8]}: differs from its last audit record ({', '.join(diff)})")
    return problems


def _norm(value: Any) -> Any:
    return None if value in ({}, None) else value


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["evidence"] = json.loads(d.pop("evidence_json") or "{}")
    d["original_proposal"] = json.loads(d.pop("original_proposal_json") or "null")
    return d


def _overlap(a: Event, b: Event) -> float:
    """Intersection over the shorter interval."""
    inter = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    shorter = min(a.end_s - a.start_s, b.end_s - b.start_s)
    return max(0.0, inter) / shorter if shorter > 0 else 0.0
