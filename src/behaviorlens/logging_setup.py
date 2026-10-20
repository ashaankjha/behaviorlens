"""Structured (JSON-lines) logging that never carries PHI, frames or filesystem paths.

Log records are event names plus a small dict of non-identifying fields (IDs,
counts, durations, exception *types*). Call sites pass fields through
``extra={"fields": {...}}``. As defence in depth, a filter redacts anything that
looks like an absolute path or the configured data directory from messages and
string fields, and drops binary or array values outright.

Logs go to stderr only (the Streamlit console); nothing is written to disk or sent
anywhere by this module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import IO, Any

from . import config

LOGGER_NAME = "behaviorlens"
REDACTED = "<redacted>"
# POSIX absolute paths, home-relative paths and Windows drive paths.
_PATH_RE = re.compile(r"(?:(?<![\w.])~?/[^\s'\"]+|\b[A-Za-z]:\\[^\s'\"]+)")
_MAX_FIELD_LEN = 300


def redact(text: str) -> str:
    for root in {str(config.data_dir()), str(config.data_dir().resolve()), str(config.PROJECT_ROOT)}:
        text = text.replace(root, REDACTED)
    return _PATH_RE.sub(REDACTED, text)[:_MAX_FIELD_LEN]


def _clean(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and len(value) <= 50:
        return [_clean(v) for v in value]
    return f"<{type(value).__name__} omitted>"  # bytes, arrays, frames, paths, large lists


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        fields = getattr(record, "fields", None)
        cleaned: dict[str, Any] = _clean(fields) if isinstance(fields, dict) else {}
        # Tracebacks contain absolute paths; keep only the exception type.
        if record.exc_info and record.exc_info[0] is not None:
            cleaned["exc_type"] = record.exc_info[0].__name__
        record.fields = cleaned
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        out.update(getattr(record, "fields", {}) or {})
        return json.dumps(out, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None, stream: IO[str] | None = None) -> logging.Logger:
    """Attach one JSON handler to the ``behaviorlens`` logger (idempotent).

    Level comes from ``level``, else ``BEHAVIORLENS_LOG_LEVEL``, else INFO.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for h in list(logger.handlers):
        if getattr(h, "_behaviorlens", False):
            logger.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactingFilter())
    handler._behaviorlens = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.setLevel((level or os.environ.get("BEHAVIORLENS_LOG_LEVEL") or "INFO").upper())
    logger.propagate = False
    return logger
