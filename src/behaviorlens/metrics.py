"""Agreement between suggested events and supplied ground truth (FR7).

Only meaningful when real ground truth exists; the UI hides these numbers
otherwise.
"""

from __future__ import annotations

import csv
import io
import math
from typing import Any

from . import config


def match_events(predicted: list[dict[str, Any]], truth: list[dict[str, Any]],
                 tolerance_s: float = 1.0) -> dict[str, Any]:
    """Greedy one-to-one matching by type with start AND end within ``tolerance_s``."""
    per_type: dict[str, dict[str, Any]] = {}
    types = sorted({e["event_type"] for e in truth} | {e["event_type"] for e in predicted})
    for t in types:
        preds = sorted((p for p in predicted if p["event_type"] == t), key=lambda e: e["start_s"])
        gts = sorted((g for g in truth if g["event_type"] == t), key=lambda e: e["start_s"])
        used: set[int] = set()
        onset_errors = []
        for p in preds:
            best, best_cost = None, None
            for j, g in enumerate(gts):
                if j in used:
                    continue
                ds, de = abs(p["start_s"] - g["start_s"]), abs(p["end_s"] - g["end_s"])
                if ds <= tolerance_s and de <= tolerance_s and (best_cost is None or ds + de < best_cost):
                    best, best_cost = j, ds + de
            if best is not None:
                used.add(best)
                onset_errors.append(abs(p["start_s"] - gts[best]["start_s"]))
        tp = len(used)
        fp, fn = len(preds) - tp, len(gts) - tp
        per_type[t] = _scores(tp, fp, fn) | {
            "mean_onset_error_s": round(sum(onset_errors) / len(onset_errors), 3) if onset_errors else None,
        }
    totals = {k: sum(v[k] for v in per_type.values()) for k in ("tp", "fp", "fn")}
    return {"tolerance_s": tolerance_s, "per_type": per_type,
            "overall": _scores(totals["tp"], totals["fp"], totals["fn"])}


def _scores(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else None)
    def r(v: float | None) -> float | None:
        return None if v is None else round(v, 3)

    return {"tp": tp, "fp": fp, "fn": fn, "precision": r(precision), "recall": r(recall), "f1": r(f1)}


def parse_ground_truth_csv(text: str) -> list[dict[str, Any]]:
    """Parse a CSV with columns event_type,start_s,end_s (extra columns ignored)."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    missing = {"event_type", "start_s", "end_s"} - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"ground-truth CSV is missing columns: {', '.join(sorted(missing))}")
    for i, row in enumerate(reader, start=2):
        if len(rows) >= config.MAX_GROUND_TRUTH_ROWS:
            raise ValueError(f"ground-truth CSV has more than {config.MAX_GROUND_TRUTH_ROWS} rows")
        try:
            start, end = float(row["start_s"]), float(row["end_s"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {i}: start_s/end_s must be numbers") from exc
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"row {i}: start_s/end_s must be finite numbers")
        if not 0 <= start < end:
            raise ValueError(f"row {i}: need 0 <= start_s < end_s")
        event_type = (row["event_type"] or "").strip()
        if not event_type or len(event_type) > 64:
            raise ValueError(f"row {i}: event_type must be 1-64 characters")
        rows.append({"event_type": event_type, "start_s": start, "end_s": end})
    return rows
