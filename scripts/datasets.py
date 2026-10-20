"""Shared real-video validation dataset format: loader, schema validation, split and scoring.

Layout (gitignored, never committed):

    data/validation/<dataset>/annotations.json
    data/validation/<dataset>/videos/<video_id>.mp4

``annotations.json`` is a list of records::

    {"video_id": str, "video": "videos/<video_id>.mp4", "fps": float, "duration_s": float,
     "camera": "fixed" | "handheld" | "unknown", "subjects": int | null,
     "events": [{"type": "standing_transition" | "sitting_down" | "hand_raised" | "other",
                 "start_s": float, "end_s": float, "source_label": str}]}

Splits are a deterministic 50/50 sha1 hash of ``f"{dataset}:{video_id}"`` (``split_key``).
The one exception is Charades, whose legacy annotations (``labels`` / ``action_timings``) are
converted on load and keep the bare-``video_id`` key, so the published rules-v0.1 to v0.3
tables reproduce exactly.

Scoring (``score_events``): a detection is correct if it overlaps a ground-truth window of the
same type widened by ``tol`` seconds; recall counts windows with at least one detection.
Wrong detections are attributed to the overlapping "confuser" windows (``sitting_down`` for
``standing_transition``) and to the source labels they overlap, for error analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

EVENT_TYPES = ("standing_transition", "sitting_down", "hand_raised", "other")
CAMERAS = ("fixed", "handheld", "unknown")
RECORD_KEYS = {"video_id", "video", "fps", "duration_s", "camera", "subjects", "events"}
EVENT_KEYS = {"type", "start_s", "end_s", "source_label"}
# Datasets split on the bare video_id (published before the shared format existed).
LEGACY_SPLIT_KEY = {"charades"}
# Charades action classes (dataset card): 154 standing up, 151 standing -> sitting.
CHARADES_MAP = {154: "standing_transition", 151: "sitting_down"}
# Which ground-truth types count as hard negatives for a predicted event type.
CONFUSERS = {"standing_transition": ("sitting_down",), "hand_raised": ()}


class DatasetError(ValueError):
    """annotations.json does not follow the shared format."""


def split_of(key: str) -> str:
    """Deterministic 50/50 split (same function as validate_real_video.split_of)."""
    return "tune" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def split_key(dataset: str, video_id: str) -> str:
    return video_id if dataset in LEGACY_SPLIT_KEY else f"{dataset}:{video_id}"


def _num(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_records(records: object, dataset: str = "?") -> None:
    """Raise DatasetError listing every schema problem (first 20)."""
    errs: list[str] = []
    if not isinstance(records, list):
        raise DatasetError(f"{dataset}: annotations must be a JSON list")
    seen: set[str] = set()
    for i, r in enumerate(records):
        where = f"{dataset}[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{where}: not an object")
            continue
        missing, extra = RECORD_KEYS - set(r), set(r) - RECORD_KEYS
        if missing:
            errs.append(f"{where}: missing {sorted(missing)}")
        if extra:
            errs.append(f"{where}: unknown keys {sorted(extra)}")
        vid = r.get("video_id")
        if not isinstance(vid, str) or not vid:
            errs.append(f"{where}: video_id must be a non-empty string")
        elif vid in seen:
            errs.append(f"{where}: duplicate video_id {vid!r}")
        else:
            seen.add(vid)
            where = f"{dataset}:{vid}"
        video = r.get("video")
        if not isinstance(video, str) or Path(video).is_absolute() or ".." in Path(video).parts \
                or not video.startswith("videos/"):
            errs.append(f"{where}: video must be a relative path under videos/")
        if not (_num(r.get("fps")) and r["fps"] > 0):
            errs.append(f"{where}: fps must be a positive number")
        if not (_num(r.get("duration_s")) and r["duration_s"] > 0):
            errs.append(f"{where}: duration_s must be a positive number")
        if r.get("camera") not in CAMERAS:
            errs.append(f"{where}: camera must be one of {CAMERAS}")
        subj = r.get("subjects")
        if subj is not None and (not isinstance(subj, int) or isinstance(subj, bool) or subj < 0):
            errs.append(f"{where}: subjects must be a non-negative int or null")
        events = r.get("events")
        if not isinstance(events, list):
            errs.append(f"{where}: events must be a list")
            continue
        for j, e in enumerate(events):
            ew = f"{where}.events[{j}]"
            if not isinstance(e, dict) or set(e) != EVENT_KEYS:
                errs.append(f"{ew}: needs exactly {sorted(EVENT_KEYS)}")
                continue
            if e["type"] not in EVENT_TYPES:
                errs.append(f"{ew}: type must be one of {EVENT_TYPES}")
            if not (_num(e["start_s"]) and _num(e["end_s"]) and 0 <= e["start_s"] <= e["end_s"]):
                errs.append(f"{ew}: need 0 <= start_s <= end_s")
            if not isinstance(e["source_label"], str):
                errs.append(f"{ew}: source_label must be a string")
    if errs:
        more = f" (+{len(errs) - 20} more)" if len(errs) > 20 else ""
        raise DatasetError("; ".join(errs[:20]) + more)


def charades_records(legacy: list[dict], root: Path) -> list[dict]:
    """Shared-format records from the legacy Charades sample annotations."""
    import cv2

    out = []
    for a in legacy:
        rel = f"videos/{a['video_id']}.mp4"
        cap = cv2.VideoCapture(str(root / rel))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if not (_num(fps) and fps > 0):
            fps = 30.0  # video not fetched yet; fps is informational only
        events = [{"type": CHARADES_MAP.get(lab, "other"), "start_s": float(s), "end_s": float(e),
                   "source_label": f"charades_c{lab:03d}"}
                  for lab, (s, e) in zip(a["labels"], a["action_timings"])]
        out.append({"video_id": a["video_id"], "video": rel, "fps": fps, "duration_s": float(a["length"]),
                    "camera": "handheld", "subjects": None, "events": events})
    return out


def dataset_dir(name: str, base: Path) -> Path:
    return base / name


def load_dataset(name: str, base: Path) -> list[dict]:
    """Validated records of ``base/<name>/annotations.json`` (legacy Charades converted)."""
    path = dataset_dir(name, base) / "annotations.json"
    if not path.is_file():
        raise DatasetError(f"{name}: {path.name} not found (fetch the dataset first)")
    records = json.loads(path.read_text())
    if name == "charades" and records and "labels" in records[0]:
        records = charades_records(records, dataset_dir(name, base))
    validate_records(records, name)
    return records


def video_path(name: str, record: dict, base: Path) -> Path:
    return dataset_dir(name, base) / record["video"]


def select(records: Iterable[dict], dataset: str, split: str) -> list[dict]:
    return [r for r in records if split == "all" or split_of(split_key(dataset, r["video_id"])) == split]


def event_counts(records: Iterable[dict]) -> dict[str, int]:
    c = Counter(e["type"] for r in records for e in r["events"])
    return {t: c.get(t, 0) for t in EVENT_TYPES}


def windows(record: dict, etype: str) -> list[tuple[float, float]]:
    return [(e["start_s"], e["end_s"]) for e in record["events"] if e["type"] == etype]


def overlaps(a: tuple[float, float], b: tuple[float, float], tol: float = 0.0) -> bool:
    return a[0] < b[1] + tol and b[0] - tol < a[1]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def match_video(record: dict, detections: Sequence[tuple[float, float]], etype: str,
                tol: float) -> dict:
    """Per-video matching of detections of ``etype`` against the record's ground truth."""
    gt = windows(record, etype)
    conf = [w for t in CONFUSERS.get(etype, ()) for w in windows(record, t)]
    found = sum(any(overlaps(d, w, tol) for d in detections) for w in gt)
    hit = [any(overlaps(d, w, tol) for w in gt) for d in detections]
    good = [d for d, h in zip(detections, hit) if h]
    bad = [d for d, h in zip(detections, hit) if not h]
    confused = sum(1 for d in bad if any(overlaps(d, w, tol) for w in conf))
    # Which source labels the wrong detections overlap (strict overlap, no tolerance).
    fp_labels = []
    for d in bad:
        labs = sorted({e["source_label"] for e in record["events"] if overlaps(d, (e["start_s"], e["end_s"]))})
        fp_labels.append(labs or ["(unlabelled)"])
    return {"windows": len(gt), "found": found, "detections": len(detections), "correct": len(good),
            "confused": confused, "fp_labels": fp_labels}


def score_events(rows: Iterable[dict]) -> dict:
    """Aggregate ``match_video`` rows: recall, precision, F1 and Wilson intervals."""
    rows = list(rows)
    w = sum(r["windows"] for r in rows)
    f = sum(r["found"] for r in rows)
    d = sum(r["detections"] for r in rows)
    c = sum(r["correct"] for r in rows)
    recall = f / w if w else None
    precision = c / d if d else None
    f1 = (2 * recall * precision / (recall + precision)
          if recall is not None and precision is not None and recall + precision > 0 else 0.0)
    fp_by_label = Counter(lab for r in rows for labs in r["fp_labels"] for lab in labs)
    return {"videos": len(rows), "windows": w, "found": f, "recall": round(recall, 3) if w else None,
            "recall_ci95": wilson(f, w), "detections": d, "correct": c,
            "precision": round(precision, 3) if d else None, "precision_ci95": wilson(c, d),
            "false_positives": d - c, "f1": round(f1, 3),
            "confused_with_sitting_down": sum(r["confused"] for r in rows),
            "fp_overlapping_labels": dict(fp_by_label.most_common())}
