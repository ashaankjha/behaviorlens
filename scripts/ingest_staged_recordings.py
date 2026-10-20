"""Ingest staged recordings (consenting ADULT volunteers) into a validation dataset.

Takes a folder of videos plus annotation files, validates them, converts each video
to an audio-free H.264 MP4 with an opaque ID, and writes the shared dataset format
that the real-video evaluator reads:

    data/validation/staged/
        annotations.json      # list of {"video_id", "video", "fps", "duration_s", "camera",
                              #          "subjects", "events": [{"type", "start_s", "end_s",
                              #          "source_label"}]}
        videos/<video_id>.mp4
        manifest.json         # local only: participant code, session, angle... per video_id
        ingest_report.json    # inter-rater agreement and disagreements for adjudication
        .id_salt              # secret salt for opaque IDs (stable across re-ingests)

See docs/recording_kit/README.md for the workflow and the source-folder layout.

Usage:
    python scripts/ingest_staged_recordings.py SOURCE_DIR [--check] [--camera fixed]
                                               [--iou 0.5] [--strict] [--allow-other-labels]
    python scripts/ingest_staged_recordings.py --delete [--yes]
    python scripts/ingest_staged_recordings.py --delete-video VIDEO_ID [--yes]
    python scripts/ingest_staged_recordings.py --delete-participant CODE [--yes]
    python scripts/ingest_staged_recordings.py --synthetic-example DIR

Source folder: one video per recording (.mp4/.mov/.avi). Annotation files share the
video's file stem: ``<stem>.csv`` or ``<stem>.<anything>.csv`` (columns
``event_type,start_s,end_s,annotator,notes``) and/or ``<stem>.eaf`` /
``<stem>.<anything>.eaf`` (ELAN). An optional ``manifest.csv`` gives per-video
metadata (``video,participant,camera,subjects,session,camera_angle_deg,distance_m,notes``).

Privacy: original filenames never reach the dataset, the logs or the report. Video IDs
are an HMAC of the content hash under a local secret salt. Audio, subtitles, chapters
and container metadata (phone GPS location, device model, creation time) are stripped.
The console names your source files so you can fix errors; the JSON log stream carries
IDs and counts only (CLAUDE.md convention).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402
from behaviorlens.logging_setup import configure_logging  # noqa: E402
from behaviorlens.video_io import VideoValidationError, ffmpeg_exe, probe_video  # noqa: E402

log = logging.getLogger("behaviorlens.recording_kit")

KIT_VERSION = "recording-kit-v1"
SCHEMA_TYPES = ("standing_transition", "sitting_down", "hand_raised", "other")
# Labels from docs/recording_kit/annotation_guide.md that are kept as type "other" (distractors
# and challenge conditions). Their source_label keeps the exact label for per-distractor analysis.
OTHER_LABELS = (
    "hand_flick", "reach_point", "stretch", "partial_stand", "lunge", "wave",
    "occlusion", "second_person", "camera_pan", "out_of_frame", "other",
)
LABEL_ALIASES = {
    "standing_up": "standing_transition", "stand_up": "standing_transition", "sit_to_stand": "standing_transition",
    "sit_down": "sitting_down", "stand_to_sit": "sitting_down",
    "hand_raise": "hand_raised", "raise_hand": "hand_raised",
}
CAMERA_VALUES = ("fixed", "handheld", "unknown")
VIDEO_EXTENSIONS = config.ALLOWED_EXTENSIONS
CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
VIDEO_ID_RE = re.compile(r"^stg_[0-9a-f]{16}$")
ADJUDICATOR = "adjudicator"
MAX_ROWS = config.MAX_GROUND_TRUTH_ROWS
MAX_ANNOTATION_BYTES = 5 * 1024 * 1024
MANIFEST_COLUMNS = ("video", "participant", "camera", "subjects", "session", "camera_angle_deg",
                    "distance_m", "notes")
# Files the delete commands may remove. Anything else found in the dataset folder is left alone.
DATASET_FILES = ("annotations.json", "manifest.json", "ingest_report.json", ".id_salt")


class IngestError(ValueError):
    """A problem with the source folder, with a message telling the user how to fix it."""


@dataclass(frozen=True)
class Annotation:
    annotator: str
    label: str  # normalised source label
    type: str  # one of SCHEMA_TYPES
    start_s: float
    end_s: float
    origin: str  # "<file>:<row>" for error messages (console only, never logged)


@dataclass
class VideoItem:
    source: Path
    annotations: list[Annotation] = field(default_factory=list)
    # Everyone who annotated this video, including annotators who marked no events.
    annotators: set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------- paths


def dataset_dir() -> Path:
    return config.data_dir() / "validation" / "staged"


def _safe_dataset_dir() -> Path:
    """The dataset folder, refusing anything that is not a real directory under data/validation."""
    root = (config.data_dir() / "validation").resolve()
    d = dataset_dir()
    if d.is_symlink():
        raise IngestError(f"{d} is a symlink; refusing to touch it. Remove the link by hand.")
    if d.exists() and d.resolve().parent != root:
        raise IngestError(f"{d} does not resolve inside {root}; refusing to touch it.")
    return d


# ---------------------------------------------------------------------------- labels


def normalise_label(raw: str, allow_other: bool, origin: str) -> tuple[str, str]:
    """Return (source_label, schema type) for a raw annotation label, or raise with a hint."""
    label = re.sub(r"[\s\-]+", "_", (raw or "").strip().lower())
    if not label:
        raise IngestError(f"{origin}: empty event_type")
    if len(label) > 64:
        raise IngestError(f"{origin}: event_type is longer than 64 characters")
    label = LABEL_ALIASES.get(label, label)
    if label in SCHEMA_TYPES and label != "other":
        return label, label
    if label in OTHER_LABELS:
        return label, "other"
    if allow_other and re.fullmatch(r"[a-z0-9_]+", label):
        return label, "other"
    known = [t for t in SCHEMA_TYPES if t != "other"] + list(OTHER_LABELS)
    close = difflib.get_close_matches(label, known + list(LABEL_ALIASES), n=1)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    raise IngestError(
        f"{origin}: unknown event_type '{raw.strip()}'.{hint} Allowed: {', '.join(known)} "
        "(see docs/recording_kit/annotation_guide.md), or pass --allow-other-labels to keep "
        "unlisted labels as type 'other'."
    )


def _annotator(raw: str, origin: str) -> str:
    name = (raw or "").strip()
    if not CODE_RE.fullmatch(name):
        raise IngestError(
            f"{origin}: annotator '{name}' must be a short code such as initials or 'A'/'B' "
            "(letters, digits, '.', '_' or '-', at most 32 characters). Don't use full names."
        )
    return name


def _time(raw: str | None, origin: str, column: str) -> float:
    try:
        value = float((raw or "").strip())
    except ValueError as exc:
        raise IngestError(f"{origin}: {column} must be a number of seconds, got '{raw}'") from exc
    if not math.isfinite(value):
        raise IngestError(f"{origin}: {column} must be a finite number")
    return value


def _check_interval(start: float, end: float, origin: str) -> None:
    if start < 0 or end <= start:
        raise IngestError(f"{origin}: need 0 <= start_s < end_s (got {start:g}, {end:g})")


# ---------------------------------------------------------------------------- parsers


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_ANNOTATION_BYTES:
        raise IngestError(f"{path.name}: annotation file is larger than {MAX_ANNOTATION_BYTES // 2**20} MB")
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError(f"{path.name}: not UTF-8 text. Save the spreadsheet as 'CSV UTF-8'.") from exc


def parse_annotation_csv(path: Path, allow_other: bool = False,
                         default_annotator: str | None = None) -> tuple[list[Annotation], set[str]]:
    """Parse ``event_type,start_s,end_s,annotator,notes`` (notes optional, extra columns ignored).

    ``default_annotator`` (from a ``<stem>.<annotator>.csv`` file name) fills an absent or empty
    annotator column, so a header-only file records that this annotator saw no events.
    Returns (annotations, annotators).
    """
    text = _read_text(path)
    sample = text[:4096]
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    reader.fieldnames = fields
    required = {"event_type", "start_s", "end_s"} | (set() if default_annotator else {"annotator"})
    missing = required - set(fields)
    if missing:
        raise IngestError(
            f"{path.name}: missing column(s) {', '.join(sorted(missing))}. "
            "Expected header: event_type,start_s,end_s,annotator,notes"
        )
    out: list[Annotation] = []
    for i, row in enumerate(reader, start=2):
        if not any((v or "").strip() for v in row.values() if isinstance(v, str)):
            continue  # blank line
        if len(out) >= MAX_ROWS:
            raise IngestError(f"{path.name}: more than {MAX_ROWS} rows")
        origin = f"{path.name} row {i}"
        label, etype = normalise_label(row.get("event_type") or "", allow_other, origin)
        start, end = _time(row.get("start_s"), origin, "start_s"), _time(row.get("end_s"), origin, "end_s")
        _check_interval(start, end, origin)
        who = _annotator(row.get("annotator") or default_annotator or "", origin)
        out.append(Annotation(who, label, etype, start, end, origin))
    annotators = {a.annotator for a in out} | ({default_annotator} if default_annotator else set())
    return out, annotators


def parse_eaf(path: Path, allow_other: bool = False) -> tuple[list[Annotation], set[str]]:
    """Parse an ELAN .eaf file with the stdlib.

    Each time-aligned annotation is one event; its value is the event label. The
    annotator is the tier's ANNOTATOR attribute, else PARTICIPANT, else the TIER_ID.
    Symbolic (reference) annotations carry no times and are ignored. Documents with a
    DOCTYPE or entity declarations are refused (no entity-expansion attacks).
    Returns (annotations, annotators of every tier, including empty tiers).
    """
    text = _read_text(path)
    if re.search(r"<!(DOCTYPE|ENTITY)", text, re.IGNORECASE):
        raise IngestError(f"{path.name}: DOCTYPE/ENTITY declarations are not allowed in .eaf files")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise IngestError(f"{path.name}: not valid XML ({exc}). Re-export it from ELAN.") from exc
    if root.tag != "ANNOTATION_DOCUMENT":
        raise IngestError(f"{path.name}: not an ELAN annotation document (root <{root.tag}>)")
    slots: dict[str, float] = {}
    for slot in root.iter("TIME_SLOT"):
        value = slot.get("TIME_VALUE")
        if value is not None:
            if not value.strip().isdigit():
                raise IngestError(f"{path.name}: TIME_VALUE '{value}' is not whole milliseconds. Re-export from ELAN.")
            slots[slot.get("TIME_SLOT_ID", "")] = int(value) / 1000.0
    out: list[Annotation] = []
    annotators: set[str] = set()
    for tier in root.iter("TIER"):
        tier_id = tier.get("TIER_ID", "")
        who_raw = tier.get("ANNOTATOR") or tier.get("PARTICIPANT") or tier_id
        annotators.add(_annotator(who_raw, f"{path.name} tier '{tier_id}'"))
        for ann in tier.iter("ALIGNABLE_ANNOTATION"):
            origin = f"{path.name} tier '{tier_id}' annotation {ann.get('ANNOTATION_ID', '?')}"
            value_el = ann.find("ANNOTATION_VALUE")
            value = (value_el.text or "") if value_el is not None else ""
            if not value.strip():
                continue  # unlabelled segment
            ref1, ref2 = ann.get("TIME_SLOT_REF1", ""), ann.get("TIME_SLOT_REF2", "")
            if ref1 not in slots or ref2 not in slots:
                raise IngestError(f"{origin}: time slot without a time value. Align it in ELAN and re-export.")
            if len(out) >= MAX_ROWS:
                raise IngestError(f"{path.name}: more than {MAX_ROWS} annotations")
            label, etype = normalise_label(value, allow_other, origin)
            start, end = slots[ref1], slots[ref2]
            _check_interval(start, end, origin)
            out.append(Annotation(_annotator(who_raw, origin), label, etype, start, end, origin))
    return out, annotators


def parse_manifest(path: Path) -> dict[str, dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(_read_text(path)))
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    reader.fieldnames = fields
    if "video" not in fields:
        raise IngestError(f"{path.name}: needs a 'video' column (the video's file name in this folder)")
    unknown = set(fields) - set(MANIFEST_COLUMNS)
    if unknown:
        raise IngestError(f"{path.name}: unknown column(s) {', '.join(sorted(unknown))}; "
                          f"allowed: {', '.join(MANIFEST_COLUMNS)}")
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(reader, start=2):
        origin = f"{path.name} row {i}"
        name = (row.get("video") or "").strip()
        if not name:
            continue
        meta: dict[str, Any] = {}
        camera = (row.get("camera") or "").strip().lower()
        if camera:
            if camera not in CAMERA_VALUES:
                raise IngestError(f"{origin}: camera must be one of {', '.join(CAMERA_VALUES)}")
            meta["camera"] = camera
        subjects = (row.get("subjects") or "").strip()
        if subjects:
            if not subjects.isdigit() or int(subjects) < 1:
                raise IngestError(f"{origin}: subjects must be a whole number >= 1 (people visible in the clip)")
            meta["subjects"] = int(subjects)
        for key in ("participant", "session"):
            value = (row.get(key) or "").strip()
            if value:
                if not CODE_RE.fullmatch(value):
                    raise IngestError(f"{origin}: {key} must be a code such as P01 or S1, never a name")
                meta[key] = value
        for key in ("camera_angle_deg", "distance_m"):
            value = (row.get(key) or "").strip()
            if value:
                meta[key] = _time(value, origin, key)
        notes = (row.get("notes") or "").strip()
        if notes:
            meta["notes"] = notes[:200]
        if name in out:
            raise IngestError(f"{origin}: video '{name}' is listed twice")
        out[name] = meta
    return out


# ---------------------------------------------------------------------------- discovery


def discover(source: Path, allow_other: bool = False) -> list[VideoItem]:
    """Pair every video in ``source`` with its annotation files and manifest row."""
    if not source.is_dir():
        raise IngestError(f"{source} is not a folder")
    _refuse_source_in_repo(source)
    videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise IngestError(f"No videos ({', '.join(VIDEO_EXTENSIONS)}) found in {source}")
    stems = {v.stem: v for v in videos}
    if len(stems) != len(videos):
        raise IngestError("Two videos share a file stem (e.g. take1.mp4 and take1.mov); rename one.")
    manifest_path = source / "manifest.csv"
    manifest = parse_manifest(manifest_path) if manifest_path.is_file() else {}
    for name in manifest:
        if not (source / name).is_file():
            raise IngestError(f"manifest.csv lists '{name}', which is not in {source}")
    items = {v.stem: VideoItem(v, meta=dict(manifest.get(v.name, {}))) for v in videos}
    orphans: list[str] = []
    for p in sorted(source.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".eaf") or p.name == "manifest.csv":
            continue
        stem = p.name[: -len(p.suffix)]
        owner = stem if stem in items else next((s for s in sorted(items, key=len, reverse=True)
                                                 if stem.startswith(s + ".")), None)
        if owner is None:
            orphans.append(p.name)
            continue
        if p.suffix.lower() == ".eaf":
            parsed, who = parse_eaf(p, allow_other)
        else:
            middle = stem[len(owner) + 1:]
            parsed, who = parse_annotation_csv(p, allow_other, middle if CODE_RE.fullmatch(middle) else None)
        items[owner].annotations.extend(parsed)
        items[owner].annotators |= who
    if orphans:
        raise IngestError(
            f"Annotation file(s) without a matching video: {', '.join(orphans)}. Name them "
            "'<video stem>.csv' or '<video stem>.<annotator>.csv' (same for .eaf)."
        )
    missing = [i.source.name for i in items.values() if not i.annotators]
    if missing:
        raise IngestError(
            f"No annotations for: {', '.join(missing)}. Every video needs an annotation file per annotator. "
            "For a take with no events, use a header-only '<video stem>.<annotator>.csv' "
            "(see annotation_guide.md, 'Takes without events')."
        )
    return list(items.values())


def _refuse_source_in_repo(source: Path) -> None:
    """Footage inside the git checkout (outside the gitignored data/) is one `git add .` from a leak."""
    src = source.resolve()
    repo = config.PROJECT_ROOT.resolve()
    if src.is_relative_to(repo) and not src.is_relative_to((repo / "data").resolve()):
        raise IngestError(
            f"{source} is inside the repository but outside the gitignored data/ folder. Move the raw "
            "recordings outside the repo (or under data/) so they can never be committed."
        )


# ---------------------------------------------------------------------------- validation


def validate_annotations(anns: list[Annotation], duration_s: float, video_name: str,
                         annotators: set[str] | None = None) -> list[str]:
    """Hard errors for one video's annotations (times beyond the video, same-label overlaps)."""
    errors: list[str] = []
    limit = duration_s + config.DURATION_TOLERANCE_S
    for a in anns:
        if a.end_s > limit:
            errors.append(f"{a.origin}: end_s {a.end_s:g} is beyond the end of {video_name} ({duration_s:g} s)")
    by_key: dict[tuple[str, str], list[Annotation]] = {}
    for a in anns:
        by_key.setdefault((a.annotator, a.label), []).append(a)
    for (who, label), group in sorted(by_key.items()):
        group.sort(key=lambda a: a.start_s)
        for prev, cur in zip(group, group[1:]):
            if cur.start_s < prev.end_s:
                errors.append(
                    f"{cur.origin}: overlaps {prev.origin} ('{label}' by annotator {who}). Merge them into one "
                    "event or fix the times; one person cannot do the same movement twice at once."
                )
    primary = sorted((annotators or {a.annotator for a in anns}) - {ADJUDICATOR})
    if len(primary) > 2:
        errors.append(
            f"{video_name}: {len(primary)} annotators ({', '.join(primary)}); the kit supports one or two "
            f"independent annotators plus an optional '{ADJUDICATOR}'."
        )
    return errors


# ---------------------------------------------------------------------------- agreement


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_intervals(a: list[tuple[float, float]], b: list[tuple[float, float]],
                    min_iou: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching by descending IoU. Returns (pairs, unmatched_a, unmatched_b)."""
    candidates = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                         if iou(x, y) >= min_iou), key=lambda c: (-c[0], c[1], c[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, score))
    pairs.sort()
    return (pairs, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b])


def _bins(intervals: Iterable[tuple[float, float]], duration_s: float, bin_s: float) -> list[bool]:
    n = max(1, math.ceil(duration_s / bin_s - 1e-9))
    out = [False] * n
    for s, e in intervals:
        first, last = int(s // bin_s), min(n - 1, int(math.ceil(e / bin_s - 1e-9)) - 1)
        for k in range(max(0, first), last + 1):
            out[k] = True
    return out


def cohens_kappa(x: list[bool], y: list[bool]) -> float | None:
    """Cohen's kappa for two binary label sequences; None when chance agreement is total."""
    n = len(x)
    if n == 0 or n != len(y):
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    if pe >= 1.0 - 1e-12:
        return None  # both raters constant and identical: kappa undefined
    return (po - pe) / (1 - pe)


def interval_ioa(a: list[tuple[float, float]], b: list[tuple[float, float]], duration_s: float,
                 interval_s: float = 10.0) -> dict[str, float | None]:
    """Interval-by-interval IOA (occurrence agreement) and exact count-per-interval IOA.

    Events are assigned to the interval containing their onset (standard ABA counting).
    """
    n = max(1, math.ceil(duration_s / interval_s - 1e-9))
    ca, cb = [0] * n, [0] * n
    for s, _ in a:
        ca[min(n - 1, int(s // interval_s))] += 1
    for s, _ in b:
        cb[min(n - 1, int(s // interval_s))] += 1
    occurrence = sum((x > 0) == (y > 0) for x, y in zip(ca, cb)) / n
    exact = sum(1.0 if x == y else 0.0 for x, y in zip(ca, cb)) / n
    count_per_interval = sum(1.0 if x == y == 0 else min(x, y) / max(x, y) for x, y in zip(ca, cb)) / n
    return {"interval_ioa": round(occurrence, 3), "exact_count_ioa": round(exact, 3),
            "mean_count_per_interval_ioa": round(count_per_interval, 3)}


def agreement(anns_a: list[Annotation], anns_b: list[Annotation], duration_s: float, min_iou: float = 0.5,
              bin_s: float = 1.0, interval_s: float = 10.0) -> dict[str, Any]:
    """Per-label agreement between two annotators of one video."""
    labels = sorted({a.label for a in anns_a} | {b.label for b in anns_b})
    per_label: dict[str, Any] = {}
    for label in labels:
        a = sorted((x.start_s, x.end_s) for x in anns_a if x.label == label)
        b = sorted((x.start_s, x.end_s) for x in anns_b if x.label == label)
        pairs, ua, ub = match_intervals(a, b, min_iou)
        onset = [abs(a[i][0] - b[j][0]) for i, j, _ in pairs]
        offset = [abs(a[i][1] - b[j][1]) for i, j, _ in pairs]
        kappa = cohens_kappa(_bins(a, duration_s, bin_s), _bins(b, duration_s, bin_s))
        denom = len(pairs) + len(ua) + len(ub)
        per_label[label] = {
            "count_a": len(a), "count_b": len(b), "matched": len(pairs),
            "only_a": len(ua), "only_b": len(ub),
            "event_agreement": round(len(pairs) / denom, 3) if denom else None,
            "mean_iou": round(sum(p[2] for p in pairs) / len(pairs), 3) if pairs else None,
            "mean_onset_diff_s": round(sum(onset) / len(onset), 3) if onset else None,
            "mean_offset_diff_s": round(sum(offset) / len(offset), 3) if offset else None,
            "kappa": None if kappa is None else round(kappa, 3),
            **interval_ioa(a, b, duration_s, interval_s),
        }
    return per_label


def pooled_agreement(items: list[tuple[list[Annotation], list[Annotation], float]], min_iou: float,
                     bin_s: float = 1.0) -> dict[str, Any]:
    """Dataset-level event agreement and kappa per label, pooling bins and events over videos."""
    labels = sorted({a.label for x, y, _ in items for a in x + y})
    out: dict[str, Any] = {}
    for label in labels:
        matched = only_a = only_b = 0
        bins_a: list[bool] = []
        bins_b: list[bool] = []
        for anns_a, anns_b, duration in items:
            a = sorted((x.start_s, x.end_s) for x in anns_a if x.label == label)
            b = sorted((x.start_s, x.end_s) for x in anns_b if x.label == label)
            pairs, ua, ub = match_intervals(a, b, min_iou)
            matched, only_a, only_b = matched + len(pairs), only_a + len(ua), only_b + len(ub)
            bins_a += _bins(a, duration, bin_s)
            bins_b += _bins(b, duration, bin_s)
        denom = matched + only_a + only_b
        kappa = cohens_kappa(bins_a, bins_b)
        out[label] = {"matched": matched, "only_a": only_a, "only_b": only_b,
                      "event_agreement": round(matched / denom, 3) if denom else None,
                      "kappa": None if kappa is None else round(kappa, 3)}
    return out


def consensus(anns: list[Annotation], min_iou: float,
              annotators: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Reference events for one video and the disagreements left for adjudication.

    * one annotator: their events, as marked;
    * an ``adjudicator`` present: the adjudicator's events (they replace the consensus);
    * two annotators: events matched one-to-one (same label, IoU >= min_iou) become one event
      spanning the UNION of the pair; unmatched events are left out and listed as disagreements.
    """
    names = sorted(annotators or {a.annotator for a in anns})
    if ADJUDICATOR in names:
        chosen = [a for a in anns if a.annotator == ADJUDICATOR]
        return [_event(a.type, a.start_s, a.end_s, a.label) for a in chosen], [], "adjudicated"
    if len(names) == 1:
        return [_event(a.type, a.start_s, a.end_s, a.label) for a in anns], [], "single"
    first, second = names
    events: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for label in sorted({a.label for a in anns}):
        xa = sorted((a for a in anns if a.annotator == first and a.label == label), key=lambda a: a.start_s)
        xb = sorted((a for a in anns if a.annotator == second and a.label == label), key=lambda a: a.start_s)
        pairs, ua, ub = match_intervals([(a.start_s, a.end_s) for a in xa], [(b.start_s, b.end_s) for b in xb],
                                        min_iou)
        for i, j, _ in pairs:
            events.append(_event(xa[i].type, min(xa[i].start_s, xb[j].start_s), max(xa[i].end_s, xb[j].end_s), label))
        for who, group, idx in ((first, xa, ua), (second, xb, ub)):
            for k in idx:
                disagreements.append({"annotator": who, "label": label, "start_s": group[k].start_s,
                                      "end_s": group[k].end_s, "reason": "no matching event from the other annotator"})
    events.sort(key=lambda e: (e["start_s"], e["type"]))
    disagreements.sort(key=lambda d: (d["start_s"], d["label"]))
    return events, disagreements, "consensus"


def _event(etype: str, start: float, end: float, label: str) -> dict[str, Any]:
    return {"type": etype, "start_s": round(start, 3), "end_s": round(end, 3), "source_label": label}


# ---------------------------------------------------------------------------- ids and video


def _salt(directory: Path) -> bytes:
    path = directory / ".id_salt"
    if path.is_file():
        return bytes.fromhex(path.read_text().strip())
    directory.mkdir(parents=True, exist_ok=True)
    value = secrets.token_bytes(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value.hex())
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opaque_id(content_sha256: str, salt: bytes) -> str:
    """Stable for the same file and dataset, unlinkable to the filename or (without the salt) the file."""
    return "stg_" + hmac.new(salt, content_sha256.encode(), hashlib.sha256).hexdigest()[:16]


def transcode(src: Path, dst: Path, fps: float, max_height: int) -> None:
    """H.264 MP4, constant frame rate, first video stream only, no audio and no metadata."""
    exe = ffmpeg_exe()
    if exe is None:
        raise IngestError("ffmpeg is unavailable; install imageio-ffmpeg (pip install -r requirements.txt)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
        "-vf", f"scale=-2:'min({max_height},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", f"{fps:.6g}", "-fps_mode", "cfr",
        "-movflags", "+faststart", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    except (subprocess.SubprocessError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"{src.name}: ffmpeg could not convert the video ({type(exc).__name__}). "
                          "Check it plays, or re-export it as MP4/H.264.") from exc
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------- dataset io


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def validate_dataset_entry(entry: dict[str, Any]) -> None:
    """Assert one annotations.json entry matches the shared format exactly."""
    keys = {"video_id", "video", "fps", "duration_s", "camera", "subjects", "events"}
    if set(entry) != keys:
        raise AssertionError(f"entry keys {sorted(entry)} != {sorted(keys)}")
    if not (isinstance(entry["video_id"], str) and entry["video"] == f"videos/{entry['video_id']}.mp4"):
        raise AssertionError("video must be videos/<video_id>.mp4")
    if not (isinstance(entry["fps"], float) and isinstance(entry["duration_s"], float)):
        raise AssertionError("fps and duration_s must be floats")
    if entry["camera"] not in CAMERA_VALUES:
        raise AssertionError("camera must be fixed|handheld|unknown")
    if entry["subjects"] is not None and not (isinstance(entry["subjects"], int) and entry["subjects"] >= 1):
        raise AssertionError("subjects must be a positive int or null")
    for ev in entry["events"]:
        if set(ev) != {"type", "start_s", "end_s", "source_label"} or ev["type"] not in SCHEMA_TYPES:
            raise AssertionError(f"bad event {ev}")
        if not (isinstance(ev["start_s"], float) and isinstance(ev["end_s"], float) and ev["start_s"] < ev["end_s"]):
            raise AssertionError(f"bad event times {ev}")
        if not isinstance(ev["source_label"], str):
            raise AssertionError("source_label must be a string")


# ---------------------------------------------------------------------------- commands


def ingest(source: Path, check_only: bool = False, default_camera: str = "unknown", min_iou: float = 0.5,
           strict: bool = False, allow_other: bool = False, max_height: int = 1080,
           out: Any = None) -> dict[str, Any]:
    """Validate ``source`` and (unless ``check_only``) add or update its videos in the dataset."""
    out = out or sys.stdout
    items = discover(source, allow_other)
    errors: list[str] = []
    probes: dict[Path, Any] = {}
    for item in items:
        try:
            probes[item.source] = probe_video(item.source)
        except VideoValidationError as exc:
            errors.append(f"{item.source.name}: {exc}")
            continue
        errors += validate_annotations(item.annotations, probes[item.source].duration_s, item.source.name,
                                       item.annotators)
    if errors:
        raise IngestError("Fix these problems and run again:\n  - " + "\n  - ".join(errors))

    planned: list[tuple[VideoItem, list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any] | None]] = []
    pooled: list[tuple[list[Annotation], list[Annotation], float]] = []
    for item in items:
        duration = probes[item.source].duration_s
        events, disagreements, mode = consensus(item.annotations, min_iou, item.annotators)
        primary = sorted(item.annotators - {ADJUDICATOR})
        agree = None
        if len(primary) == 2:
            a = [x for x in item.annotations if x.annotator == primary[0]]
            b = [x for x in item.annotations if x.annotator == primary[1]]
            agree = {"annotators": primary, "per_label": agreement(a, b, duration, min_iou)}
            pooled.append((a, b, duration))
        planned.append((item, events, disagreements, mode, agree))
    pending = sum(len(p[2]) for p in planned)
    if strict and pending:
        raise IngestError(f"{pending} unresolved annotator disagreement(s) and --strict is set. Add an "
                          f"'{ADJUDICATOR}' annotation file for each listed video (annotation_guide.md).")

    directory = _safe_dataset_dir()
    salt = _salt(directory) if not check_only else secrets.token_bytes(32)
    dataset = {e["video_id"]: e for e in _load_json(directory / "annotations.json", [])}
    manifest: dict[str, Any] = _load_json(directory / "manifest.json", {})
    report: dict[str, Any] = _load_json(directory / "ingest_report.json", {"videos": {}})
    rows = []
    for item, events, disagreements, mode, agree in planned:
        video_id = opaque_id(sha256_file(item.source), salt)
        camera = item.meta.get("camera", default_camera)
        rows.append((item.source.name, video_id if not check_only else "(check only)", len(events),
                     len(disagreements), mode))
        if check_only:
            continue
        dst = directory / "videos" / f"{video_id}.mp4"
        transcode(item.source, dst, probes[item.source].fps, max_height)
        meta = probe_video(dst)
        entry = {"video_id": video_id, "video": f"videos/{video_id}.mp4", "fps": float(round(meta.fps, 3)),
                 "duration_s": float(meta.duration_s), "camera": camera,
                 "subjects": item.meta.get("subjects"), "events": events}
        validate_dataset_entry(entry)
        dataset[video_id] = entry
        manifest[video_id] = {k: v for k, v in item.meta.items() if k not in ("camera", "subjects")}
        report["videos"][video_id] = {"reference": mode, "annotators": sorted(item.annotators),
                                      "agreement": agree, "disagreements": disagreements}
        log.info("staged_video_ingested", extra={"fields": {
            "video_id": video_id, "events": len(events), "disagreements": len(disagreements), "reference": mode}})

    width = max(len(r[0]) for r in rows)
    print(f"{'source file':<{width}}  video_id              events  disagreements  reference", file=out)
    for name, vid, n, d, mode in rows:
        print(f"{name:<{width}}  {vid:<20}  {n:>6}  {d:>13}  {mode}", file=out)
    print("(This mapping is shown once and not stored. Keep it with your private session log if you need "
          "to trace a video_id back to a recording.)", file=out)
    if pooled:
        print("\nInter-rater agreement this run (pooled; event agreement = matched / all events, "
              f"IoU >= {min_iou}; kappa on {1.0:g} s bins):", file=out)
        for label, s in pooled_agreement(pooled, min_iou).items():
            print(f"  {label:<20} matched {s['matched']:>3}  only A {s['only_a']:>3}  only B {s['only_b']:>3}  "
                  f"agreement {s['event_agreement']}  kappa {s['kappa']}", file=out)
    if pending:
        print(f"\n{pending} disagreement(s) left out of the reference set; see ingest_report.json and "
              "annotation_guide.md 'Adjudication'.", file=out)

    summary = {"videos": len(planned), "events": sum(len(p[1]) for p in planned), "disagreements": pending,
               "check_only": check_only}
    if check_only:
        print("\nCheck passed; nothing was written.", file=out)
        log.info("staged_ingest_checked", extra={"fields": summary})
        return summary
    entries = [dataset[k] for k in sorted(dataset)]
    _write_json(directory / "annotations.json", entries)
    _write_json(directory / "manifest.json", {k: manifest[k] for k in sorted(manifest)})
    report.update({"kit_version": KIT_VERSION, "consensus_rule": f"union of one-to-one matches, IoU >= {min_iou}",
                   "pooled_agreement_last_run": pooled_agreement(pooled, min_iou) if pooled else None})
    _write_json(directory / "ingest_report.json", report)
    summary["dataset_videos"] = len(entries)
    log.info("staged_ingest_done", extra={"fields": summary})
    print(f"\nDataset now has {len(entries)} video(s) in data/validation/staged/.", file=out)
    return summary


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Refusing to delete without confirmation; pass --yes.", file=sys.stderr)
        return False
    return input(f"{prompt} Type 'delete' to confirm: ").strip() == "delete"


def _remove_caches(video_ids: Iterable[str]) -> int:
    """Remove evaluator landmark caches whose filename contains a deleted video_id."""
    cache = config.data_dir() / "validation" / "cache"
    ids = [v for v in video_ids if VIDEO_ID_RE.fullmatch(v)]
    removed = 0
    if cache.is_dir() and not cache.is_symlink() and ids:
        for p in cache.iterdir():
            if p.is_file() and any(v in p.name for v in ids):
                p.unlink()
                removed += 1
    return removed


def delete_videos(video_ids: list[str], assume_yes: bool = False, out: Any = None) -> int:
    """Remove some videos (e.g. a withdrawn participant) and their annotations from the dataset."""
    out = out or sys.stdout
    for v in video_ids:
        if not VIDEO_ID_RE.fullmatch(v):
            raise IngestError(f"'{v}' is not a staged video_id (expected stg_ + 16 hex characters)")
    directory = _safe_dataset_dir()
    dataset = _load_json(directory / "annotations.json", [])
    in_dataset = {e["video_id"] for e in dataset} & set(video_ids)
    on_disk = {v for v in video_ids if (directory / "videos" / f"{v}.mp4").is_file()}
    present = sorted(in_dataset | on_disk)
    if not present:
        print("None of these videos are in the dataset; nothing to delete.", file=out)
        return 0
    if not _confirm(f"Delete {len(present)} video(s) and their annotations?", assume_yes):
        return 0
    _write_json(directory / "annotations.json", [e for e in dataset if e["video_id"] not in present])
    manifest = _load_json(directory / "manifest.json", {})
    _write_json(directory / "manifest.json", {k: v for k, v in manifest.items() if k not in present})
    report = _load_json(directory / "ingest_report.json", {"videos": {}})
    report["videos"] = {k: v for k, v in report.get("videos", {}).items() if k not in present}
    _write_json(directory / "ingest_report.json", report)
    for v in present:
        (directory / "videos" / f"{v}.mp4").unlink(missing_ok=True)
    caches = _remove_caches(present)
    log.info("staged_videos_deleted", extra={"fields": {"videos": len(present), "cache_files": caches}})
    print(f"Deleted {len(present)} video(s) and {caches} cache file(s).", file=out)
    return len(present)


def delete_participant(code: str, assume_yes: bool = False, out: Any = None) -> int:
    out = out or sys.stdout
    if not CODE_RE.fullmatch(code):
        raise IngestError("participant must be a code such as P01")
    manifest = _load_json(_safe_dataset_dir() / "manifest.json", {})
    ids = sorted(k for k, v in manifest.items() if v.get("participant") == code)
    if not ids:
        print(f"No videos recorded for participant {code} in manifest.json; nothing deleted. If their clips "
              "were ingested without a participant code, delete them by video_id.", file=out)
        return 0
    return delete_videos(ids, assume_yes, out)


def delete_dataset(assume_yes: bool = False, out: Any = None) -> bool:
    """Remove the staged dataset: known files only, then the folders if empty."""
    out = out or sys.stdout
    directory = _safe_dataset_dir()
    if not directory.exists():
        print("No staged dataset to delete.", file=out)
        return False
    videos_dir = directory / "videos"
    videos = sorted(p for p in videos_dir.glob("*.mp4") if VIDEO_ID_RE.fullmatch(p.stem)) if videos_dir.is_dir() else []
    if not _confirm(f"Delete the staged dataset ({len(videos)} video(s))?", assume_yes):
        return False
    for p in videos:
        p.unlink()
    for p in videos_dir.glob("*.part.mp4") if videos_dir.is_dir() else []:
        p.unlink()
    for name in DATASET_FILES:
        (directory / name).unlink(missing_ok=True)
    caches = _remove_caches(p.stem for p in videos)
    leftovers = [p.name for p in directory.rglob("*")]
    for d in (videos_dir, directory):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    log.info("staged_dataset_deleted", extra={"fields": {"videos": len(videos), "cache_files": caches,
                                                         "leftover_files": len(leftovers)}})
    print(f"Deleted {len(videos)} video(s), the annotations and {caches} cache file(s).", file=out)
    if [n for n in leftovers if n != "videos"]:
        print("Left unrecognised files in place (not created by this script): "
              f"{', '.join(n for n in leftovers if n != 'videos')}", file=out)
    print("Not deleted: your raw recordings, consent forms, OS backups or snapshots. Remove raw recordings "
          "yourself per the consent retention terms (README 'Delete').", file=out)
    return True


# ---------------------------------------------------------------------------- synthetic example


def make_synthetic_example(directory: Path, fps: float = 10.0) -> list[Path]:
    """Two SYNTHETIC stick-figure clips (from the demo timeline) with two annotators each.

    Annotator A marks the scripted ground truth; annotator B is offset by up to 0.3 s and
    misses one event, so the agreement report and the adjudication list are non-trivial.
    MediaPipe does not find a pose in stick figures: this exercises plumbing only.
    """
    from behaviorlens.demo_generator import GROUND_TRUTH, render_frame
    from behaviorlens.video_io import FrameWriter

    directory.mkdir(parents=True, exist_ok=True)
    clips = {"synthetic_take01": (0.0, 20.0), "synthetic_take02": (20.0, 40.0)}
    written = []
    for name, (t0, t1) in clips.items():
        path = directory / f"{name}.mp4"
        with FrameWriter(path, 640, 480, fps) as writer:
            t = t0
            while t < t1 - 1e-9:
                writer.write(render_frame(t))
                t += 1.0 / fps
        targets = [(str(g["event_type"]), float(str(g["start_s"])) - t0, float(str(g["end_s"])) - t0)
                   for g in GROUND_TRUTH
                   if g["event_type"] in ("standing_transition", "hand_raised") and t0 <= float(str(g["start_s"])) < t1]
        context = ([("hand_flick", 12.6, 12.85), ("sitting_down", 13.5, 14.7)] if t0 == 0
                   else [("occlusion", 3.0, 5.0), ("out_of_frame", 10.0, 12.0)])
        rows_a = targets + context
        # B: onsets 0.2 s later, offsets 0.3 s later, and misses the last target event.
        rows_b = [(lab, s + 0.2, e + 0.3) for lab, s, e in targets[:-1]] + context
        for who, rows in (("A", rows_a), ("B", rows_b)):
            with (directory / f"{name}.{who}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["event_type", "start_s", "end_s", "annotator", "notes"])
                for lab, s, e in rows:
                    w.writerow([lab, f"{s:.2f}", f"{e:.2f}", who, "synthetic"])
        written.append(path)
    with (directory / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "participant", "camera", "subjects", "session", "camera_angle_deg", "notes"])
        for name in clips:
            w.writerow([f"{name}.mp4", "SYN01", "fixed", 1, "S0", 0, "SYNTHETIC stick figure, not real footage"])
    return written


# ---------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", nargs="?", type=Path, help="folder of videos + annotation files")
    ap.add_argument("--check", action="store_true", help="validate only; write nothing")
    ap.add_argument("--camera", choices=CAMERA_VALUES, default="unknown",
                    help="camera value for videos without one in manifest.csv (default: unknown)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for matching two annotators' events (default 0.5)")
    ap.add_argument("--strict", action="store_true", help="fail if any annotator disagreement is unadjudicated")
    ap.add_argument("--allow-other-labels", action="store_true", help="keep unlisted labels as type 'other'")
    ap.add_argument("--max-height", type=int, default=1080, help="downscale taller videos (default 1080)")
    ap.add_argument("--delete", action="store_true", help="delete the whole staged dataset")
    ap.add_argument("--delete-video", action="append", default=[], metavar="VIDEO_ID")
    ap.add_argument("--delete-participant", metavar="CODE", help="delete every video of one participant code")
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation when deleting")
    ap.add_argument("--synthetic-example", type=Path, metavar="DIR",
                    help="write two SYNTHETIC stick-figure clips + annotations to DIR (plumbing demo)")
    args = ap.parse_args(argv)
    configure_logging()
    commands = [bool(args.source), args.delete, bool(args.delete_video), bool(args.delete_participant),
                bool(args.synthetic_example)]
    if sum(commands) != 1:
        ap.error("give exactly one of SOURCE, --delete, --delete-video, --delete-participant, --synthetic-example")
    if not 0 < args.iou <= 1:
        ap.error("--iou must be in (0, 1]")
    try:
        if args.delete:
            delete_dataset(args.yes)
        elif args.delete_video:
            delete_videos(args.delete_video, args.yes)
        elif args.delete_participant:
            delete_participant(args.delete_participant, args.yes)
        elif args.synthetic_example:
            for p in make_synthetic_example(args.synthetic_example):
                print(f"wrote SYNTHETIC clip {p.name} (+ A/B annotations, manifest.csv)")
        else:
            ingest(args.source, check_only=args.check, default_camera=args.camera, min_iou=args.iou,
                   strict=args.strict, allow_other=args.allow_other_labels, max_height=args.max_height)
    except IngestError as exc:
        log.info("staged_ingest_failed", extra={"fields": {"error": type(exc).__name__}})
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
