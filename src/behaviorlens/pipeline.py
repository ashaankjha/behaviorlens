"""Orchestration: session creation, pose extraction, rule analysis, persistence.

Core logic is UI-independent. ``analyze_pose`` runs the rules on any
PoseSequence (synthetic or inferred) without touching video.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np

from . import config, demo_generator
from .config import AnalysisSettings, RuleParams
from .event_rules import detect_all, observability_stats
from .models import NUM_LANDMARKS, VISIBILITY, X, Y, AnalysisResult, PoseSequence, VideoMeta, new_id
from .pose import MediaPipePoseEstimator, PoseEstimator
from .storage import Store
from .video_io import (
    FrameWriter,
    VideoValidationError,
    iter_sampled_frames,
    make_browser_preview,
    probe_video,
    sanitize_filename,
    sha256_file,
    validate_upload,
)

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


# ---------------------------------------------------------------- analysis
def analyze_pose(seq: PoseSequence, params: RuleParams, session_id: str) -> AnalysisResult:
    stats = observability_stats(seq, params)
    warnings: list[str] = []
    if len(seq) == 0 or stats["pose_detected_ratio"] == 0:
        warnings.append("No person was detected in any sampled frame, so no events could be suggested.")
    elif stats["pose_detected_ratio"] < 0.5:
        warnings.append(f"A person was detected in only {stats['pose_detected_ratio']:.0%} of sampled frames.")
    if stats["multi_person_frame_ratio"] > 0:
        warnings.append(
            f"More than one person was detected in {stats['multi_person_frame_ratio']:.0%} of frames. "
            "Events are attributed to the largest detected person; attribution is not reliable."
        )
    if 0 < len(seq) and stats["posture_rule_observable_ratio"] < 0.5:
        warnings.append(
            f"Hips and knees were visible in only {stats['posture_rule_observable_ratio']:.0%} of frames; "
            "standing events are unobservable elsewhere (not assumed absent)."
        )
    events = detect_all(seq, params, session_id)
    stats["event_counts"] = {t: sum(e.event_type == t for e in events) for t in config.EVENT_TYPES}
    return AnalysisResult(events=events, warnings=warnings, stats=stats)


def extract_pose(video_path: Path, meta: VideoMeta, estimator: PoseEstimator, sample_fps: float,
                 progress: Progress = _noop) -> tuple[PoseSequence, list[str]]:
    warnings: list[str] = []
    times, indices, rows, people = [], [], [], []
    total = max(1, int(meta.duration_s * min(sample_fps, meta.fps)))
    for k, (frame_idx, t, frame) in enumerate(iter_sampled_frames(video_path, meta, sample_fps, warnings)):
        lms, n = estimator.detect(frame, t)
        times.append(t)
        indices.append(frame_idx)
        rows.append(lms if lms is not None else np.full((NUM_LANDMARKS, 5), np.nan))
        people.append(n)
        if k % 5 == 0:
            progress(min(0.99, (k + 1) / total), f"Pose inference: frame {k + 1}/{total}")
    if not times:
        raise VideoValidationError("No frames could be decoded from this video.")
    seq = PoseSequence(
        timestamps_s=np.array(times), frame_indices=np.array(indices), landmarks=np.stack(rows),
        sample_fps=min(sample_fps, meta.fps), duration_s=meta.duration_s,
        num_people=np.array(people), data_mode="video_inference",
    )
    return seq, warnings


# ---------------------------------------------------------------- sessions
def _copy_bounded(src: BinaryIO, dst: BinaryIO, limit: int, chunk: int = 1 << 20) -> int:
    """Copy at most ``limit`` bytes; raise as soon as the source turns out to be larger."""
    total = 0
    while block := src.read(chunk):
        total += len(block)
        if total > limit:
            raise VideoValidationError("File exceeds the 100 MB upload limit.")
        dst.write(block)
    return total


def create_upload_session(store: Store, filename: str, fileobj: BinaryIO, size_bytes: int) -> str:
    """Validate and store an uploaded video locally. Raises VideoValidationError."""
    head = fileobj.read(16)
    ext = validate_upload(filename, size_bytes, head)
    fileobj.seek(0)
    session_id = None
    tmp_dir = store.root / "sessions" / "_incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{new_id()}{ext}"
    try:
        with open(tmp, "wb") as f:
            size = _copy_bounded(fileobj, f, config.MAX_UPLOAD_BYTES)
        meta = probe_video(tmp)
        checksum = sha256_file(tmp)
        session_id = store.create_session(
            source_filename=sanitize_filename(filename), checksum=checksum,
            data_mode="video_inference", meta=meta.model_dump())
        sdir = store.session_dir(session_id)
        video = sdir / f"source{ext}"
        shutil.move(str(tmp), video)
        preview = make_browser_preview(video, sdir / "preview.mp4", meta)
        store.update_session(session_id, video_relpath=store.relpath(video),
                             preview_relpath=store.relpath(preview))
        log.info("session_created", extra={"fields": {"session_id": session_id, "data_mode": "video_inference",
                                                      "size_bytes": size, "duration_s": meta.duration_s,
                                                      "fps": meta.fps, "codec": meta.codec}})
        return session_id
    except Exception as exc:
        log.warning("upload_rejected", extra={"fields": {"exc_type": type(exc).__name__}})
        tmp.unlink(missing_ok=True)
        if session_id:
            store.delete_session(session_id)
        raise


def create_demo_session(store: Store) -> str:
    assets = demo_generator.ensure_demo_assets()
    seq = PoseSequence.load_npz(assets["pose"])
    truth = json.loads(assets["truth"].read_text())["events"]
    meta = {"fps": 15.0, "width": 640, "height": 480,
            "frame_count": int(seq.duration_s * 15), "duration_s": seq.duration_s}
    session_id = store.create_session(
        source_filename="SYNTHETIC_demo_session", checksum="synthetic-demo-" + config.ALGORITHM_VERSION,
        data_mode="synthetic", meta=meta, ground_truth=truth)
    sdir = store.session_dir(session_id)
    shutil.copy(assets["pose"], sdir / "pose_fixture.npz")
    fields: dict[str, Any] = {}
    if assets["video"].is_file():
        video = sdir / "demo.mp4"
        shutil.copy(assets["video"], video)
        fields = {"video_relpath": store.relpath(video), "preview_relpath": store.relpath(video)}
    if fields:
        store.update_session(session_id, **fields)
    log.info("session_created", extra={"fields": {"session_id": session_id, "data_mode": "synthetic"}})
    return session_id


def _pose_cache_path(store: Store, checksum: str, model_version: str, sample_fps: float) -> Path:
    safe_model = model_version.replace("@", "_").replace("/", "_")
    return store.cache_dir() / f"{checksum}_{safe_model}_{sample_fps:g}fps.npz"


def run_analysis(store: Store, session_id: str, settings: AnalysisSettings | None = None,
                 estimator_factory: Callable[[], PoseEstimator] | None = None,
                 progress: Progress = _noop) -> dict[str, Any]:
    """Run (or re-run) analysis for a session and store suggestions.

    Pending suggestions from earlier runs are replaced; reviewed and manual
    events are preserved. The UI asks for explicit confirmation before calling
    this when pending suggestions exist.
    """
    settings = settings or AnalysisSettings()
    session = store.get_session(session_id)
    previous_status = session["status"]
    store.update_session(session_id, status="processing", error=None)
    started = time.monotonic()
    log.info("analysis_started", extra={"fields": {"session_id": session_id, "data_mode": session["data_mode"],
                                                   "sample_fps": settings.sample_fps}})
    try:
        warnings: list[str] = []
        model_ver = None
        if session["data_mode"] == "synthetic":
            fixture = store.session_dir(session_id) / "pose_fixture.npz"
            seq = PoseSequence.load_npz(fixture)
            if settings.sample_fps != seq.sample_fps:
                warnings.append(f"Synthetic fixture is sampled at {seq.sample_fps:g} fps; "
                                f"requested {settings.sample_fps:g} fps ignored.")
            progress(0.5, "Loaded synthetic pose fixture (no inference)")
        else:
            video = store.path(session["video_relpath"])
            if video is None or not video.is_file():
                raise VideoValidationError("The source video for this session is missing. Re-upload it.")
            meta = probe_video(video)
            factory = estimator_factory or MediaPipePoseEstimator
            estimator = factory()
            try:
                model_ver = estimator.model_version
                cache = _pose_cache_path(store, session["checksum"], model_ver, settings.sample_fps)
                if cache.is_file():
                    seq = PoseSequence.load_npz(cache)
                    progress(0.9, "Loaded cached pose landmarks")
                else:
                    seq, warnings = extract_pose(video, meta, estimator, settings.sample_fps, progress)
                    seq.save_npz(cache)
            finally:
                estimator.close()
        result = analyze_pose(seq, settings.rules, session_id)
        warnings += result.warnings
        analysis_id = store.add_analysis(
            session_id, settings=settings.to_dict(), stats=result.stats, warnings=warnings,
            model_version=model_ver, data_mode=seq.data_mode, pose=seq)
        counts = store.save_suggestions(session_id, analysis_id, result.events)
        store.update_session(session_id, status="review_ready")
        store.refresh_review_status(session_id)
        log.info("analysis_completed", extra={"fields": {
            "session_id": session_id, "analysis_id": analysis_id, "model_version": model_ver,
            "algorithm_version": config.ALGORITHM_VERSION, "frames": len(seq),
            "event_counts": result.stats.get("event_counts"), "warnings": len(warnings),
            "elapsed_s": round(time.monotonic() - started, 2), **counts}})
        progress(1.0, "Analysis complete")
        return {"analysis_id": analysis_id, "warnings": warnings, "stats": result.stats, **counts}
    except Exception as exc:
        status = "failed" if previous_status in ("ready", "processing", "failed") else previous_status
        store.update_session(session_id, status=status, error=str(exc))
        log.warning("analysis_failed", extra={"fields": {"session_id": session_id, "exc_type": type(exc).__name__,
                                                         "elapsed_s": round(time.monotonic() - started, 2)}})
        raise


# ---------------------------------------------------------------- overlay (P1)
_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
          (23, 25), (25, 27), (24, 26), (26, 28)]


def render_overlay(store: Store, session_id: str, max_height: int = 360,
                   visibility_threshold: float = 0.5) -> Path:
    """Low-resolution local overlay video of sampled frames with pose landmarks."""
    session = store.get_session(session_id)
    analysis = store.latest_analysis(session_id)
    if session["data_mode"] != "video_inference" or analysis is None:
        raise ValueError("Overlay is only available for analyzed uploaded videos.")
    seq = store.load_landmarks(analysis)
    video = store.path(session["video_relpath"])
    if seq is None or video is None or not video.is_file():
        raise ValueError("The landmarks or source video for this session are missing. Re-run analysis.")
    meta = probe_video(video)
    scale = min(1.0, max_height / meta.height)
    w, h = int(meta.width * scale) // 2 * 2, int(meta.height * scale) // 2 * 2
    out = store.session_dir(session_id) / "overlay.mp4"
    lookup = {int(fi): k for k, fi in enumerate(seq.frame_indices)}
    has_pose = seq.has_pose
    with FrameWriter(out, w, h, seq.sample_fps) as writer:
        for frame_idx, t, frame in iter_sampled_frames(video, meta, seq.sample_fps):
            frame = cv2.resize(frame, (w, h))
            k = lookup.get(frame_idx)
            if k is not None and has_pose[k]:
                lm = seq.landmarks[k]
                vis = np.nan_to_num(lm[:, VISIBILITY]) >= visibility_threshold
                pts = {i: (int(lm[i, X] * w), int(lm[i, Y] * h)) for i in range(NUM_LANDMARKS) if vis[i]}
                for a, b in _BONES:
                    if a in pts and b in pts:
                        cv2.line(frame, pts[a], pts[b], (80, 230, 120), 2, cv2.LINE_AA)
                for p in pts.values():
                    cv2.circle(frame, p, 3, (40, 180, 255), -1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "no pose", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{t:6.1f}s", (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    store.update_session(session_id, overlay_relpath=store.relpath(out))
    return out


# ---------------------------------------------------------------- thumbnails (P1)
def event_thumbnails(store: Store, session_id: str, start_s: float, end_s: float,
                     offset_s: float = 0.5, height: int = 200) -> list[tuple[float, Path]]:
    """Local JPEG frames just before an event starts and just after it ends.

    Cached in the session directory (removed by Delete session) and keyed by
    timestamp, so editing an event's times produces fresh thumbnails. Shown in
    the UI only; never exported.
    """
    session = store.get_session(session_id)
    video = store.path(session.get("video_relpath"))
    if video is None or not video.is_file():
        return []
    fps = float(session["fps"] or 0)
    last = max(0.0, (session["duration_s"] or 0) - (1.0 / fps if fps else 0.0))
    times = [max(0.0, start_s - offset_s), min(last, end_s + offset_s)]
    out_dir = store.session_dir(session_id) / "thumbs"
    out_dir.mkdir(exist_ok=True)
    results = []
    cap = cv2.VideoCapture(str(video))
    try:
        for t in times:
            path = out_dir / f"{t:08.2f}.jpg"
            if not path.is_file():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)) if fps else 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                scale = height / frame.shape[0]
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), height))
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            results.append((round(t, 2), path))
    finally:
        cap.release()
    return results
