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
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np

from . import config, demo_generator
from .camera_motion import CAMERA_MOTION_VERSION, CameraMotionEstimator, camera_motion_stats
from .config import AnalysisSettings, RuleParams, TrackingParams
from .event_rules import detect_all, observability_stats
from .models import NUM_LANDMARKS, VISIBILITY, X, Y, AnalysisResult, PoseSequence, VideoMeta, new_id
from .pose import MediaPipePoseEstimator, PoseEstimator
from .storage import Store
from .tracking import MultiPoseSequence, TrackingResult, annotate_events, track_people
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
def analyze_pose(seq: PoseSequence, params: RuleParams, session_id: str,
                 tracking: dict[str, Any] | None = None) -> AnalysisResult:
    """Run the rules on one person's sequence. ``tracking`` (from ``TrackingResult.stats``)
    says that ``seq`` is one tracked subject among the people in view."""
    stats = observability_stats(seq, params)
    stats.update(camera_motion_stats(seq, params))
    warnings: list[str] = []
    if len(seq) == 0 or stats["pose_detected_ratio"] == 0:
        warnings.append("No person was detected in any sampled frame, so no events could be suggested.")
    elif stats["pose_detected_ratio"] < 0.5:
        who = "The subject" if tracking and tracking["substantial_tracks"] > 1 else "A person"
        warnings.append(f"{who} was detected in only {stats['pose_detected_ratio']:.0%} of sampled frames.")
    if tracking is not None:
        stats["tracking"] = tracking
        if tracking["substantial_tracks"] > 1:
            how = "by the reviewer" if tracking["subject_selection"] == "reviewer" else \
                "automatically (greatest coverage × size)"
            warnings.append(
                f"{tracking['substantial_tracks']} people were tracked. Events are computed for subject "
                f"track #{tracking['subject_track_id']} only ({tracking['subject_coverage']:.0%} of frames, "
                f"chosen {how}). Track numbers come from position continuity between frames, not recognition: "
                "check the pose overlay and choose another subject under Analysis controls if needed.")
    elif stats["multi_person_frame_ratio"] > 0:
        warnings.append(
            f"More than one person was detected in {stats['multi_person_frame_ratio']:.0%} of frames. "
            "Events are attributed to the largest detected person; attribution is not reliable."
        )
    if stats.get("camera_moving_ratio"):
        warnings.append(
            f"The camera moved (pan, tilt, zoom or shake) in {stats['camera_moving_ratio']:.0%} of sampled frames. "
            "Posture is treated as unobservable there, because landmark positions mix camera and body movement."
        )
    if 0 < len(seq) and stats["posture_rule_observable_ratio"] < 0.5:
        warnings.append(
            f"Hips and knees were visible in only {stats['posture_rule_observable_ratio']:.0%} of frames; "
            "standing events are unobservable elsewhere (not assumed absent)."
        )
    events = detect_all(seq, params, session_id)
    stats["event_counts"] = {t: sum(e.event_type == t for e in events) for t in config.EVENT_TYPES}
    return AnalysisResult(events=events, warnings=warnings, stats=stats)


def analyze_multi_pose(multi: MultiPoseSequence, params: RuleParams, session_id: str,
                       tracking: TrackingParams | None = None, subject_track_id: int | None = None,
                       ) -> tuple[AnalysisResult, TrackingResult, PoseSequence]:
    """Track everyone, then run the rules on one subject track only.

    The subject is ``subject_track_id`` when it exists, otherwise the track with the
    greatest coverage × mean bounding-box size. Frames without the subject are
    unobservable for the rules (never filled in from another person).
    """
    tracked = track_people(multi, tracking)
    subject, selection, notes = tracked.auto_subject(), "auto", []
    if subject_track_id is not None:
        if subject_track_id in tracked.track_ids:
            subject, selection = subject_track_id, "reviewer"
        else:
            notes.append(f"Requested subject track #{subject_track_id} does not exist in this analysis; "
                         "the automatic subject was used instead.")
    seq = tracked.track_sequence(subject)
    result = analyze_pose(seq, params, session_id, tracking=tracked.stats(subject, selection))
    annotate_events(result.events, tracked, subject)
    result.warnings[:0] = notes
    return result, tracked, seq


def extract_multi_pose(video_path: Path, meta: VideoMeta, estimator: PoseEstimator, sample_fps: float,
                       progress: Progress = _noop, camera_motion: bool = True,
                       ) -> tuple[MultiPoseSequence, list[str]]:
    """Every detected person per sampled frame, plus (by default) camera motion estimated with
    all detected people excluded from the background. Estimators without ``detect_all``
    (single person) contribute at most one detection per frame."""
    warnings: list[str] = []
    detect_all = getattr(estimator, "detect_all", None)
    motion_est = CameraMotionEstimator() if camera_motion else None
    times, indices, frames, motion = [], [], [], []
    total = max(1, int(meta.duration_s * min(sample_fps, meta.fps)))
    for k, (frame_idx, t, frame) in enumerate(iter_sampled_frames(video_path, meta, sample_fps, warnings)):
        if detect_all is not None:
            people = detect_all(frame, t)
        else:
            lms, _ = estimator.detect(frame, t)
            people = [] if lms is None else [lms]
        if motion_est is not None:
            motion.append(motion_est.update(frame, people))
        times.append(t)
        indices.append(frame_idx)
        frames.append(people)
        if k % 5 == 0:
            progress(min(0.99, (k + 1) / total), f"Pose inference: frame {k + 1}/{total}")
    if not times:
        raise VideoValidationError("No frames could be decoded from this video.")
    multi = MultiPoseSequence.from_frames(times, indices, frames, sample_fps=min(sample_fps, meta.fps),
                                          duration_s=meta.duration_s, data_mode="video_inference",
                                          camera_motion=np.stack(motion) if motion else None)
    return multi, warnings


def extract_pose(video_path: Path, meta: VideoMeta, estimator: PoseEstimator, sample_fps: float,
                 progress: Progress = _noop, camera_motion: bool = True) -> tuple[PoseSequence, list[str]]:
    """Pose landmarks (and, by default, camera motion) for each sampled frame."""
    warnings: list[str] = []
    times, indices, rows, people, motion = [], [], [], [], []
    motion_est = CameraMotionEstimator() if camera_motion else None
    total = max(1, int(meta.duration_s * min(sample_fps, meta.fps)))
    for k, (frame_idx, t, frame) in enumerate(iter_sampled_frames(video_path, meta, sample_fps, warnings)):
        lms, n = estimator.detect(frame, t)
        if motion_est is not None:
            motion.append(motion_est.update(frame, lms))
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
        camera_motion=np.stack(motion) if motion else None,
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


def _pose_cache_path(store: Store, checksum: str, model_version: str, sample_fps: float,
                     max_people: int | None = None) -> Path:
    """Landmark cache. ``max_people`` set: all-people landmarks (num_poses changes MediaPipe's
    output, so it is part of the key); tracking runs after the cache, on every analysis."""
    safe_model = model_version.replace("@", "_").replace("/", "_")
    multi = f"_multi{max_people}" if max_people is not None else ""
    # The camera-motion version is part of the key: older caches lack motion data.
    return store.cache_dir() / f"{checksum}_{safe_model}_{sample_fps:g}fps{multi}_{CAMERA_MOTION_VERSION}.npz"


def tracks_path(store: Store, session_id: str, analysis_id: str) -> Path:
    """All tracked people of one analysis (for the overlay); removed with the session."""
    return store.session_dir(session_id) / f"tracks_{analysis_id}.npz"


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
        tracked: TrackingResult | None = None
        if session["data_mode"] == "synthetic":
            fixture = store.session_dir(session_id) / "pose_fixture.npz"
            seq = PoseSequence.load_npz(fixture)
            if settings.sample_fps != seq.sample_fps:
                warnings.append(f"Synthetic fixture is sampled at {seq.sample_fps:g} fps; "
                                f"requested {settings.sample_fps:g} fps ignored.")
            progress(0.5, "Loaded synthetic pose fixture (no inference)")
            result = analyze_pose(seq, settings.rules, session_id)
        else:
            video = store.path(session["video_relpath"])
            if video is None or not video.is_file():
                raise VideoValidationError("The source video for this session is missing. Re-upload it.")
            meta = probe_video(video)
            factory = estimator_factory or partial(MediaPipePoseEstimator, max_people=settings.max_people)
            estimator = factory()
            try:
                model_ver = estimator.model_version
                cache = _pose_cache_path(store, session["checksum"], model_ver, settings.sample_fps,
                                         settings.max_people)
                if cache.is_file():
                    multi = MultiPoseSequence.load_npz(cache)
                    progress(0.9, "Loaded cached pose landmarks")
                else:
                    multi, warnings = extract_multi_pose(video, meta, estimator, settings.sample_fps, progress)
                    multi.save_npz(cache)
            finally:
                estimator.close()
            result, tracked, seq = analyze_multi_pose(multi, settings.rules, session_id, settings.tracking,
                                                      settings.subject_track_id)
        warnings += result.warnings
        analysis_id = store.add_analysis(
            session_id, settings=settings.to_dict(), stats=result.stats, warnings=warnings,
            model_version=model_ver, data_mode=seq.data_mode, pose=seq)
        if tracked is not None:
            tracked.save_npz(tracks_path(store, session_id, analysis_id),
                             result.stats["tracking"]["subject_track_id"])
            # A previous overlay may label another subject; it is re-rendered on request.
            old = store.path(session.get("overlay_relpath"))
            if old is not None:
                old.unlink(missing_ok=True)
                store.update_session(session_id, overlay_relpath=None)
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
    tracks_file = tracks_path(store, session_id, analysis["analysis_id"])
    tracks = TrackingResult.load_overlay_data(tracks_file) if tracks_file.is_file() else None
    with FrameWriter(out, w, h, seq.sample_fps) as writer:
        for frame_idx, t, frame in iter_sampled_frames(video, meta, seq.sample_fps):
            frame = cv2.resize(frame, (w, h))
            k = lookup.get(frame_idx)
            if tracks is not None and k is not None and tracks[0].counts[k] > 0:
                multi, assignment, subject = tracks
                # Other people first, so the subject's skeleton is drawn on top.
                slots = sorted((j for j in range(multi.max_people) if assignment[k, j] > 0),
                               key=lambda j: assignment[k, j] == subject)
                for j in slots:
                    tid = int(assignment[k, j])
                    _draw_pose(frame, multi.poses[k, j], w, h, visibility_threshold, tid == subject,
                               f"#{tid} subject" if tid == subject else f"#{tid}")
            elif tracks is None and k is not None and seq.has_pose[k]:
                _draw_pose(frame, seq.landmarks[k], w, h, visibility_threshold, True, None)
            else:
                cv2.putText(frame, "no pose", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{t:6.1f}s", (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    store.update_session(session_id, overlay_relpath=store.relpath(out))
    return out


def _draw_pose(frame: np.ndarray, lm: np.ndarray, w: int, h: int, visibility_threshold: float,
               subject: bool, label: str | None) -> None:
    """Skeleton (green for the subject, grey for other tracked people) and a track-number label."""
    vis = np.nan_to_num(lm[:, VISIBILITY]) >= visibility_threshold
    pts = {i: (int(lm[i, X] * w), int(lm[i, Y] * h)) for i in range(NUM_LANDMARKS) if vis[i]}
    bone, joint = ((80, 230, 120), (40, 180, 255)) if subject else ((170, 170, 170), (120, 120, 120))
    for a, b in _BONES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], bone, 2, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(frame, p, 3, joint, -1, cv2.LINE_AA)
    if label:
        finite = np.isfinite(lm[:, X]) & np.isfinite(lm[:, Y])
        if finite.any():
            x = int(np.clip(np.min(lm[finite, X]) * w, 2, w - 80))
            y = int(np.clip(np.min(lm[finite, Y]) * h - 6, 14, h - 4))
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bone, 1, cv2.LINE_AA)


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
