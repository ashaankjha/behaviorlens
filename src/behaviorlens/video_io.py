"""Video validation, metadata probing, frame sampling and preview transcoding."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from . import config
from .models import VideoMeta

log = logging.getLogger(__name__)


class VideoValidationError(ValueError):
    """Raised with a user-facing, actionable message."""


def sanitize_filename(name: str) -> str:
    """Basename only, restricted character set, bounded length."""
    base = Path(name.replace("\\", "/")).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "video"
    return f"{stem[:80]}{ext}"


def validate_upload(filename: str, size_bytes: int, head: bytes) -> str:
    """Check extension, size and container signature. Returns the lowercased extension.

    The browser-supplied MIME type is never trusted; the first bytes of the file
    must look like an MP4/MOV (``ftyp`` box) or AVI (``RIFF....AVI ``) container.
    """
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported file type '{ext or 'none'}'. Upload an MP4 (H.264 recommended), MOV or AVI file."
        )
    if size_bytes <= 0:
        raise VideoValidationError("The uploaded file is empty. Choose a different video.")
    if size_bytes > config.MAX_UPLOAD_BYTES:
        raise VideoValidationError(
            f"File is {size_bytes / 1e6:.0f} MB; the limit is {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB. "
            "Trim or compress the clip and try again."
        )
    is_iso = len(head) >= 12 and head[4:8] == b"ftyp"
    is_avi = len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI "
    if ext in (".mp4", ".mov") and not is_iso:
        raise VideoValidationError(
            "This file does not look like a valid MP4/MOV video (missing container header). "
            "Re-export it from your camera or editor as MP4."
        )
    if ext == ".avi" and not is_avi:
        raise VideoValidationError("This file does not look like a valid AVI video. Re-export it as MP4.")
    return ext


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _fourcc(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()


def probe_video(path: Path) -> VideoMeta:
    """Read container metadata with OpenCV and enforce duration/fps limits."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoValidationError(
                "The video could not be decoded (it may be corrupted or use an unsupported codec). "
                "Re-encode it as MP4/H.264, e.g. `ffmpeg -i in.mov -c:v libx264 out.mp4`."
            )
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        codec = _fourcc(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise VideoValidationError(
            "No frames could be decoded from this video (it may be corrupted). Try re-exporting it as MP4/H.264."
        )
    if not math.isfinite(fps) or fps < config.MIN_FPS or fps > config.MAX_FPS:
        raise VideoValidationError(
            f"The video reports an unusable frame rate ({fps:g} fps). Supported range is "
            f"{config.MIN_FPS:g}-{config.MAX_FPS:g} fps; re-encode with a constant frame rate, "
            "e.g. `ffmpeg -i in.mp4 -r 30 out.mp4`."
        )
    if frames <= 0:
        raise VideoValidationError("The video reports no frames. Re-export it as MP4/H.264.")
    duration = frames / fps
    if duration < config.MIN_DURATION_S:
        raise VideoValidationError(
            f"The clip is only {duration:.2f}s long; at least {config.MIN_DURATION_S:g}s is needed to detect events."
        )
    if duration > config.MAX_DURATION_S:
        raise VideoValidationError(
            f"The clip is {duration / 60:.1f} minutes long; the limit is {config.MAX_DURATION_S // 60} minutes. "
            "Trim it and try again."
        )
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    return VideoMeta(fps=fps, frame_count=frames, width=width, height=height,
                     duration_s=round(duration, 3), codec=codec)


def sample_frame_indices(meta: VideoMeta, target_fps: float) -> np.ndarray:
    """Source frame indices closest to a uniform ``target_fps`` grid.

    Timestamps are later derived from these indices and the source FPS, never
    from wall-clock processing time, so they are monotonic and reproducible.
    """
    target = min(target_fps, meta.fps)
    n = max(1, int(math.floor(meta.duration_s * target + 1e-9)))
    idx = np.round(np.arange(n) * meta.fps / target).astype(int)
    idx = idx[idx < meta.frame_count]
    return np.unique(idx)


def iter_sampled_frames(path: Path, meta: VideoMeta, target_fps: float,
                        warnings: list[str] | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, BGR frame) for sampled frames.

    Decoding is sequential (grab/retrieve), which is robust for codecs whose
    seeking is imprecise. A truncated/corrupted tail stops iteration with a warning.
    """
    wanted = sample_frame_indices(meta, target_fps)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoValidationError("The video could not be opened for decoding.")
    try:
        pos = 0
        for target in wanted:
            while pos < target:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read() if pos == target else (False, None)
            pos += 1
            if not ok or frame is None:
                if warnings is not None:
                    warnings.append(
                        f"Decoding stopped at frame {target} of {meta.frame_count} "
                        f"({target / meta.fps:.1f}s); the rest of the file could not be read."
                    )
                return
            yield int(target), round(target / meta.fps, 4), frame
    finally:
        cap.release()


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - depends on environment
        return None


def make_browser_preview(src: Path, dst: Path, meta: VideoMeta, max_height: int = 720) -> Path:
    """Return a path the browser can play.

    MP4 files that are already H.264 are used as-is; anything else is transcoded
    locally to H.264 with the bundled ffmpeg. Falls back to the original file
    (with a log warning) if transcoding is unavailable.
    """
    if src.suffix.lower() == ".mp4" and meta.codec in ("avc1", "h264"):
        return src
    exe = ffmpeg_exe()
    if exe is None:
        log.warning("ffmpeg unavailable; preview may not play in the browser")
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe, "-y", "-loglevel", "error", "-i", str(src),
        "-vf", f"scale=-2:'min({max_height},ih)'",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return dst
    except (subprocess.SubprocessError, OSError):
        log.warning("Preview transcode failed; using original file")
        return src


class FrameWriter:
    """Pipe raw BGR frames into ffmpeg to produce a browser-playable H.264 MP4."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        exe = ffmpeg_exe()
        if exe is None:
            raise RuntimeError("ffmpeg is unavailable; install imageio-ffmpeg")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width, height)
        self.proc = subprocess.Popen(
            [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait(timeout=600) != 0:
            raise RuntimeError("ffmpeg failed to encode video")

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
