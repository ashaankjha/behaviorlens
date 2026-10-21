"""Person tracking across sampled frames.

MediaPipe returns an unordered list of people per frame. Choosing the largest
person independently in every frame (rules-v0.2) lets attribution jump between
people mid-clip. This module links detections into tracks using geometry only:
landmark bounding-box overlap and centre distance after a constant-velocity
prediction, plus size similarity. There is **no appearance, face or identity
model**. A track ID means "one position-continuous body in this clip"; a person
unseen for longer than ``TrackingParams.max_gap_s`` gets a new ID rather than a
guessed one.

Pure numpy, so it runs on synthetic sequences without video.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .config import TrackingParams
from .models import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    NUM_LANDMARKS,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    VISIBILITY,
    X,
    Y,
    DataMode,
    Event,
    PoseSequence,
)
from .pose import select_primary

# Exhaustive assignment is exact and cheap for the few people MediaPipe returns per frame.
_EXACT_MAX_DETECTIONS = 6
_EXACT_MAX_TRACKS = 10
# Speed cap for the constant-velocity prediction (normalized image widths per second).
_MAX_SPEED = 2.0
# Floor for degenerate (zero-width) boxes, so ratios and scales stay finite.
_MIN_AREA = 1e-4
# Per-track rows kept in analysis stats (largest scores first); the total is reported separately.
_MAX_TRACKS_IN_STATS = 10
# Frames at each end of a track whose median box describes it for fragment merging.
_JUNCTION_FRAMES = 3
# Landmarks whose frame-to-frame displacement measures an identity jump: the body core, not the
# arms or legs, so a fast arm raise or a leg flickering out of view does not count as a jump.
_JUMP_LANDMARKS = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
_JUMP_MIN_LANDMARKS = 3


# ---------------------------------------------------------------- multi-person landmarks
@dataclass
class MultiPoseSequence:
    """Every detected person per sampled frame.

    ``poses`` has shape (N, P, 33, 5) with the PoseSequence channels; slots at or
    beyond ``counts[k]`` are NaN. Slot order within a frame carries no identity.
    """

    timestamps_s: np.ndarray
    frame_indices: np.ndarray
    poses: np.ndarray
    counts: np.ndarray
    sample_fps: float
    duration_s: float
    data_mode: DataMode = "video_inference"
    # Per-frame camera motion (camera_motion.MOTION_CHANNELS); None when not estimated.
    camera_motion: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamps_s = np.asarray(self.timestamps_s, dtype=float)
        self.frame_indices = np.asarray(self.frame_indices, dtype=int)
        self.poses = np.asarray(self.poses, dtype=float)
        if self.camera_motion is not None:
            self.camera_motion = np.asarray(self.camera_motion, dtype=float)
            if len(self.camera_motion) != len(self.timestamps_s):
                raise ValueError("camera_motion must have one row per frame")
        self.counts = np.asarray(self.counts, dtype=int)
        n = len(self.timestamps_s)
        if self.poses.ndim != 4 or self.poses.shape[0] != n or self.poses.shape[2:] != (NUM_LANDMARKS, 5):
            raise ValueError(f"poses must have shape ({n}, P, 33, 5), got {self.poses.shape}")
        if self.counts.shape != (n,) or np.any(self.counts < 0) or np.any(self.counts > self.poses.shape[1]):
            raise ValueError("counts must give 0..P detections per frame")
        if n > 1 and np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps must be strictly increasing")

    def __len__(self) -> int:
        return len(self.timestamps_s)

    @property
    def max_people(self) -> int:
        return int(self.poses.shape[1])

    def detections(self, k: int) -> list[np.ndarray]:
        return [self.poses[k, j] for j in range(int(self.counts[k]))]

    @classmethod
    def from_frames(cls, timestamps_s: ArrayLike, frame_indices: ArrayLike, frames: list[list[np.ndarray]],
                    sample_fps: float, duration_s: float, data_mode: DataMode = "video_inference",
                    camera_motion: np.ndarray | None = None) -> "MultiPoseSequence":
        slots = max([len(f) for f in frames] + [1])
        poses = np.full((len(frames), slots, NUM_LANDMARKS, 5), np.nan)
        for k, people in enumerate(frames):
            for j, lm in enumerate(people):
                poses[k, j] = lm
        return cls(timestamps_s=np.asarray(timestamps_s, dtype=float),
                   frame_indices=np.asarray(frame_indices, dtype=int), poses=poses,
                   counts=np.array([len(f) for f in frames], dtype=int), sample_fps=sample_fps,
                   duration_s=duration_s, data_mode=data_mode, camera_motion=camera_motion)

    @classmethod
    def from_pose_sequence(cls, seq: PoseSequence) -> "MultiPoseSequence":
        """One slot per frame: the people a single-person sequence did not keep are unknown."""
        return cls(timestamps_s=seq.timestamps_s, frame_indices=seq.frame_indices,
                   poses=seq.landmarks[:, None], counts=seq.has_pose.astype(int),
                   sample_fps=seq.sample_fps, duration_s=seq.duration_s, data_mode=seq.data_mode,
                   camera_motion=seq.camera_motion)

    def largest_person_sequence(self) -> PoseSequence:
        """rules-v0.2 attribution: ``pose.select_primary`` in each frame, chosen independently."""
        out = np.full((len(self), NUM_LANDMARKS, 5), np.nan)
        for k in range(len(self)):
            if self.counts[k]:
                out[k] = select_primary(self.detections(k))
        return self._sequence(out, self.counts)

    def _sequence(self, landmarks: np.ndarray, num_people: np.ndarray) -> PoseSequence:
        return PoseSequence(timestamps_s=self.timestamps_s, frame_indices=self.frame_indices,
                            landmarks=landmarks, sample_fps=self.sample_fps, duration_s=self.duration_s,
                            num_people=num_people, data_mode=self.data_mode, camera_motion=self.camera_motion)

    def save_npz(self, path: str | Path) -> None:
        extra: dict[str, Any] = ({} if self.camera_motion is None
                                 else {"camera_motion": self.camera_motion.astype(np.float32)})
        np.savez_compressed(path, timestamps_s=self.timestamps_s, frame_indices=self.frame_indices,
                            poses=self.poses.astype(np.float32), counts=self.counts,
                            meta=np.array([self.sample_fps, self.duration_s]),
                            data_mode=np.array(self.data_mode), **extra)

    @classmethod
    def load_npz(cls, path: str | Path) -> "MultiPoseSequence":
        with np.load(path, allow_pickle=False) as d:
            return cls(timestamps_s=d["timestamps_s"], frame_indices=d["frame_indices"],
                       poses=d["poses"].astype(float), counts=d["counts"], sample_fps=float(d["meta"][0]),
                       duration_s=float(d["meta"][1]), data_mode=_data_mode(str(d["data_mode"])),
                       # Optional: caches written before camera-motion estimation lack it.
                       camera_motion=d["camera_motion"].astype(float) if "camera_motion" in d.files else None)


# ---------------------------------------------------------------- geometry
def landmark_bbox(lm: np.ndarray, visibility_threshold: float | None = None) -> np.ndarray | None:
    """(x0, y0, x1, y1) of all landmark positions, or only of those with at least
    ``visibility_threshold`` (falling back to all if fewer than 3 qualify).

    Tracking uses all positions: MediaPipe places every landmark each frame, and that box
    is far steadier than the visible-landmark box, which jumps whenever the legs flicker
    in and out of visibility (LEARNINGS M26).
    """
    finite = np.isfinite(lm[:, X]) & np.isfinite(lm[:, Y])
    use = finite
    if visibility_threshold is not None:
        visible = finite & (np.nan_to_num(lm[:, VISIBILITY], nan=0.0) >= visibility_threshold)
        use = visible if visible.sum() >= 3 else finite
    if not use.any():
        return None
    xs, ys = lm[use, X], lm[use, Y]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def _area(b: np.ndarray) -> float:
    return max(float((b[2] - b[0]) * (b[3] - b[1])), _MIN_AREA)


def _center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0.0) * max(h, 0.0)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------- association
@dataclass
class _Track:
    track_id: int
    box: np.ndarray
    last_t: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    updates: int = 0

    def predict(self, t: float, max_gap_s: float) -> np.ndarray:
        dx, dy = self.velocity * min(t - self.last_t, max_gap_s)
        return np.asarray(self.box + np.array([dx, dy, dx, dy]), dtype=float)

    def update(self, box: np.ndarray, t: float) -> None:
        dt = t - self.last_t
        if dt > 0:
            v = (_center(box) - _center(self.box)) / dt
            self.velocity = v if self.updates == 0 else 0.5 * self.velocity + 0.5 * v
            speed = float(np.hypot(*self.velocity))
            if speed > _MAX_SPEED:
                self.velocity *= _MAX_SPEED / speed
        self.box, self.last_t = box, t
        self.updates += 1


def match_cost(track: _Track, box: np.ndarray, t: float, p: TrackingParams) -> float | None:
    """Cost of continuing ``track`` with ``box`` at time ``t``; None when gated out."""
    dt = t - track.last_t
    pred = track.predict(t, p.max_gap_s)
    size = abs(math.log(_area(box) / _area(track.box)))
    if size > math.log(p.max_size_ratio):
        return None
    overlap = iou(pred, box)
    dist = float(np.hypot(*(_center(pred) - _center(box)))) / math.sqrt(_area(track.box))
    if overlap < p.min_iou and dist > p.max_center_dist + p.gate_growth_per_s * dt:
        return None
    return (1.0 - overlap) + p.dist_weight * dist + p.size_weight * size + p.stale_weight * dt


def assign(cost: list[list[float | None]]) -> list[int | None]:
    """Track index (or None) per detection: most matches first, then least total cost.

    Exhaustive for small frames (exact, deterministic); greedy by cost otherwise.
    """
    n_det = len(cost)
    n_tr = len(cost[0]) if cost else 0
    if n_det == 0:
        return []
    if n_det <= _EXACT_MAX_DETECTIONS and n_tr <= _EXACT_MAX_TRACKS:
        best: tuple[int, float, list[int | None]] = (-1, math.inf, [None] * n_det)

        def search(j: int, used: frozenset[int], matched: int, total: float, chosen: list[int | None]) -> None:
            nonlocal best
            if j == n_det:
                if matched > best[0] or (matched == best[0] and total < best[1] - 1e-12):
                    best = (matched, total, chosen)
                return
            for i in range(n_tr):
                c = cost[j][i]
                if c is not None and i not in used:
                    search(j + 1, used | {i}, matched + 1, total + c, chosen + [i])
            search(j + 1, used, matched, total, chosen + [None])

        search(0, frozenset(), 0, 0.0, [])
        return best[2]
    pairs = sorted((c, j, i) for j, row in enumerate(cost) for i, c in enumerate(row) if c is not None)
    out: list[int | None] = [None] * n_det
    used: set[int] = set()
    for _, j, i in pairs:
        if out[j] is None and i not in used:
            out[j] = i
            used.add(i)
    return out


# ---------------------------------------------------------------- results
@dataclass(frozen=True)
class TrackSummary:
    track_id: int
    frames: int
    coverage: float  # share of all sampled frames in which the track is present
    first_s: float
    last_s: float
    mean_bbox_area: float  # normalized image area of the landmark bounding box
    score: float  # coverage x mean_bbox_area, used to choose the default subject
    substantial: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _data_mode(value: str) -> DataMode:
    if value not in ("synthetic", "video_inference"):
        raise ValueError(f"unknown data_mode {value!r}")
    return cast(DataMode, value)

@dataclass
class TrackingResult:
    multi: MultiPoseSequence
    # (N, P) track ID per detection slot; 0 = empty slot or dropped duplicate.
    assignment: np.ndarray
    # (N, P, 4) landmark bounding box per slot (NaN when unassigned).
    boxes: np.ndarray
    params: TrackingParams
    duplicates_dropped: int = 0
    fragments_merged: int = 0
    identity_splits: int = 0

    @property
    def track_ids(self) -> list[int]:
        return sorted(int(t) for t in np.unique(self.assignment) if t > 0)

    def present(self, track_id: int) -> np.ndarray:
        return np.asarray(np.any(self.assignment == track_id, axis=1))

    def people_per_frame(self) -> np.ndarray:
        return np.asarray(np.sum(self.assignment > 0, axis=1))

    def track_landmarks(self, track_id: int | None) -> np.ndarray:
        out = np.full((len(self.multi), NUM_LANDMARKS, 5), np.nan)
        if track_id is None:
            return out
        k, j = np.nonzero(self.assignment == track_id)
        out[k] = self.multi.poses[k, j]
        return out

    def track_sequence(self, track_id: int | None) -> PoseSequence:
        """PoseSequence of one track; frames where it is absent are NaN (unobservable).

        With ``identity_jump_guard = "mask"``, frames inside an identity-jump window
        (``jump_masked_frames``) are NaN as well. ``num_people`` keeps the number of tracked
        people per frame.
        """
        landmarks = self.track_landmarks(track_id)
        landmarks[self.jump_masked_frames(track_id)] = np.nan
        return self.multi._sequence(landmarks, self.people_per_frame())

    def identity_jump_frames(self, track_id: int | None) -> np.ndarray:
        """Frames where the track's body core jumped by at least ``identity_jump_threshold``."""
        jumps = identity_jumps(self.track_landmarks(track_id))
        return np.asarray(np.nan_to_num(jumps, nan=0.0) >= self.params.identity_jump_threshold)

    def jump_masked_frames(self, track_id: int | None) -> np.ndarray:
        """Frames made unobservable by the "mask" guard: from each identity jump of the track,
        for ``identity_jump_mask_s``. All False unless the guard is "mask"."""
        ts = self.multi.timestamps_s
        masked = np.zeros(len(ts), dtype=bool)
        if self.params.identity_jump_guard != "mask" or track_id is None:
            return masked
        for k in np.flatnonzero(self.identity_jump_frames(track_id)):
            masked |= (ts >= ts[k] - 1e-9) & (ts < ts[k] + self.params.identity_jump_mask_s - 1e-9)
        return masked

    def summaries(self) -> list[TrackSummary]:
        n = max(len(self.multi), 1)
        out = []
        for tid in self.track_ids:
            k, j = np.nonzero(self.assignment == tid)
            areas = [_area(b) for b in self.boxes[k, j]]
            coverage = len(k) / n
            mean_area = float(np.mean(areas))
            out.append(TrackSummary(
                track_id=tid, frames=len(k), coverage=round(coverage, 3),
                first_s=round(float(self.multi.timestamps_s[k.min()]), 3),
                last_s=round(float(self.multi.timestamps_s[k.max()]), 3),
                mean_bbox_area=round(mean_area, 4), score=round(coverage * mean_area, 5),
                substantial=len(k) / self.multi.sample_fps >= self.params.substantial_min_s - 1e-9))
        return out

    def auto_subject(self) -> int | None:
        """Greatest coverage x mean size; ties go to the earlier track."""
        ranked = sorted(self.summaries(), key=lambda s: (-s.coverage * s.mean_bbox_area, s.track_id))
        return ranked[0].track_id if ranked else None

    def largest_track_per_frame(self) -> np.ndarray:
        """Track ID of the largest detection per frame (0 if none): what v0.2 would have picked."""
        areas = np.where(self.assignment > 0, (self.boxes[..., 2] - self.boxes[..., 0])
                         * (self.boxes[..., 3] - self.boxes[..., 1]), -1.0)
        best = np.argmax(np.nan_to_num(areas, nan=-1.0), axis=1)
        return np.where(np.any(self.assignment > 0, axis=1), self.assignment[np.arange(len(best)), best], 0)

    def stats(self, subject_id: int | None, selection: str) -> dict[str, Any]:
        """Tracking and identity-switch statistics for analysis stats."""
        n = max(len(self.multi), 1)
        summaries = self.summaries()
        largest = self.largest_track_per_frame()
        seen = largest[largest > 0]
        switches = int(np.sum(seen[1:] != seen[:-1])) if seen.size > 1 else 0
        subject = next((s for s in summaries if s.track_id == subject_id), None)
        top = sorted(summaries, key=lambda s: (-s.score, s.track_id))[:_MAX_TRACKS_IN_STATS]
        return {
            "method": "landmark_bbox_iou_centroid_constant_velocity",
            "max_people": self.multi.max_people,
            "tracks": len(summaries),
            "substantial_tracks": sum(s.substantial for s in summaries),
            "subject_track_id": subject_id,
            "subject_selection": selection,
            "subject_coverage": subject.coverage if subject else 0.0,
            "multi_track_frame_ratio": round(float(np.sum(self.people_per_frame() > 1)) / n, 3),
            # How often per-frame "largest person" attribution (rules-v0.2) changes track.
            "largest_person_switches": switches,
            "subject_is_largest_ratio": round(float(np.mean(seen == subject_id)), 3) if seen.size else 0.0,
            "duplicate_detections_dropped": self.duplicates_dropped,
            "fragments_merged": self.fragments_merged,
            **self._guard_stats(subject_id),
            "per_track": [s.to_dict() for s in sorted(top, key=lambda s: s.track_id)],
            "params": self.params.to_dict(),
        }

    def _guard_stats(self, subject_id: int | None) -> dict[str, Any]:
        """Identity-jump guard statistics; empty when the guard is off (rules-v0.3 output unchanged)."""
        p = self.params
        if p.identity_jump_guard == "off":
            return {}
        return {"identity_jump_guard": {
            "mode": p.identity_jump_guard, "threshold": p.identity_jump_threshold,
            "mask_s": p.identity_jump_mask_s if p.identity_jump_guard == "mask" else None,
            "tracks_split": self.identity_splits,
            "subject_jumps": int(self.identity_jump_frames(subject_id).sum()) if subject_id is not None else 0,
            "subject_masked_frames": int(self.jump_masked_frames(subject_id).sum()),
        }}

    def save_npz(self, path: str | Path, subject_id: int | None) -> None:
        np.savez_compressed(path, timestamps_s=self.multi.timestamps_s, frame_indices=self.multi.frame_indices,
                            poses=self.multi.poses.astype(np.float32), counts=self.multi.counts,
                            meta=np.array([self.multi.sample_fps, self.multi.duration_s]),
                            data_mode=np.array(self.multi.data_mode), assignment=self.assignment,
                            subject=np.array(subject_id if subject_id is not None else 0))

    @staticmethod
    def load_overlay_data(path: str | Path) -> tuple[MultiPoseSequence, np.ndarray, int | None]:
        """Landmarks, per-slot track IDs and the subject saved by ``save_npz`` (for overlays)."""
        multi = MultiPoseSequence.load_npz(path)
        with np.load(path, allow_pickle=False) as d:
            subject = int(d["subject"])
            return multi, d["assignment"], subject or None


# ---------------------------------------------------------------- tracking
def _dedupe(boxes: list[tuple[int, np.ndarray]], max_iou: float) -> tuple[list[tuple[int, np.ndarray]], int]:
    """Largest first; drop detections that overlap a kept one by more than ``max_iou``."""
    kept: list[tuple[int, np.ndarray]] = []
    for slot, box in sorted(boxes, key=lambda sb: -_area(sb[1])):
        if all(iou(box, k) <= max_iou for _, k in kept):
            kept.append((slot, box))
    return kept, len(boxes) - len(kept)


def track_people(multi: MultiPoseSequence, params: TrackingParams | None = None) -> TrackingResult:
    """Link per-frame detections into tracks with stable IDs (1, 2, … in order of appearance)."""
    p = params or TrackingParams()
    n, slots = len(multi), multi.max_people
    assignment = np.zeros((n, slots), dtype=int)
    boxes = np.full((n, slots, 4), np.nan)
    tracks: list[_Track] = []
    dropped = 0
    for k in range(n):
        t = float(multi.timestamps_s[k])
        found = [(j, b) for j, lm in enumerate(multi.detections(k))
                 if (b := landmark_bbox(lm)) is not None]
        dets, d = _dedupe(found, p.duplicate_iou)
        dropped += d
        alive = [tr for tr in tracks if t - tr.last_t <= p.max_gap_s + 1e-9]
        choice = assign([[match_cost(tr, box, t, p) for tr in alive] for _, box in dets])
        for (slot, box), i in zip(dets, choice):
            if i is None:
                track = _Track(len(tracks) + 1, box, t)
                tracks.append(track)
            else:
                track = alive[i]
                track.update(box, t)
            assignment[k, slot] = track.track_id
            boxes[k, slot] = box
    result = TrackingResult(multi=multi, assignment=assignment, boxes=boxes, params=p, duplicates_dropped=dropped)
    if p.merge_fragments:
        result = merge_fragments(result)
    return split_at_identity_jumps(result) if p.identity_jump_guard == "split" else result


def identity_jumps(landmarks: np.ndarray) -> np.ndarray:
    """Per frame of one track's (N, 33, 5) landmarks: how far its body core moved since the
    previous frame in which the track was present, in units of the track's size there.

    The displacement is the median over nose, shoulders and hips (landmark positions, as for
    tracking; at least 3 must be finite in both frames), and the size is the square root of the
    earlier frame's all-landmark box area. NaN for frames without the track or without an
    earlier frame. A value near 1 means the core moved by a whole body size in one sampled
    frame: usually a different person (or a phantom pose) took over the track (H-03).
    """
    out = np.full(len(landmarks), np.nan)
    prev: int | None = None
    for k in range(len(landmarks)):
        box = landmark_bbox(landmarks[k])
        if box is None:
            continue
        if prev is not None:
            a = landmarks[prev, _JUMP_LANDMARKS][:, [X, Y]]
            b = landmarks[k, _JUMP_LANDMARKS][:, [X, Y]]
            d = np.hypot(*(b - a).T)
            d = d[np.isfinite(d)]
            prev_box = landmark_bbox(landmarks[prev])
            if len(d) >= _JUMP_MIN_LANDMARKS and prev_box is not None:
                out[k] = float(np.median(d)) / math.sqrt(_area(prev_box))
        prev = k
    return out


def split_at_identity_jumps(result: TrackingResult) -> TrackingResult:
    """Cut every track at each identity jump (``identity_jump_threshold``) into separate tracks,
    then renumber IDs in order of appearance. Never joins anything, so frames are only ever moved
    to a new track ID; the subject is then chosen among the pieces as usual, and frames of the
    other pieces are unobservable for it."""
    a = result.assignment
    out = np.zeros_like(a)
    pieces = 0
    next_id = 1
    for tid in result.track_ids:
        jumps = result.identity_jump_frames(tid)
        cur = next_id
        next_id += 1
        for k in np.flatnonzero(result.present(tid)):
            if jumps[k]:
                cur = next_id
                next_id += 1
                pieces += 1
            out[k][a[k] == tid] = cur
    first = {int(t): int(np.flatnonzero(np.any(out == t, axis=1))[0]) for t in np.unique(out) if t > 0}
    new_id = {t: n for n, t in enumerate(sorted(first, key=lambda t: (first[t], t)), 1)}
    renumbered = np.vectorize(lambda t: new_id.get(int(t), 0), otypes=[int])(out) if out.size else out
    return TrackingResult(multi=result.multi, assignment=renumbered, boxes=result.boxes, params=result.params,
                          duplicates_dropped=result.duplicates_dropped,
                          fragments_merged=result.fragments_merged, identity_splits=pieces)


def _junction_box(boxes: np.ndarray, frames: np.ndarray) -> np.ndarray:
    return np.asarray(np.median(boxes[frames], axis=0), dtype=float)


def merge_fragments(result: TrackingResult) -> TrackingResult:
    """Join tracks that are temporally disjoint and geometrically continuous (one person split
    across untracked gaps), then renumber IDs in order of appearance.

    Tracks are taken in order of first appearance; each joins the best-matching chain that ended
    before it started, within ``merge_max_gap_s``, ``merge_max_center_dist`` (in sizes of the
    chain's end box) and ``merge_max_size_ratio``. A chain only ever grows at its end, so a merged
    track is never present twice in one frame: **two tracks that coexist in any frame are never
    joined**, directly or through a chain (H-03). Geometry only, like tracking itself: two people
    who take turns at the same spot with the same apparent size would be joined.
    """
    p = result.params
    a = result.assignment
    ts = result.multi.timestamps_s
    box_of: dict[int, np.ndarray] = {}  # track -> (N, 4) box per frame (NaN when absent)
    for tid in result.track_ids:
        k, j = np.nonzero(a == tid)
        per = np.full((len(ts), 4), np.nan)
        per[k] = result.boxes[k, j]
        box_of[tid] = per
    frames_of = {tid: np.flatnonzero(np.isfinite(b[:, 0])) for tid, b in box_of.items()}
    chains: list[list[int]] = []
    for tid in sorted(frames_of, key=lambda t: (frames_of[t][0], t)):
        start = frames_of[tid][:_JUNCTION_FRAMES]
        head = _junction_box(box_of[tid], start)
        best: tuple[float, int] | None = None
        for c, chain in enumerate(chains):
            last = chain[-1]
            end = frames_of[last][-_JUNCTION_FRAMES:]
            if end[-1] >= start[0]:  # overlaps (or touches) in time: possibly another person
                continue
            gap = float(ts[start[0]] - ts[end[-1]])
            tail = _junction_box(box_of[last], end)
            size = abs(math.log(_area(head) / _area(tail)))
            dist = float(np.hypot(*(_center(head) - _center(tail)))) / math.sqrt(_area(tail))
            if gap > p.merge_max_gap_s + 1e-9 or size > math.log(p.merge_max_size_ratio) \
                    or dist > p.merge_max_center_dist:
                continue
            cost = dist + size + gap / max(p.merge_max_gap_s, 1e-9)
            if best is None or cost < best[0]:
                best = (cost, c)
        if best is None:
            chains.append([tid])
        else:
            chains[best[1]].append(tid)
    new_id = {tid: n for n, chain in enumerate(sorted(chains, key=lambda ch: frames_of[ch[0]][0]), 1)
              for tid in chain}
    merged = np.vectorize(lambda t: new_id.get(int(t), 0), otypes=[int])(a) if a.size else a.copy()
    return TrackingResult(multi=result.multi, assignment=merged, boxes=result.boxes, params=p,
                          duplicates_dropped=result.duplicates_dropped,
                          fragments_merged=len(new_id) - len(chains))


def annotate_events(events: list[Event], result: TrackingResult, subject_id: int | None) -> None:
    """Add the subject track and how often anyone else was in view to each event's evidence."""
    ts = result.multi.timestamps_s
    others = result.people_per_frame() - (result.present(subject_id) if subject_id is not None else 0)
    substantial = sum(s.substantial for s in result.summaries())
    for e in events:
        mask = (ts >= e.start_s - 1e-9) & (ts < e.end_s - 1e-9)
        e.evidence = {**e.evidence, "tracking": {
            "subject_track_id": subject_id,
            "tracked_people": substantial,
            "other_person_present_ratio": round(float(np.mean(others[mask] > 0)), 3) if mask.any() else 0.0,
        }}
