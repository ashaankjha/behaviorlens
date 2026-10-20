"""Camera-motion estimation on synthetic videos with known pans, tilts, zooms and rotations."""

import cv2
import numpy as np
import pytest

from behaviorlens import pipeline
from behaviorlens.camera_motion import (
    CAMERA_MOTION_VERSION,
    METHOD_FEATURES,
    METHOD_PHASE,
    MOVING,
    STATIC,
    UNKNOWN_MOTION,
    CameraMotionEstimator,
    camera_motion_state,
    camera_motion_stats,
    camera_moving_mask,
    person_box,
)
from behaviorlens.config import RuleParams
from behaviorlens.models import (
    MOTION_CHANNELS,
    MOTION_DX,
    MOTION_DY,
    MOTION_INLIERS,
    MOTION_LOG_SCALE,
    MOTION_METHOD,
    MOTION_ROT_DEG,
    NUM_LANDMARKS,
    VISIBILITY,
    X,
    Y,
    PoseSequence,
)
from behaviorlens.video_io import probe_video

from conftest import make_seq

W, H = 640, 480
P = RuleParams()


@pytest.fixture(scope="module")
def world() -> np.ndarray:
    """A large textured background (3x the frame) to point a virtual camera at."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (H * 3 // 8, W * 3 // 8), dtype=np.uint8)
    tex = cv2.GaussianBlur(cv2.resize(noise, (W * 3, H * 3), interpolation=cv2.INTER_CUBIC), (5, 5), 0)
    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)


def view(world: np.ndarray, cx: float, cy: float, rot_deg: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """Camera centred on world point (cx, cy), rotated (counter-clockwise) and zoomed."""
    m = cv2.getRotationMatrix2D((cx, cy), rot_deg, zoom)
    m[:, 2] += [W / 2 - cx, H / 2 - cy]
    return cv2.warpAffine(world, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


CX, CY = W * 1.5, H * 1.5


def figure_landmarks(x0: float, y0: float, w: float = 0.18, h: float = 0.6) -> np.ndarray:
    """(33, 5) landmarks spread over a person-sized box (normalized coordinates)."""
    lm = np.zeros((NUM_LANDMARKS, 5))
    rng = np.random.default_rng(0)
    lm[:, X] = x0 + rng.uniform(0, w, NUM_LANDMARKS)
    lm[:, Y] = y0 + rng.uniform(0, h, NUM_LANDMARKS)
    lm[[0, 1], X], lm[[0, 1], Y] = (x0, x0 + w), (y0, y0 + h)  # make the box exact
    lm[:, VISIBILITY] = 0.9
    return lm


def draw_figure(frame: np.ndarray, x0: float, y0: float, phase: int, w: float = 0.18, h: float = 0.6) -> np.ndarray:
    """A strongly textured 'person' patch whose texture also changes, so it carries plenty of
    trackable corners that move independently of the background."""
    out = frame.copy()
    px0, py0, pw, ph = int(x0 * W), int(y0 * H), int(w * W), int(h * H)
    rng = np.random.default_rng(100 + phase)
    patch = cv2.resize(rng.integers(0, 255, (ph // 6, pw // 6), dtype=np.uint8), (pw, ph),
                       interpolation=cv2.INTER_NEAREST)
    out[py0:py0 + ph, px0:px0 + pw] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return out


# ------------------------------------------------------------------ estimator accuracy
@pytest.mark.parametrize("dx_px,dy_px,rot,zoom", [
    (0, 0, 0.0, 1.0),     # static
    (24, 0, 0.0, 1.0),    # pan right
    (-16, 0, 0.0, 1.0),   # pan left
    (0, 18, 0.0, 1.0),    # tilt down
    (0, 0, 3.0, 1.0),     # roll
    (0, 0, 0.0, 1.06),    # zoom in
    (12, -8, 1.5, 0.97),  # combined
])
def test_known_camera_motion_is_recovered(world, dx_px, dy_px, rot, zoom):
    est = CameraMotionEstimator()
    first = est.update(view(world, CX, CY))
    assert first[MOTION_METHOD] == 0 and np.isnan(first[MOTION_DX])  # no previous frame
    row = est.update(view(world, CX + dx_px, CY + dy_px, rot, zoom))
    assert row[MOTION_METHOD] == METHOD_FEATURES
    # Moving the camera right shifts the image content left.
    assert row[MOTION_DX] == pytest.approx(-dx_px / W, abs=0.004)
    assert row[MOTION_DY] == pytest.approx(-dy_px / H, abs=0.004)
    # In image coordinates (y down) a counter-clockwise camera roll appears as negative rotation.
    assert row[MOTION_ROT_DEG] == pytest.approx(-rot, abs=0.2)
    assert row[MOTION_LOG_SCALE] == pytest.approx(np.log(zoom), abs=0.006)
    assert row[MOTION_INLIERS] > 0.8


def test_fast_pan_falls_back_to_phase_correlation_or_features(world):
    est = CameraMotionEstimator()
    est.update(view(world, CX, CY))
    row = est.update(view(world, CX + 96, CY))  # 15% of the frame in one sample
    assert row[MOTION_METHOD] in (METHOD_FEATURES, METHOD_PHASE)
    assert row[MOTION_DX] == pytest.approx(-96 / W, abs=0.01)


def test_textureless_frames_are_unknown_not_static():
    est = CameraMotionEstimator()
    flat = np.full((H, W, 3), 90, np.uint8)
    est.update(flat)
    row = est.update(flat)
    assert row[MOTION_METHOD] == 0 and np.isnan(row[MOTION_DX])


def test_person_box_pads_and_clips():
    lm = figure_landmarks(0.0, 0.3, 0.2, 0.5)
    x0, y0, x1, y1 = person_box(lm, pad=0.15, min_visibility=0.3)
    assert x0 == 0.0 and y0 < 0.3 and x1 > 0.2 and y1 > 0.8
    hidden = lm.copy()
    hidden[:, VISIBILITY] = 0.0
    assert person_box(hidden, 0.15, 0.3) is None
    assert person_box(None, 0.15, 0.3) is None


# ------------------------------------------------------------------ sequences
def _run(world, cams, figures=None, fps: float = 5.0) -> PoseSequence:
    """Estimate motion for a list of camera poses (cx, cy, rot, zoom) and optional figures."""
    est = CameraMotionEstimator()
    rows, lms = [], []
    for k, cam in enumerate(cams):
        frame = view(world, *cam)
        lm = None
        if figures is not None:
            fx, fy = figures[k]
            frame = draw_figure(frame, fx, fy, phase=k)
            lm = figure_landmarks(fx, fy)
        rows.append(est.update(frame, lm))
        lms.append(lm if lm is not None else np.full((NUM_LANDMARKS, 5), np.nan))
    n = len(cams)
    return PoseSequence(timestamps_s=np.arange(n) / fps, frame_indices=np.arange(n), landmarks=np.stack(lms),
                        sample_fps=fps, duration_s=n / fps, camera_motion=np.stack(rows))


def test_static_camera_with_moving_person_stays_static(world):
    n = 20
    cams = [(CX, CY, 0.0, 1.0)] * n
    # The figure walks across a third of the frame, bobbing up and down, with changing texture.
    figures = [(0.2 + 0.02 * k, 0.25 + 0.05 * (k % 2)) for k in range(n)]
    seq = _run(world, cams, figures)
    state = camera_motion_state(seq, P)
    assert state[0] == UNKNOWN_MOTION
    assert np.all(state[1:] == STATIC)
    assert np.nanmax(np.abs(seq.camera_motion[:, [MOTION_DX, MOTION_DY]])) < 0.002
    assert not camera_moving_mask(seq, P).any()


def test_pan_with_moving_person_is_flagged_only_while_panning(world):
    # Static for 1.6 s, pans 40 px per sample (0.31 frame widths per second at 5 fps) for
    # 1.2 s, then static again. The figure moves independently the whole time.
    shifts = [0] * 8 + [40] * 6 + [0] * 8
    cams, x = [], CX
    for s in shifts:
        x += s
        cams.append((x, CY, 0.0, 1.0))
    figures = [(0.55 - 0.015 * k, 0.3) for k in range(len(cams))]
    seq = _run(world, cams, figures)
    state = camera_motion_state(seq, P)
    moving = np.flatnonzero(state == MOVING)
    assert moving.tolist() == list(range(8, 14))
    assert np.all(state[1:8] == STATIC) and np.all(state[14:] == STATIC)
    assert np.flatnonzero(camera_moving_mask(seq, P)).tolist() == list(range(8, 14))
    # Optional padding widens the posture mask by one sampled frame on each side.
    padded = RuleParams(camera_motion_pad_frames=1)
    assert np.flatnonzero(camera_moving_mask(seq, padded)).tolist() == list(range(7, 15))
    stats = camera_motion_stats(seq, P)
    assert stats["camera_motion_available"] and stats["camera_moving_ratio"] == pytest.approx(6 / 22, abs=1e-3)
    assert stats["camera_motion_intervals"] == [[1.4, 2.6]]


def test_slow_zoom_is_flagged(world):
    cams = [(CX, CY, 0.0, 1.0 + 0.04 * k) for k in range(6)]  # ~20%/s
    state = camera_motion_state(_run(world, cams), P)
    assert np.all(state[1:] == MOVING)


def test_handheld_jitter_below_threshold_is_static(world):
    rng = np.random.default_rng(3)
    cams = [(CX + rng.normal(0, 2), CY + rng.normal(0, 2), 0.0, 1.0) for _ in range(10)]
    state = camera_motion_state(_run(world, cams), P)
    assert np.all(state[1:] == STATIC)


def test_low_inlier_ratio_is_unknown():
    seq = make_seq(2)
    cm = np.zeros((len(seq), MOTION_CHANNELS))
    cm[:, MOTION_DX] = 0.2
    cm[:, MOTION_METHOD] = METHOD_FEATURES
    cm[:, MOTION_INLIERS] = 0.3
    seq.camera_motion = cm
    assert np.all(camera_motion_state(seq, P)[1:] == UNKNOWN_MOTION)
    cm[:, MOTION_INLIERS] = 0.9
    assert np.all(camera_motion_state(seq, P)[1:] == MOVING)
    assert not camera_moving_mask(seq, RuleParams(mask_camera_motion=False)).any()


# ------------------------------------------------------------------ persistence
def test_motion_round_trips_through_npz(tmp_path, world):
    seq = _run(world, [(CX + 10 * k, CY, 0.0, 1.0) for k in range(4)])
    seq.save_npz(tmp_path / "s.npz")
    back = PoseSequence.load_npz(tmp_path / "s.npz")
    assert back.camera_motion.shape == (4, MOTION_CHANNELS)
    assert np.allclose(back.camera_motion, seq.camera_motion, equal_nan=True, atol=1e-6)


def test_old_npz_without_motion_still_loads_and_masks_nothing(tmp_path):
    seq = make_seq(4)
    # The exact pre-camera-motion file layout.
    np.savez_compressed(tmp_path / "old.npz", timestamps_s=seq.timestamps_s, frame_indices=seq.frame_indices,
                        landmarks=seq.landmarks.astype(np.float32), num_people=seq.num_people,
                        meta=np.array([seq.sample_fps, seq.duration_s]), data_mode=np.array(seq.data_mode))
    old = PoseSequence.load_npz(tmp_path / "old.npz")
    assert old.camera_motion is None
    assert np.all(camera_motion_state(old, P) == UNKNOWN_MOTION)
    assert not camera_moving_mask(old, P).any()
    assert camera_motion_stats(old, P)["camera_motion_available"] is False


def test_motion_shape_is_validated():
    seq = make_seq(1)
    with pytest.raises(ValueError, match="camera_motion"):
        PoseSequence(timestamps_s=seq.timestamps_s, frame_indices=seq.frame_indices, landmarks=seq.landmarks,
                     sample_fps=5, duration_s=1, camera_motion=np.zeros((2, MOTION_CHANNELS)))


# ------------------------------------------------------------------ pipeline integration
class _NoPose:
    model_version = "none@1"

    def detect(self, frame, t):
        return None, 0

    def close(self):
        pass


def test_extract_pose_records_camera_motion_from_video(tmp_path, world):
    path = tmp_path / "pan.mp4"
    fps = 15
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    x = CX
    for k in range(fps * 4):  # 2 s still, 2 s panning 6 px per source frame (0.14 frame widths/s)
        x += 6 if k >= fps * 2 else 0
        writer.write(view(world, x, CY))
    writer.release()
    seq, _ = pipeline.extract_pose(path, probe_video(path), _NoPose(), 5.0)
    assert seq.camera_motion is not None and seq.camera_motion.shape == (len(seq), MOTION_CHANNELS)
    dx = seq.camera_motion[:, MOTION_DX]
    t = seq.timestamps_s
    assert np.nanmax(np.abs(dx[(t > 0) & (t < 1.9)])) < 0.003
    assert np.allclose(dx[t > 2.1], -18 / W, atol=0.004)  # 3 source frames per sample
    off, _ = pipeline.extract_pose(path, probe_video(path), _NoPose(), 5.0, camera_motion=False)
    assert off.camera_motion is None


def test_pose_cache_key_includes_camera_motion_version(store):
    p = pipeline._pose_cache_path(store, "abc", "lite@1", 5.0)
    assert CAMERA_MOTION_VERSION in p.name


# ------------------------------------------------------------------ rules-v0.3 integration with tracking
class _NoPeople:
    """Multi-person estimator that never detects anyone (background-only motion)."""

    model_version = "nopeople@1"

    def detect(self, frame, t):
        return None, 0

    def detect_all(self, frame, t):
        return []

    def close(self):
        pass


def _pan_video(path, world, fps=15, px_per_frame=6):
    """2 s still, then 2 s panning ``px_per_frame`` per source frame (6 px = 0.14 frame widths/s,
    below the 0.2/s threshold; 12 px = 0.28/s, above it)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    x = CX
    for k in range(fps * 4):
        x += px_per_frame if k >= fps * 2 else 0
        writer.write(view(world, x, CY))
    writer.release()
    return path


def test_multi_person_extraction_records_camera_motion_and_tracks_carry_it(tmp_path, world):
    """The production path (extract_multi_pose -> tracking -> subject track) must keep motion;
    before integration it was silently dropped there."""
    from behaviorlens.tracking import MultiPoseSequence

    path = _pan_video(tmp_path / "pan.mp4", world)
    multi, _ = pipeline.extract_multi_pose(path, probe_video(path), _NoPeople(), 5.0)
    assert multi.camera_motion is not None and multi.camera_motion.shape == (len(multi), MOTION_CHANNELS)
    dx = multi.camera_motion[:, MOTION_DX]
    assert np.allclose(dx[multi.timestamps_s > 2.1], -18 / W, atol=0.004)
    _, tracked, seq = pipeline.analyze_multi_pose(multi, P, "s")
    assert seq.camera_motion is not None and np.array_equal(seq.camera_motion, multi.camera_motion, equal_nan=True)
    # npz round trip, and old multi caches without motion still load
    multi.save_npz(tmp_path / "m.npz")
    again = MultiPoseSequence.load_npz(tmp_path / "m.npz")
    assert np.allclose(again.camera_motion, multi.camera_motion, equal_nan=True)
    MultiPoseSequence(timestamps_s=multi.timestamps_s, frame_indices=multi.frame_indices, poses=multi.poses,
                      counts=multi.counts, sample_fps=5.0, duration_s=multi.duration_s).save_npz(tmp_path / "old.npz")
    assert MultiPoseSequence.load_npz(tmp_path / "old.npz").camera_motion is None


def test_run_analysis_reports_camera_motion_on_the_tracking_path(store, tmp_path, world):
    import io

    path = _pan_video(tmp_path / "pan.mp4", world, px_per_frame=12)
    data = path.read_bytes()
    sid = pipeline.create_upload_session(store, "pan.mp4", io.BytesIO(data), len(data))
    res = pipeline.run_analysis(store, sid, estimator_factory=_NoPeople)
    assert res["stats"]["camera_motion_available"] is True
    assert res["stats"]["camera_moving_ratio"] > 0.3


def test_every_detected_person_is_excluded_from_the_background_fit(world):
    """Two people moving independently in front of a static camera: excluding both keeps the
    camera static; with one excluded the other's texture can read as camera motion."""
    est_all, est_one = CameraMotionEstimator(), CameraMotionEstimator()
    rows_all, rows_one = [], []
    for k in range(12):
        a, b = (0.05 + 0.02 * k, 0.2), (0.6 - 0.02 * k, 0.25)
        frame = draw_figure(draw_figure(view(world, CX, CY), *a, phase=k, w=0.3), *b, phase=k + 50, w=0.3)
        people = [figure_landmarks(*a, w=0.3), figure_landmarks(*b, w=0.3)]
        rows_all.append(est_all.update(frame, people))
        rows_one.append(est_one.update(frame, people[0]))
    dx_all = np.abs(np.stack(rows_all)[1:, MOTION_DX])
    assert np.nanmax(dx_all) < 0.003  # everyone excluded: the camera reads as still
