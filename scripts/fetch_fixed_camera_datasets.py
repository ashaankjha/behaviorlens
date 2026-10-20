"""Download fixed-camera, temporally labelled datasets of consenting adults into
data/validation/<dataset>/ in the shared format of scripts/datasets.py (gitignored).

Sources (academic / research use only; videos, frames and derived files are NEVER
committed or redistributed):

* utkinect: UTKinect-Action3D (Xia, Chen & Aggarwal, CVPRW 2012),
  https://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html. One stationary Kinect, 10 adult
  subjects x 2 sequences of 10 actions (walk, sitDown, standUp, pickUp, carry, throw, push,
  pull, waveHands, clapHands). RGB is a 1.8 GB zip of 480x640 JPGs, saved only while the
  skeleton was tracked, so frame numbers have gaps.

  Timing: frame numbers are always even and advance at 60 per second. The zip's per-file
  modification times are linear in the frame number at 58-60 numbers/s in all 20 sequences
  (residual <= 2.5 s, the zip timestamp resolution), i.e. 30 fps capture with the counter
  advancing by 2 per frame; so t = (frame - first_frame) / 60. This also matches the
  dataset page ("framerate 30 f/s ... final frame rate about 15 f/sec": 13 fps effective).
  Each sequence becomes a constant 30 fps MP4 in which every recorded image is held until
  the next recorded frame number (ffmpeg concat with per-frame durations), so video time
  equals capture time and untracked gaps show the last recorded image.

* pkummd1 / pkummd2: PKU-MMD Phase 1 / Phase 2 (Liu et al., 2017),
  https://struct002.github.io/PKUMMD/ (licence CC BY-NC 4.0). Kinect v2, three fixed views;
  only the middle (front, "M") view is used. Phase 1: 359 sequences of 3-4 min, 51 actions;
  Phase 2: 333 shorter sequences. Labels are 30 fps frame ranges. Raw 1080p AVIs
  (~98 MB / ~22 MB each) are transcoded to 960x540, 10 fps H.264 MP4 and then deleted, so
  Phase 1 needs ~2 GB of disk instead of ~35 GB. Pose models resize frames to <= 256 px
  internally and validation samples at 5 fps, so neither change affects the landmarks
  materially.

* pkummd1_fresh: the PKU-MMD Phase 1 front-view videos NOT in the `--limit/--stride` subset
  (the complement of `pkummd1 --limit 85 --stride 4` when run as
  `pkummd1_fresh --limit 85 --stride 4`). Same source, labels and transcoding as pkummd1,
  stored as its own dataset (data/validation/pkummd1_fresh/), so it can be held back as a
  one-shot confirmation set (docs/VALIDATION.md, "Frontal posture and track merging").

Usage:
    python scripts/fetch_fixed_camera_datasets.py utkinect pkummd1 pkummd2 [--limit N] [--stride N] [--workers 4]

The evaluated PKU-MMD Phase 1 subset is `pkummd1 --limit 85 --stride 4` (see docs/VALIDATION.md);
its complement is `pkummd1_fresh --limit 85 --stride 4`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets  # noqa: E402

from behaviorlens import config  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (behaviorlens-validation)"}

UTK_BASE = "https://cvrc.ece.utexas.edu/KinectDatasets"
UTK_MAP = {"standUp": "standing_transition", "sitDown": "sitting_down"}
# waveHands -> hand_raised only after viewing frames (see docs/VALIDATION.md); set by UTK_WAVE.
UTK_WAVE = "hand_raised"
UTK_UNITS_PER_S = 60.0

# Drive folders (public, "anyone with the link"); Phase 1 needs the resourcekey published on
# the mirror page http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html.
PKU = {
    "pkummd1": {
        "rgb_folder": ("0B20a4UzO-OyMbUhxV3FzSUlRNXc", "0-tlHTUWjVR1nlCkoD70tV4Q"),
        "labels": ("0B20a4UzO-OyMbExFbThVUzJLdzQ", "0-gZzIdIv4_nDvyD7DswNFoQ"),
        "name": re.compile(r"^(\d{4})-M\.avi$"),
    },
    "pkummd2": {
        "rgb_folder": ("1fC-vLUjEZWQrhlMdY8ps6AtZiAK_RnQK", ""),
        "labels": ("1m4-gQbbsd9bJIpeU95bPQ8J1UGGoQfYo", ""),
        "name": re.compile(r"^(A\d\dN\d\d)-M_color\.avi$"),
    },
}
PKU_FPS = 30.0
# Action ids from Actions.xlsx (identical in both phases).
PKU_ACTIONS = {
    1: "bow", 2: "brushing hair", 3: "brushing teeth", 4: "check time (from watch)", 5: "cheer up",
    6: "clapping", 7: "cross hands in front (say stop)", 8: "drink water", 9: "drop",
    10: "eat meal/snack", 11: "falling", 12: "giving something to other person", 13: "hand waving",
    14: "handshaking", 15: "hopping (one foot jumping)", 16: "hugging other person", 17: "jump up",
    18: "kicking other person", 19: "kicking something", 20: "make a phone call/answer phone",
    21: "pat on back of other person", 22: "pickup", 23: "playing with phone/tablet",
    24: "point finger at the other person", 25: "pointing to something with finger",
    26: "punching/slapping other person", 27: "pushing other person", 28: "put on a hat/cap",
    29: "put something inside pocket", 30: "reading", 31: "rub two hands together", 32: "salute",
    33: "sitting down", 34: "standing up", 35: "take off a hat/cap", 36: "take off glasses",
    37: "take off jacket", 38: "take out something from pocket", 39: "taking a selfie",
    40: "tear up paper", 41: "throw", 42: "touch back (backache)",
    43: "touch chest (stomachache/heart pain)", 44: "touch head (headache)",
    45: "touch neck (neckache)", 46: "typing on a keyboard",
    47: "use a fan (with hand or paper)/feeling warm", 48: "wear jacket", 49: "wear on glasses",
    50: "wipe face", 51: "writing",
}
PKU_INTERACTION = {12, 14, 16, 18, 21, 24, 26, 27}
# Mappings checked by viewing frames (docs/VALIDATION.md): "hand waving" raised one hand above
# the shoulder in every viewed instance; "cheer up" did not consistently (arms overhead ~2 s,
# ~0.5 s, or never above the shoulders), so it stays "other".
PKU_MAP = {34: "standing_transition", 33: "sitting_down", 13: "hand_raised"}


def vdir() -> Path:
    return config.data_dir() / "validation"


def raw_dir(name: str) -> Path:
    d = config.data_dir() / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
        return r.read()


def _download(url: str, dest: Path) -> None:
    """Stream to dest (via a .part file, so an interrupted download is never mistaken for a
    complete one)."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):  # Drive answers bursts with HTTP 500/429; back off and retry
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, \
                    part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.rename(dest)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** (attempt + 1))


def drive_file_url(file_id: str, resource_key: str = "") -> str:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    return url + (f"&resourcekey={resource_key}" if resource_key else "")


def list_drive_folder(folder_id: str, resource_key: str = "") -> dict[str, tuple[str, str]]:
    """name -> (file id, resource key) for a public Drive folder (no API key needed)."""
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    page = _get(url + (f"&resourcekey={resource_key}" if resource_key else "")).decode()
    out = {}
    for m in re.finditer(r'<a href="(https://drive.google.com/[^"]*)"[^>]*>.*?flip-entry-title">([^<]*)<',
                         page, re.S):
        link = html.unescape(m.group(1))
        fid = re.search(r"/d/([^/]+)/", link)
        if fid:
            rk = re.search(r"resourcekey=([^&]+)", link)
            out[html.unescape(m.group(2))] = (fid.group(1), rk.group(1) if rk else "")
    return out


def extract_7z(archive: Path, dest: Path) -> None:
    exe = shutil.which("7z") or shutil.which("7za")
    if exe:
        subprocess.run([exe, "x", "-y", f"-o{dest}", str(archive)], check=True, stdout=subprocess.DEVNULL)
        return
    try:
        import py7zr  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("need 7z (apt-get install p7zip-full) or py7zr in a throwaway venv to unpack labels")
    py7zr.SevenZipFile(archive).extractall(dest)


def write_annotations(name: str, records: list[dict]) -> None:
    datasets.validate_records(records, name)
    (vdir() / name / "annotations.json").write_text(json.dumps(records, indent=1))
    counts = datasets.event_counts(records)
    print(f"{name}: {len(records)} videos, events {counts}")


# ---------------------------------------------------------------- UTKinect

def parse_utk_labels(text: str) -> dict[str, dict[str, tuple[int, int] | None]]:
    out: dict[str, dict[str, tuple[int, int] | None]] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"s\d\d_e\d\d", line):
            cur = out.setdefault(line, {})
        elif ":" in line and cur is not None:
            k, v = line.split(":", 1)
            nums = v.split()
            cur[k.strip()] = (int(nums[0]), int(nums[1])) if len(nums) >= 2 and "NaN" not in nums else None
    return out


def utk_frame_durations(frames: list[int], units_per_s: float = UTK_UNITS_PER_S) -> list[float]:
    """Seconds each recorded image is held: until the next recorded frame number. The last
    image gets one nominal frame (2 numbers)."""
    return [(b - a) / units_per_s for a, b in zip(frames, frames[1:])] + [2 / units_per_s]


def fetch_utkinect() -> None:
    raw = raw_dir("utkinect")
    for f in ("actionLabel.txt", "RGB.zip"):
        if not (raw / f).is_file():
            print(f"downloading {f} ...")
            _download(f"{UTK_BASE}/{f}", raw / f)
    labels = parse_utk_labels((raw / "actionLabel.txt").read_text())
    dest = vdir() / "utkinect"
    (dest / "videos").mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(raw / "RGB.zip")
    by_seq: dict[str, list[tuple[int, str]]] = {}
    for n in zf.namelist():
        m = re.fullmatch(r"RGB/(s\d\d_e\d\d)/colorImg(\d+)\.jpg", n)  # skips "colorImg2040 - Copy.jpg"
        if m:
            by_seq.setdefault(m.group(1), []).append((int(m.group(2)), n))
    records = []
    for seq in sorted(by_seq):
        items = sorted(by_seq[seq])
        frames = [f for f, _ in items]
        out = dest / "videos" / f"{seq}.mp4"
        durations = utk_frame_durations(frames)
        if not out.is_file():
            with tempfile.TemporaryDirectory() as tmp:
                lines = []
                for i, ((_, name), d) in enumerate(zip(items, durations)):
                    p = Path(tmp) / f"{i:05d}.jpg"
                    p.write_bytes(zf.read(name))
                    lines += [f"file '{p}'", f"duration {d:.6f}"]
                lines.append(f"file '{Path(tmp) / f'{len(items) - 1:05d}.jpg'}'")  # concat quirk
                (Path(tmp) / "list.txt").write_text("\n".join(lines) + "\n")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                                "-i", str(Path(tmp) / "list.txt"), "-vf", "fps=30,format=yuv420p",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)],
                               check=True)
        f0 = frames[0]
        events = []
        for action, rng in labels.get(seq, {}).items():
            if rng is None:
                continue
            etype = UTK_MAP.get(action, UTK_WAVE if action == "waveHands" else "other")
            events.append({"type": etype, "start_s": round((rng[0] - f0) / UTK_UNITS_PER_S, 3),
                           "end_s": round((rng[1] - f0) / UTK_UNITS_PER_S, 3), "source_label": action})
        records.append({"video_id": seq, "video": f"videos/{seq}.mp4", "fps": 30.0,
                        "duration_s": round(sum(durations), 3), "camera": "fixed", "subjects": 1,
                        "events": sorted(events, key=lambda e: e["start_s"])})
    write_annotations("utkinect", records)


# ---------------------------------------------------------------- PKU-MMD

def parse_pku_label(text: str, fps: float = PKU_FPS) -> tuple[list[dict], int]:
    """Events from one PKU-MMD label file and the number of subjects implied by its actions."""
    events, interaction = [], False
    for line in text.splitlines():
        parts = [int(x) for x in line.replace(",", " ").split()]
        if len(parts) < 3:
            continue
        aid, s, e = parts[:3]
        interaction |= aid in PKU_INTERACTION
        events.append({"type": PKU_MAP.get(aid, "other"), "start_s": round(s / fps, 3),
                       "end_s": round(e / fps, 3), "source_label": PKU_ACTIONS.get(aid, str(aid))})
    return sorted(events, key=lambda ev: ev["start_s"]), 2 if interaction else 1


def transcode(src: Path, dst: Path) -> None:
    part = dst.with_name(dst.stem + ".part.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf", "fps=10,scale=960:-2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(part)], check=True)
    part.rename(dst)


def probe(path: Path) -> tuple[float, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate:format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    num, den = info["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den), float(info["format"]["duration"])


def subset_ids(labelled: list[str], limit: int | None, stride: int | None,
               complement: bool = False) -> list[str]:
    """Union of the first `limit` and every `stride`-th id (sorted); with `complement`, every
    other labelled id. No filter at all selects everything (and its complement is empty)."""
    chosen = set(labelled)
    if limit or stride:
        chosen = set(labelled[:limit or 0]) | set(labelled[::stride] if stride else [])
    return sorted(set(labelled) - chosen if complement else chosen)


def fetch_pkummd(name: str, limit: int | None, workers: int, stride: int | None = None) -> None:
    complement = name.endswith("_fresh")
    cfg = PKU[name.removesuffix("_fresh")]
    raw = raw_dir(name)
    lab_dir = raw / "labels"
    if not lab_dir.is_dir():
        archive = raw / "labels.7z"
        _download(drive_file_url(*cfg["labels"]), archive)
        extract_7z(archive, lab_dir)
    label_files = {p.stem: p for p in lab_dir.rglob("*-M.txt")}
    files = {m.group(1): ids for n, ids in list_drive_folder(*cfg["rgb_folder"]).items()
             if (m := cfg["name"].match(n))}
    labelled = sorted(v for v in files if f"{v}-M" in label_files)
    vids = subset_ids(labelled, limit, stride, complement)
    if complement and not (limit or stride):
        sys.exit(f"{name}: give the --limit/--stride of the subset whose complement you want")
    dest = vdir() / name
    (dest / "videos").mkdir(parents=True, exist_ok=True)

    def one(vid: str) -> str | None:
        out = dest / "videos" / f"{vid}.mp4"
        if out.is_file():
            return vid
        src = raw / f"{vid}-M.avi"
        try:
            if not src.is_file():
                _download(drive_file_url(*files[vid]), src)
            transcode(src, out)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"{name} {vid}: {exc}", file=sys.stderr)
            return None
        finally:
            src.unlink(missing_ok=True)
        return vid

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, vids))
    records = []
    # every labelled video on disk, so successive subset runs accumulate
    keep = set(vids) if complement else set(labelled)  # never let a fresh set absorb used videos
    for vid in (v for v in labelled if v in keep and (dest / "videos" / f"{v}.mp4").is_file()):
        fps, duration = probe(dest / "videos" / f"{vid}.mp4")
        events, subjects = parse_pku_label(label_files[f"{vid}-M"].read_text())
        records.append({"video_id": vid, "video": f"videos/{vid}.mp4", "fps": fps,
                        "duration_s": round(duration, 3), "camera": "fixed", "subjects": subjects,
                        "events": events})
    write_annotations(name, records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=["utkinect", "pkummd1", "pkummd2", "pkummd1_fresh"])
    ap.add_argument("--limit", type=int, help="first N videos per PKU-MMD phase (for a quick subset)")
    ap.add_argument("--stride", type=int, help="every N-th PKU-MMD video (sorted ids); unites with --limit")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for d in a.datasets:
        if d == "utkinect":
            fetch_utkinect()
        else:
            fetch_pkummd(d, a.limit, a.workers, a.stride)
