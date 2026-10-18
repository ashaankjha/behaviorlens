"""Download real, moving-person validation clips into data/validation/ (gitignored).

Sources (research-use datasets; clips are NEVER committed or redistributed):

* charades: Charades (Sigurdsson et al., 2016), videos acted out at home by
  consenting crowd workers following scripts, with temporal action annotations.
  We use the public 126-clip sample hosted on Hugging Face
  (Aditya02/Charades-Action-Sequence-Sample). Label 154 "Someone is standing up
  from somewhere" and 151 "Someone is going from standing to sitting" give
  ground truth for standing_transition.
* ucf101: UCF101 (Soomro et al., 2012) action clips. Only selected classes are
  pulled from the 7 GB zip with HTTP range requests (no full download). Used as
  a clip-level check for hand_raised: JumpingJack/PullUps (arms overhead) versus
  JumpRope/BodyWeightSquats/WalkingWithDog (arms at or below shoulder height).

Usage:
    python scripts/fetch_validation_clips.py [--charades] [--ucf101] [--per-class N]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402

HF = "https://huggingface.co/datasets"
CHARADES_REPO = f"{HF}/Aditya02/Charades-Action-Sequence-Sample"
UCF_ZIP = f"{HF}/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"
UCF_POSITIVE = ("JumpingJack", "PullUps")
UCF_NEGATIVE = ("JumpRope", "BodyWeightSquats", "WalkingWithDog")
UA = {"User-Agent": "behaviorlens-validation/0.1"}


def out_dir() -> Path:
    d = config.data_dir() / "validation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_charades() -> None:
    dest = out_dir() / "charades"
    (dest / "videos").mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    rows = []
    for split in ("train", "validation", "test"):
        raw = _get(f"{CHARADES_REPO}/resolve/main/data/{split}-00000-of-00001.parquet")
        rows += pq.read_table(io.BytesIO(raw)).to_pylist()
    tree = json.loads(_get(f"{HF.replace('/datasets', '/api/datasets')}/Aditya02/"
                           "Charades-Action-Sequence-Sample/tree/main/videos"))
    available = {Path(x["path"]).stem for x in tree}
    meta = []
    for r in rows:
        if r["video_id"] not in available:
            continue
        meta.append({"video_id": r["video_id"], "length": r["length"], "scene": r["scene"],
                     "labels": r["labels"], "action_timings": r["action_timings"]})
        path = dest / "videos" / f"{r['video_id']}.mp4"
        if not path.is_file():
            path.write_bytes(_get(f"{CHARADES_REPO}/resolve/main/videos/{r['video_id']}.mp4"))
    (dest / "annotations.json").write_text(json.dumps(meta))
    print(f"charades: {len(meta)} clips in {dest}")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for zipfile)."""

    def __init__(self, url: str, block: int = 1 << 18):
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()  # signed CDN URL after redirects
            self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos, self.block, self.cache = 0, block, {}

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            self.cache[i] = _get(self.url, {"Range": f"bytes={start}-{end}"})
            if len(self.cache) > 256:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_ucf101(per_class: int) -> None:
    dest = out_dir() / "ucf101"
    dest.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(HttpRangeFile(UCF_ZIP))
    labels = {}
    for cls in UCF_POSITIVE + UCF_NEGATIVE:
        members = sorted(n for n in zf.namelist() if f"/{cls}/" in n and n.endswith(".avi"))
        # one clip per group (different people/scenes) for diversity
        by_group: dict[str, str] = {}
        for n in members:
            by_group.setdefault(Path(n).stem.split("_g")[1].split("_")[0], n)
        for n in list(by_group.values())[:per_class]:
            path = dest / Path(n).name
            if not path.is_file():
                path.write_bytes(zf.read(n))
            labels[path.name] = {"class": cls, "arms_overhead": cls in UCF_POSITIVE}
    (dest / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"ucf101: {len(labels)} clips in {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--charades", action="store_true")
    ap.add_argument("--ucf101", action="store_true")
    ap.add_argument("--per-class", type=int, default=8)
    a = ap.parse_args()
    if not (a.charades or a.ucf101):
        a.charades = a.ucf101 = True
    if a.charades:
        fetch_charades()
    if a.ucf101:
        fetch_ucf101(a.per_class)
