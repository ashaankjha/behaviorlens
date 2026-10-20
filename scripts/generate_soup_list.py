"""Generate docs/regulatory/soup_list.md from constraints.txt and a pip-audit run.

Usage (from the repo root, inside the project venv):
    python scripts/generate_soup_list.py            # runs pip-audit (needs network)
    python scripts/generate_soup_list.py --no-audit # metadata only

Each pinned package is classified as runtime (reachable from requirements.txt) or
development/test only, with its licence and summary from installed metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "regulatory" / "soup_list.md"

# Why BehaviorLens depends on each direct dependency, and where the SOUP boundary sits.
PURPOSE = {
    "mediapipe": "Pose landmark inference (Pose Landmarker, VIDEO mode). Wrapped by `pose.py`. "
                 "**Inference-affecting.**",
    "opencv-python-headless": "Video decode, frame sampling, resizing, thumbnails (`video_io.py`, `pipeline.py`). "
                              "**Inference-affecting** (decoded pixels).",
    "opencv-contrib-python": "Installed by mediapipe; ships the same `cv2` module as the headless wheel (AUDIT D7).",
    "imageio-ffmpeg": "Bundled static ffmpeg binary for H.264 previews and overlays (`video_io.py`).",
    "numpy": "Landmark arrays and feature math (`features.py`, `event_rules.py`). **Inference-affecting.**",
    "pydantic": "Typed validation of `Event` and `VideoMeta` (`models.py`).",
    "streamlit": "Local review UI and HTTP server (`app.py`). Network-facing component.",
    "plotly": "Timeline chart in the UI.",
    "pandas": "Tables in the UI.",
    "pytest": "Test runner.",
    "pyarrow": "Charades metadata parsing in `scripts/fetch_validation_clips.py` only.",
    "ruff": "Lint (dev).",
    "mypy": "Static type checking (dev).",
    "pytest-cov": "Coverage measurement and floor (dev).",
    "pip-audit": "Dependency vulnerability scan (dev/CI).",
    "playwright": "Browser end-to-end check (`e2e-verify-streamlit` skill, dev).",
}
INFERENCE_AFFECTING = {"mediapipe", "opencv-python-headless", "opencv-contrib-python", "numpy"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[norm(name)] = ver.strip()
    return pins


def read_requirements(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            out.append(norm(Requirement(line).name))
    return out


def closure(roots: list[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            reqs = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for r in reqs:
            req = Requirement(r)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                stack.append(norm(req.name))
    return seen


def licence(name: str) -> str:
    try:
        md = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    lic = md.get("License-Expression") or ""
    if not lic:
        classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
        first_line = (md.get("License") or "").strip().splitlines()[:1]
        lic = ", ".join(classifiers) or (first_line[0][:40] if first_line else "")
    return (lic or "see package metadata").replace("|", "/")


def summary(name: str) -> str:
    try:
        return (metadata.metadata(name).get("Summary") or "").replace("|", "/")[:90]
    except metadata.PackageNotFoundError:
        return ""


def run_audit() -> tuple[dict[str, list[str]], str]:
    cmd = [sys.executable, "-m", "pip_audit", "-r", str(ROOT / "constraints.txt"), "--no-deps", "--disable-pip",
           "--format", "json", "--progress-spinner", "off"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}, f"pip-audit failed to run: {proc.stderr.strip()[:200]}"
    vulns = {norm(d["name"]): [v["id"] for v in d.get("vulns", [])] for d in data.get("dependencies", [])}
    n = sum(len(v) for v in vulns.values())
    return vulns, f"pip-audit {metadata.version('pip-audit')} on {date.today().isoformat()}: " + (
        "no known vulnerabilities" if n == 0 else f"{n} known vulnerabilities")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audit", action="store_true")
    args = ap.parse_args()
    pins = read_pins(ROOT / "constraints.txt")
    runtime = closure(read_requirements(ROOT / "requirements.txt")) - {"pytest", "pyarrow"}
    runtime |= {"opencv-contrib-python"} & set(pins)
    direct = set(read_requirements(ROOT / "requirements.txt")) | set(read_requirements(ROOT / "requirements-dev.txt"))
    vulns, audit_line = ({}, "pip-audit not run (--no-audit)") if args.no_audit else run_audit()

    lines = [
        "# SOUP list (software of unknown provenance)",
        "",
        "> **DRAFT — requires review by qualified regulatory/QA personnel; not a regulatory submission.**",
        "",
        "| Field | Value |", "|---|---|",
        "| Document ID | BL-REG-SOUP-001 |", "| Version | 0.1 (draft) |",
        f"| Date | {date.today().isoformat()} |", "| Status | Draft |", "| Owner | TBD (manufacturer) |",
        "",
        "Generated by `scripts/generate_soup_list.py` from `constraints.txt` (exact tested versions) and installed "
        "package metadata. **Do not edit by hand**; regenerate after any dependency change.",
        "",
        f"**Known-anomaly / CVE check:** {audit_line} (`pip-audit -r constraints.txt --no-deps --disable-pip`, "
        "OSV/PyPI advisory data). The same scan runs in CI on every push. A clean scan means no *published* "
        "advisory matched; it is not evidence that a component is defect-free. Functional anomalies of SOUP are "
        "not tracked by pip-audit; the manufacturer must review each runtime item's issue tracker and release "
        "notes (IEC 62304 §7.1.3) before release.",
        "",
        "Other SOUP outside pip's view: the CPython 3.11 interpreter and its `sqlite3` module (bundled SQLite), "
        "the static ffmpeg binary inside `imageio-ffmpeg`, the MediaPipe `pose_landmarker_lite` model weights "
        "(SHA-256 pinned in `config.MODEL_SHA256`), the system `libegl1`/`libgles2` libraries, and the "
        "`python:3.11-slim` base image used by the Dockerfile.",
        "",
        "Inference-affecting components (a version change requires re-running the real-video validation, "
        "`validate-real-video` skill, before release): "
        + ", ".join(f"`{n}`" for n in sorted(INFERENCE_AFFECTING)) + ".",
        "",
        "## Runtime components",
        "",
        "| Component | Version | Licence | Purpose in BehaviorLens | Known vulnerabilities |",
        "|---|---|---|---|---|",
    ]
    def row(name: str) -> str:
        purpose = PURPOSE.get(name) or (("Direct dependency. " if name in direct else "Transitive dependency. ")
                                        + summary(name))
        v = vulns.get(name)
        vtext = "not checked" if args.no_audit else (", ".join(v) if v else "none found")
        return f"| `{name}` | {pins[name]} | {licence(name)} | {purpose} | {vtext} |"

    lines += [row(n) for n in sorted(pins) if n in runtime]
    lines += ["", "## Development, test and validation tooling (not shipped in the Docker image)", "",
              "| Component | Version | Licence | Purpose in BehaviorLens | Known vulnerabilities |",
              "|---|---|---|---|---|"]
    lines += [row(n) for n in sorted(pins) if n not in runtime]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(pins)} components; {audit_line})")


if __name__ == "__main__":
    main()
