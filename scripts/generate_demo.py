"""Regenerate the SYNTHETIC demo fixture, ground truth and stick-figure video in sample_data/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402
from behaviorlens.demo_generator import DEMO_POSE, DEMO_TRUTH, DEMO_VIDEO, ensure_demo_assets  # noqa: E402

if __name__ == "__main__":
    out = config.sample_data_dir()
    for name in (DEMO_POSE, DEMO_TRUTH, DEMO_VIDEO):
        (out / name).unlink(missing_ok=True)
    for kind, path in ensure_demo_assets(out).items():
        print(f"{kind:6s} {path.relative_to(config.PROJECT_ROOT)} ({path.stat().st_size // 1024} KB)")
