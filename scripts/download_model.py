"""Download the MediaPipe Pose Landmarker (lite) model into assets/models/.

The model is distributed by Google under the terms in its model card:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker#models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behaviorlens import config  # noqa: E402
from behaviorlens.pose import download_model, model_version  # noqa: E402

if __name__ == "__main__":
    dest = config.model_path()
    if dest.is_file() and "--force" not in sys.argv:
        print(f"Model already present: {dest} ({model_version(dest)})")
    else:
        print(f"Downloading {config.MODEL_URL}")
        download_model(dest)
        print(f"Saved {dest} ({model_version(dest)})")
    print(f"License / model card: {config.MODEL_CARD_URL}")
