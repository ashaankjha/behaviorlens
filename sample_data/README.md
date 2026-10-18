# sample_data

Only **synthetic** assets live here: no real people and no patient data.

| File | What it is |
|---|---|
| `demo_synthetic_pose.npz` | Programmatically generated pose landmarks (40 s at 5 fps) with scripted events, noise, an occlusion, a dropout and an out-of-frame period |
| `demo_ground_truth.json` | The scripted event timeline used for the agreement metrics |
| `demo_synthetic.mp4` | Stick-figure rendering of the same skeleton, for timeline playback |

Regenerate deterministically with `python scripts/generate_demo.py`. The demo session bypasses MediaPipe inference, so it does not validate real pose extraction.
