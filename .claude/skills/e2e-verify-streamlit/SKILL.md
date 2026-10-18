---
name: e2e-verify-streamlit
description: Launch the BehaviorLens Streamlit app on a throwaway data dir and drive the full review flow in headless Chromium with Playwright (load demo, click a timeline bar, check thumbnails, confirm/reject, export CSV, details tab), taking screenshots and failing on any Streamlit exception. Use after UI or pipeline changes, before pushing, or when asked for screenshots.
---

# End-to-end UI verification

Unit tests and `AppTest` can't click Plotly charts or play video. This skill runs the real app in a real browser.

## Steps

```bash
SCR="${SCRATCHPAD:?set SCRATCHPAD to a scratch dir}/e2e"   # always a fresh dir (M12); unset would mean /e2e (M23)
rm -rf "$SCR"
D=.claude/skills/e2e-verify-streamlit/scripts
URL=$($D/run_app.sh "$SCR/data" 8502)       # waits for /_stcore/health
.venv/bin/python $D/drive_demo.py "$URL" "$SCR/out" --vp9 "$SCR/data"
$D/stop_app.sh "$SCR/data"                  # pid-file based; never pkill -f "streamlit run"
```

- `drive_demo.py` prints a JSON report and exits 1 if any check fails. Its checks: `timeline_has_bars`, `bar_click_selects_event`, `thumbnails_shown`, `video_plays_and_seeks`, `export_reflects_review`, `no_streamlit_exceptions`.
- Screenshots land in `$SCR/out/0*.png`. Look at them with the Read tool before reporting success.
- To refresh README screenshots, copy only **synthetic-demo** screenshots into `docs/screenshots/`. Never commit screenshots of real people.

## Gotchas (details in docs/LEARNINGS.md)

- Playwright's Chromium **can't decode H.264**, so the app's previews show `MEDIA_ERR_SRC_NOT_SUPPORTED`. `--vp9` re-encodes the scratch demo preview to WebM so playback and seeking can be verified. One console "Video source error" from the first, H.264 load is expected.
- Plotly's drag layer swallows `locator.click()`. The driver clicks bar centres with `page.mouse.click`.
- Streamlit scrolls inside its own container, so `full_page` screenshots are cut off. The driver uses a 1440×1750 viewport.
- If the session selectbox jumps to another session after an interaction, check that labels are unique (M2).
- Chromium path: `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`. Don't run `playwright install`. Install the Python package with `.venv/bin/pip install playwright` if it's missing.

## Testing a real upload

Use `page.locator("input[type=file]").first.set_input_files(path)`, click **Create session from upload**, then **Run analysis**, and wait for the text "Analysis complete". Use a clip made from a public sample photo of an adult in a temp dir (see LEARNINGS §2), never real footage that you'd commit.
