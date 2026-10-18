"""Drive the BehaviorLens demo review flow in headless Chromium and report problems.

Usage: python drive_demo.py <url> <out-dir> [--vp9 <data-dir>]

--vp9 swaps the demo session's preview for a VP9/WebM copy in the *scratch* DB, because
Playwright's Chromium has no H.264 decoder (docs/LEARNINGS.md §2). Use it only to verify
playback and seeking; never on real data.

Exit code 1 if any Streamlit exception or failed check is found.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def swap_to_vp9(data_dir: Path) -> None:
    import imageio_ffmpeg

    db = sqlite3.connect(data_dir / "behaviorlens.db")
    for sid, rel in db.execute("SELECT session_id, preview_relpath FROM sessions WHERE data_mode='synthetic'"):
        if not rel or rel.endswith(".webm"):
            continue
        src = data_dir / rel
        dst = src.with_name(src.stem + "_vp9.webm")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                        "-c:v", "libvpx-vp9", "-b:v", "300k", "-deadline", "realtime", str(dst)], check=True)
        db.execute("UPDATE sessions SET preview_relpath=? WHERE session_id=?",
                   (str(Path(rel).with_name(dst.name)), sid))
    db.commit()


def main() -> int:
    url, out = sys.argv[1], Path(sys.argv[2])
    vp9_dir = Path(sys.argv[sys.argv.index("--vp9") + 1]) if "--vp9" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"checks": {}, "console_errors": []}
    ok = True

    def check(name: str, cond: bool, detail=None) -> None:
        nonlocal ok
        report["checks"][name] = {"ok": bool(cond), "detail": detail}
        ok &= bool(cond)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        page = browser.new_page(viewport={"width": 1440, "height": 1750})
        page.on("console", lambda m: report["console_errors"].append(m.text) if m.type == "error" else None)
        page.goto(url)
        page.wait_for_selector("text=Sessions", timeout=30000)
        time.sleep(2)
        page.get_by_role("button", name="Load synthetic demo session").click()
        page.wait_for_selector("text=Candidate observations timeline", timeout=30000)
        time.sleep(2)
        if vp9_dir:
            swap_to_vp9(vp9_dir)
            page.reload()
            page.wait_for_selector("text=Candidate observations timeline", timeout=30000)
            time.sleep(2)
        page.screenshot(path=str(out / "01_demo_loaded.png"))

        # Plotly's drag layer intercepts clicks: click the bar's centre like a user would.
        bars = page.locator(".js-plotly-plot .barlayer .point path")
        check("timeline_has_bars", bars.count() >= 5, bars.count())
        bb = bars.nth(0).bounding_box()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        time.sleep(3)
        check("bar_click_selects_event", page.get_by_text("Supporting evidence").count() == 1)
        check("thumbnails_shown", page.locator("[data-testid=stImage]").count() >= 2,
              page.locator("[data-testid=stImage]").count())
        page.screenshot(path=str(out / "02_event_selected.png"))

        page.get_by_role("button", name="✓ Confirm").click()
        time.sleep(2)
        page.get_by_role("button", name="Next pending").click()
        time.sleep(2)
        page.get_by_role("button", name="✗ Reject").click()
        time.sleep(2)
        page.screenshot(path=str(out / "03_after_review.png"))

        video_state = page.evaluate("""async () => { const v = document.querySelector('video');
            for (let i = 0; i < 40 && v.readyState < 1; i++) await new Promise(r => setTimeout(r, 250));
            return {readyState: v.readyState, currentTime: v.currentTime, error: v.error && v.error.code,
                    h264: v.canPlayType('video/mp4; codecs="avc1.42E01E"')}; }""")
        report["video"] = video_state
        if vp9_dir:
            check("video_plays_and_seeks", video_state["readyState"] >= 1 and video_state["error"] is None,
                  video_state)

        page.get_by_role("tab", name="Export").click()
        time.sleep(2)
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download CSV").click()
        csv_path = out / "export.csv"
        dl.value.save_as(str(csv_path))
        statuses = [line.split(",")[7] for line in csv_path.read_text().splitlines()[1:]]
        check("export_reflects_review", "confirmed" in statuses and "rejected" in statuses, statuses)

        page.get_by_role("tab", name="Analysis details").click()
        time.sleep(2)
        page.screenshot(path=str(out / "04_details.png"))
        exceptions = page.locator("[data-testid=stException]").count()
        check("no_streamlit_exceptions", exceptions == 0, exceptions)
        browser.close()

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
