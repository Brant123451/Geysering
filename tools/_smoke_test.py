# -*- coding: utf-8 -*-
"""Headless smoke test for curve_digitizer.html v2 (#demo image).

Demo image geometry (image pixels): axes box (90,520)-(740,70), x 0..10, y 0..2.
v2 flow: calibration clicks open an inline popup for value entry;
after the 3rd point the tool auto-switches to trace mode; snap-to-curve is on.
"""
import math
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
URL = (HERE / "curve_digitizer.html").as_uri() + "#demo"
SCREENSHOT = Path(tempfile.gettempdir()) / "geysering_curve_digitizer_smoke.png"
errors = []

def screen_of(page, ix, iy):
    return page.evaluate(
        """([ix, iy]) => {
            const r = cv.getBoundingClientRect();
            const q = toScreen({x: ix, y: iy});
            return [r.left + q.x, r.top + q.y];
        }""", [ix, iy])

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 900})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    page.wait_for_function("typeof S !== 'undefined' && S.img !== null")
    page.wait_for_timeout(300)

    # after demo load the tool must auto-enter calibrate mode
    assert page.evaluate("S.mode") == "cal", "not in cal mode after image load"
    print("[ok] demo loaded, auto calibrate mode")

    # calibration point 1: origin -> popup -> x=0, y=0
    x, y = screen_of(page, 90, 520)
    page.mouse.click(x, y)
    page.wait_for_selector("#calpop", state="visible")
    page.fill("#cp-in1", "0"); page.fill("#cp-in2", "0")
    page.click("#cp-ok")
    # point 2: x-ref -> x=10
    x, y = screen_of(page, 740, 520)
    page.mouse.click(x, y)
    page.wait_for_selector("#calpop", state="visible")
    page.fill("#cp-in1", "10"); page.click("#cp-ok")
    # point 3: y-ref -> y=2
    x, y = screen_of(page, 90, 70)
    page.mouse.click(x, y)
    page.wait_for_selector("#calpop", state="visible")
    page.fill("#cp-in1", "2"); page.click("#cp-ok")
    assert page.evaluate("calReady()"), "calibration not ready"
    assert page.evaluate("S.mode") == "trace", "did not auto-switch to trace mode"
    print("[ok] 3-point calibration via popups, auto-switched to trace")

    # trace 4 points on the sine curve (snap on)
    clicked = []
    for xd in [1.0, 3.0, 5.0, 8.0]:
        yd = 1 + 0.8 * math.sin(0.6 * xd)
        ix = 90 + (740 - 90) * xd / 10.0
        iy = 520 + (70 - 520) * yd / 2.0
        x, y = screen_of(page, ix + 1.5, iy - 1.5)   # deliberately offset; snap should fix
        page.mouse.click(x, y)
        clicked.append((xd, yd))
    n = page.evaluate("S.series[0].pts.length")
    assert n == 4, f"expected 4 points, got {n}"
    data = page.evaluate("S.series[0].pts.map(p => imgToData(p))")
    for (xd, yd), d in zip(clicked, data):
        dx, dy = abs(d["x"] - xd), abs(d["y"] - yd)
        assert dx < 0.08 and dy < 0.03, f"coord error: clicked ({xd},{yd:.3f}) got ({d['x']:.3f},{d['y']:.3f})"
    print("[ok] 4 points traced with snap, coords accurate:",
          [(round(d['x'], 3), round(d['y'], 3)) for d in data])

    # point table shows 4 rows
    rows = page.locator("#pttable tr[data-pi]").count()
    assert rows == 4, f"point table rows = {rows}"
    print("[ok] point table lists 4 points")

    # undo / redo
    page.keyboard.press("Control+z")
    assert page.evaluate("S.series[0].pts.length") == 3, "undo failed"
    page.keyboard.press("Control+y")
    assert page.evaluate("S.series[0].pts.length") == 4, "redo failed"
    print("[ok] undo/redo works")

    # right-click deletes nearest point
    x, y = screen_of(page, *[(90 + 65 * 8), 520 + (70 - 520) * (1 + 0.8 * math.sin(4.8)) / 2][0:2]) if False else screen_of(page, 90 + 65 * 8, 520 + (70 - 520) * (1 + 0.8 * math.sin(4.8)) / 2)
    page.mouse.click(x, y, button="right")
    assert page.evaluate("S.series[0].pts.length") == 3, "right-click delete failed"
    print("[ok] right-click delete works")

    # wheel zoom
    s0 = page.evaluate("S.view.s")
    cx, cy = screen_of(page, 400, 300)
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, -240)
    page.wait_for_timeout(100)
    assert page.evaluate("S.view.s") > s0, "wheel zoom failed"
    print("[ok] wheel zoom works")

    # export rows exist for all points
    nrows = page.evaluate("exportRows().length")
    assert nrows == 3, f"exportRows length {nrows}"
    print("[ok] export produces rows")

    page.screenshot(path=str(SCREENSHOT))
    b.close()

if errors:
    print("[FAIL] console errors:", errors)
    sys.exit(1)
print("[ok] no console errors")
print("ALL PASS")
