"""
FastTracker Web GUI screenshot capture script.
Uses Playwright to automate the browser, take screenshots, and save as PNG.
"""
import asyncio
import os
import sys
import time

# Use the correct Python's playwright
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), 'screenshots')
BASE_URL = 'http://localhost:5000'

async def main():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use a large viewport to capture the full UI
        context = await browser.new_context(
            viewport={'width': 1600, 'height': 1000},
            device_scale_factor=2,  # Retina-quality
        )
        page = await context.new_page()

        # ---- 1. Full GUI overview ----
        print('[1/6] Navigating to FastTracker Web GUI...')
        await page.goto(BASE_URL, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, '01_full_gui.png'))
        print('  -> 01_full_gui.png captured')

        # ---- 2. Set launch point (Pyongyang area) & target (Tokyo area) ----
        print('[2/6] Setting launch/target points via presets...')
        # We'll use JavaScript to directly set points since map clicks are complex
        await page.evaluate("""() => {
            setLaunchPoint(39.04, 125.76);   // Pyongyang area
            setTargetPoint(35.68, 139.69);   // Tokyo area
            setSensorPoint(34.79, 131.13);   // Sensor between
        }""")
        await page.wait_for_timeout(1500)
        # Fit map to show all points
        await page.evaluate("""() => {
            if (launchPoint && targetPoint) {
                map.fitBounds([
                    [launchPoint.lat, launchPoint.lon],
                    [targetPoint.lat, targetPoint.lon]
                ], { padding: [40, 40] });
            }
        }""")
        await page.wait_for_timeout(1000)
        await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, '02_map_points.png'))
        print('  -> 02_map_points.png captured')

        # ---- 3. Control panel - TARGET ----
        print('[3/6] Capturing Target panel...')
        # Target is already active by default
        ctrl_panel = page.locator('#control-panel')
        await ctrl_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '03_target_panel.png'))
        print('  -> 03_target_panel.png captured')

        # ---- 4. Control panel - SENSOR ----
        print('[4/6] Capturing Sensor panel...')
        await page.click('text=Sensor')
        await page.wait_for_timeout(500)
        await ctrl_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '04_sensor_panel.png'))
        print('  -> 04_sensor_panel.png captured')

        # ---- 5. Control panel - TRACKER ----
        print('[5/6] Capturing Tracker panel...')
        await page.click('text=Tracker')
        await page.wait_for_timeout(500)
        await ctrl_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '05_tracker_panel.png'))
        print('  -> 05_tracker_panel.png captured')

        # ---- 6. Generate trajectory & capture ----
        print('[6/6] Generating trajectory...')
        # Switch back to Target panel for the button
        await page.click('text=Target')
        await page.wait_for_timeout(300)

        # Click Generate Trajectory
        await page.click('#btn-generate')
        # Wait for loading overlay to disappear (up to 60s)
        await page.wait_for_selector('.loading.active', state='hidden', timeout=60000)
        await page.wait_for_timeout(2000)

        # Capture the trajectory 3D plot
        viz_panel = page.locator('#viz-panel')
        await viz_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '06_trajectory_3d.png'))
        print('  -> 06_trajectory_3d.png captured')

        # ---- 7. Run tracker & capture tracking results ----
        print('[7] Running tracker...')
        await page.click('#btn-run-tracker')
        await page.wait_for_selector('.loading.active', state='hidden', timeout=120000)
        await page.wait_for_timeout(2000)

        # Click Tracking tab
        await page.click('[data-viz-tab="tracking"]')
        await page.wait_for_timeout(2000)
        await viz_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '07_tracking_3d.png'))
        print('  -> 07_tracking_3d.png captured')

        # ---- 8. Evaluation tab ----
        print('[8] Capturing Evaluation tab...')
        await page.click('[data-viz-tab="evaluation"]')
        await page.wait_for_timeout(1500)
        await viz_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '08_evaluation.png'))
        print('  -> 08_evaluation.png captured')

        # ---- 9. Timeline tab ----
        print('[9] Capturing Timeline tab...')
        await page.click('[data-viz-tab="timeline"]')
        await page.wait_for_timeout(1500)
        await viz_panel.screenshot(path=os.path.join(SCREENSHOTS_DIR, '09_timeline.png'))
        print('  -> 09_timeline.png captured')

        # ---- 10. Full page after simulation ----
        print('[10] Capturing full page after simulation...')
        # Switch back to Tracking tab for full view
        await page.click('[data-viz-tab="tracking"]')
        await page.wait_for_timeout(1000)
        await page.screenshot(path=os.path.join(SCREENSHOTS_DIR, '10_full_after_sim.png'))
        print('  -> 10_full_after_sim.png captured')

        await browser.close()

    print(f'\nAll screenshots saved to {SCREENSHOTS_DIR}/')
    # List files
    for f in sorted(os.listdir(SCREENSHOTS_DIR)):
        size = os.path.getsize(os.path.join(SCREENSHOTS_DIR, f))
        print(f'  {f}: {size/1024:.0f} KB')

if __name__ == '__main__':
    asyncio.run(main())
