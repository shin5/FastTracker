"""Re-capture sensor and tracker panels with correct selectors."""
import asyncio, os
from playwright.async_api import async_playwright

DIR = os.path.join(os.path.dirname(__file__), 'screenshots')

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={'width': 1600, 'height': 1000}, device_scale_factor=2)
        page = await ctx.new_page()
        await page.goto('http://localhost:5000', wait_until='networkidle')
        await page.wait_for_timeout(1500)

        # Set points so the panel shows real data
        await page.evaluate("""() => {
            setLaunchPoint(39.04, 125.76);
            setTargetPoint(35.68, 139.69);
            setSensorPoint(34.79, 131.13);
        }""")
        await page.wait_for_timeout(1000)

        ctrl = page.locator('#control-panel')

        # Sensor panel
        await page.evaluate("() => switchPanel('sensor')")
        await page.wait_for_timeout(800)
        await ctrl.screenshot(path=os.path.join(DIR, '04_sensor_panel.png'))
        print('04_sensor_panel.png recaptured')

        # Tracker panel
        await page.evaluate("() => switchPanel('tracker')")
        await page.wait_for_timeout(800)
        await ctrl.screenshot(path=os.path.join(DIR, '05_tracker_panel.png'))
        print('05_tracker_panel.png recaptured')

        await browser.close()

asyncio.run(main())
