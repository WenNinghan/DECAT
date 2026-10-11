#!/usr/bin/env python3
"""Visit the public DECAT app, optionally wake it, and verify its input form.

Install requirements.txt and run ``python -m playwright install chromium`` first.
This script never enters a compound or clicks Run Prediction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_URL = "https://decat-ozone.streamlit.app/"
APP_TITLE = "Ozone Reaction Rate Constant (logk) Prediction System"
WAKE_BUTTON = "Yes, get this app back up!"
READY_DWELL_SECONDS = 3.0
BLOCKED_TEXT = re.compile(
    r"this app has (?:gone over|exceeded) its resource limits|"
    r"\boh no\b|error running (?:your|this) app|"
    r"you (?:do not|don't) have access to this app|"
    r"(?:sign|log) in to (?:view|access|continue)|"
    r"verify (?:that )?you are human|checking your browser|"
    r"connection (?:error|failed)|unable to connect to streamlit|"
    r"model loading failed|(?:ModuleNotFoundError|ImportError|RuntimeError):",
    re.IGNORECASE,
)


class BlockedPage(Exception):
    """A visible resource, authentication, security, or application error."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timeout_value(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 5 <= seconds <= 180:
        raise argparse.ArgumentTypeError("must be between 5 and 180 seconds")
    return seconds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Public app URL; override for local fixtures.")
    parser.add_argument("--output-dir", type=Path, default=Path("activation-artifacts"))
    parser.add_argument("--timeout-seconds", type=timeout_value, default=180.0,
                        help="Maximum navigation and readiness time per attempt (5–180).")
    parser.add_argument("--attempts", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
        parser.error("--url must be an HTTP(S) URL without embedded credentials")
    return args


async def first_visible(locator):
    for element in await locator.all():
        if await element.is_visible():
            return element
    return None


async def inspect_page(page):
    """Look inside every frame, including Community Cloud's streamlitApp iframe."""
    ready_frame = None
    wake_button = None
    for frame in page.frames:
        if frame.is_detached():
            continue
        try:
            error = await first_visible(frame.locator(
                '[data-testid="stException"], [data-testid="stAlertContentError"]'
            ))
            if error is None:
                error = await first_visible(frame.get_by_text(BLOCKED_TEXT))
            if error is not None:
                detail = (await error.inner_text(timeout=1000)).strip()
                raise BlockedPage(detail[:600] or "Visible application error")
            if await first_visible(frame.locator('input[type="password"]')) is not None:
                raise BlockedPage("A login form is visible; anonymous access is required")
            title = await first_visible(frame.get_by_text(APP_TITLE, exact=True))
            chemical = await first_visible(frame.get_by_role(
                "textbox", name="Chemical name or CAS", exact=True
            ))
            prediction = await first_visible(frame.get_by_role(
                "button", name="Run Prediction", exact=True
            ))
            # The prediction button is intentionally disabled until a compound is entered.
            if title is not None and chemical is not None and prediction is not None:
                if await chemical.is_enabled():
                    ready_frame = frame
            if wake_button is None:
                wake_button = await first_visible(frame.get_by_role(
                    "button", name=WAKE_BUTTON, exact=True
                ))
        except BlockedPage:
            raise
        except Exception:
            # Frame replacement while the app wakes is transient; other errors are not.
            if frame.is_detached():
                continue
            raise
    return ready_frame, wake_button


async def wait_for_ready(page, attempt, result):
    stable_since = None
    while True:
        ready_frame, wake_button = await inspect_page(page)
        if wake_button is not None:
            stable_since = None
            if not result["wake_click_attempted"]:
                # Do not repeat an ambiguous click, including across retry attempts.
                result["wake_click_attempted"] = True
                attempt["wake_click_attempted"] = True
                await wake_button.click(timeout=5000)
                attempt["wake_clicks"] += 1
                result["wake_clicks"] += 1
        elif ready_frame is not None:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= READY_DWELL_SECONDS:
                attempt["ready_frame_url"] = ready_frame.url
                return
        else:
            stable_since = None
        await asyncio.sleep(0.5)


async def run_attempt(browser, args, number, result):
    started = time.monotonic()
    attempt = {"number": number, "started_at": utc_now(), "status": "failed",
               "wake_click_attempted": False, "wake_clicks": 0, "retryable": True}
    context = None
    page = None

    async def visit():
        nonlocal context, page
        attempt["stage"] = "create_context"
        context = await browser.new_context(viewport={"width": 1440, "height": 1080})
        page = await context.new_page()
        page.set_default_timeout(5000)
        attempt["stage"] = "navigate"
        response = await page.goto(args.url, wait_until="domcontentloaded",
                                   timeout=min(60000, args.timeout_seconds * 1000))
        if response is not None:
            attempt["http_status"] = response.status
            if response.status >= 400:
                raise BlockedPage(f"Navigation returned HTTP {response.status}")
        attempt["stage"] = "wait_for_form"
        await wait_for_ready(page, attempt, result)

    try:
        await asyncio.wait_for(visit(), timeout=args.timeout_seconds)
        attempt.update(status="success", reason="ready", retryable=False)
    except BlockedPage as exc:
        attempt.update(reason="blocked_page", error=str(exc), retryable=False)
    except asyncio.TimeoutError:
        attempt.update(reason="timeout", error="The form did not remain ready before the deadline")
    except Exception as exc:
        attempt.update(reason="browser_error", error=f"{type(exc).__name__}: {exc}"[:1200])
    finally:
        if page is not None:
            attempt["final_url"] = page.url
            screenshot = args.output_dir / f"attempt-{number}.png"
            try:
                await page.screenshot(path=str(screenshot), full_page=True, timeout=10000)
                attempt["screenshot"] = screenshot.name
            except Exception as exc:
                attempt["screenshot_error"] = f"{type(exc).__name__}: {exc}"[:600]
                if attempt["status"] == "success":
                    attempt.update(status="failed", reason="screenshot_error", retryable=True)
        if context is not None:
            try:
                await asyncio.wait_for(context.close(), timeout=5)
            except Exception as exc:
                attempt["cleanup_error"] = f"{type(exc).__name__}: {exc}"[:600]
                if attempt["status"] == "success":
                    attempt.update(status="failed", reason="cleanup_error", retryable=False)
        attempt["duration_seconds"] = round(time.monotonic() - started, 2)
        attempt["finished_at"] = utc_now()
    return attempt


async def activate(args, result):
    playwright = None
    browser = None
    try:
        from playwright.async_api import async_playwright

        playwright = await asyncio.wait_for(async_playwright().start(), timeout=30)
        browser = await playwright.chromium.launch(headless=True, timeout=30000)
        result["browser_version"] = browser.version
        for number in range(1, args.attempts + 1):
            attempt = await run_attempt(browser, args, number, result)
            result["attempts"].append(attempt)
            print(f"Attempt {number}: {attempt['status']} ({attempt['reason']})", flush=True)
            result["status"] = attempt["status"]
            result["reason"] = attempt["reason"]
            if attempt["status"] == "success" or not attempt["retryable"]:
                break
            if number < args.attempts:
                await asyncio.sleep(2)
    except Exception as exc:
        result.update(status="failed", reason="browser_setup_error",
                      error=f"{type(exc).__name__}: {exc}"[:1200])
    finally:
        for name, resource in (("browser", browser), ("playwright", playwright)):
            if resource is not None:
                try:
                    close = resource.close() if name == "browser" else resource.stop()
                    await asyncio.wait_for(close, timeout=10)
                except Exception as exc:
                    result.setdefault("cleanup_errors", []).append(f"{name}: {exc}"[:600])
                    result.update(status="failed", reason="cleanup_error")


def main():
    args = parse_args()
    started = time.monotonic()
    result = {"url": args.url, "started_at": utc_now(), "status": "failed",
              "reason": "not_started", "timeout_seconds": args.timeout_seconds,
              "max_attempts": args.attempts, "ready_dwell_seconds": READY_DWELL_SECONDS,
              "wake_click_attempted": False, "wake_clicks": 0, "attempts": []}
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(activate(args, result))
    except KeyboardInterrupt:
        result.update(status="failed", reason="interrupted")
    except Exception as exc:
        result.update(status="failed", reason="execution_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["finished_at"] = utc_now()
        result["duration_seconds"] = round(time.monotonic() - started, 2)
        try:
            report = args.output_dir / "status.json"
            report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{result['status']}: {result['reason']}; report: {report}", flush=True)
        except OSError as exc:
            print(f"Cannot write status report: {exc}", file=sys.stderr)
            result["status"] = "failed"
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
