"""
HTML fetcher for the promo leakage detector.

For SSR fixture files (kind="fixture"): reads the local HTML file directly —
no browser needed since the content is already present in the static markup.

For live sites and the CSR fixture (kind="live"): launches headless Chromium
via Playwright so JS-rendered content is fully hydrated before HTML is returned.
Playwright also handles file:// URLs, which is how the csr_shell_mock fixture
is served — its kind is "live" in site_config so the JS executes.
"""

from __future__ import annotations

import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

from src.core.logging_config import get_logger

logger = get_logger(__name__)

USER_AGENT = (
    "RevWiseLeakageBot/1.0 (+affiliate-integrity-check; "
    "contact=leakage-bot@revwise-placeholder.example)"
)

# Per-host rate limiter state: netloc -> monotonic timestamp of last request.
# Only consulted for kind="live" with http/https URLs.
_last_request: dict[str, float] = {}
_MIN_GAP_SECONDS: float = 3.0


class RobotsDisallowedError(Exception):
    """Raised when robots.txt disallows crawling the target URL."""


def _check_robots(url: str) -> None:
    """
    Verify that USER_AGENT is permitted to fetch url according to robots.txt.

    Fails closed: if robots.txt cannot be fetched or parsed for any reason,
    the URL is treated as disallowed and a warning is logged before raising.
    file:// URLs are silently skipped (no robots.txt concept for local files).

    Raises RobotsDisallowedError if crawling is not permitted.
    """
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return

    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)

    try:
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
    except Exception as exc:
        logger.warning(
            "robots.txt fetch failed — treating as disallowed (fail closed)",
            extra={"url": url, "robots_url": robots_url, "error": str(exc)},
        )
        raise RobotsDisallowedError(
            f"Could not fetch robots.txt for {url} — treating as disallowed"
        ) from exc

    logger.info(
        "robots.txt checked",
        extra={"url": url, "robots_url": robots_url, "allowed": allowed},
    )
    if not allowed:
        raise RobotsDisallowedError(f"robots.txt disallows crawling: {url}")


def _rate_limit(host: str) -> None:
    """
    Block until at least _MIN_GAP_SECONDS have elapsed since the last request
    to host. Updates the timestamp after the wait.

    Only called for kind="live" with http/https URLs.
    """
    now = time.monotonic()
    last = _last_request.get(host)
    if last is not None:
        elapsed = now - last
        if elapsed < _MIN_GAP_SECONDS:
            wait = _MIN_GAP_SECONDS - elapsed
            logger.info(
                "rate limiting — sleeping before next request",
                extra={"host": host, "sleep_seconds": round(wait, 2)},
            )
            time.sleep(wait)
    _last_request[host] = time.monotonic()


def fetch_html(url: str, kind: str = "live") -> str:
    """
    Fetch and return the fully rendered HTML for a given URL.

    Parameters
    ----------
    url  : A file:// URI (for fixtures) or https:// URL (for live sites).
    kind : "fixture" — read the local file directly (SSR static content only;
                        use only when no JS is needed to reveal the codes).
           "live"    — launch headless Chromium via Playwright so JS-rendered
                        content is hydrated before HTML is returned. Required
                        for all live http(s) targets and for csr_shell_mock
                        (a local file:// fixture that depends on JS execution).

    Returns
    -------
    Raw HTML string of the fully rendered page.

    Raises
    ------
    RobotsDisallowedError : robots.txt forbids crawling the URL (live only).
    playwright.sync_api.TimeoutError : page did not load within 10 seconds.
    """
    if kind == "fixture":
        path = Path(urlparse(url).path)
        logger.info("reading fixture file directly", extra={"path": str(path)})
        return path.read_text(encoding="utf-8")

    # kind == "live": Playwright render path
    parsed = urlparse(url)
    is_http = parsed.scheme in ("http", "https")

    if is_http:
        _check_robots(url)
        _rate_limit(parsed.netloc)

    from playwright.sync_api import sync_playwright

    logger.info("launching Playwright", extra={"url": url})
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=10_000)
                html = page.content()
                logger.info(
                    "Playwright fetch complete",
                    extra={"url": url, "html_length": len(html)},
                )
                return html
            finally:
                browser.close()
    except Exception as exc:
        logger.error(
            "Playwright fetch failed",
            extra={"url": url, "error": str(exc)},
        )
        raise
