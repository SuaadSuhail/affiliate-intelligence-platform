"""
SEO rank-tracking data source for the SEO signal checker.

Mirrors src/scraping/fetcher.py's fixture-first, real-source-later shape:
for the mock fixture (kind="fixture"), reads a local JSON file shaped like a
real rank-tracking API's bulk position-tracking export — one entry per
tracked keyword, with current position, previous position, search volume,
a ranking URL, and a checked_at timestamp. Modeled on SEMrush's Position
Tracking API response shape (position / previous_position / search_volume /
url / date fields) so swapping in the real API later means adding a
fetch_seo_data(kind="live") branch, not rewriting the schema check_seo()
consumes downstream.

For a live API (kind="live"): NOT IMPLEMENTED YET — raises NotImplementedError.
LIVE_API_CONFIGURED stays False until a real branch exists; src.ingestion.
etl_pipeline.seed_demo_seo_scan() checks this flag before running a scan as
a side effect of routine ingestion (see that module for why).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.core.logging_config import get_logger

logger = get_logger(__name__)

FIXTURE_PATH: Path = (
    Path(__file__).parent.parent.parent / "data" / "mock" / "seo" / "rank_tracking_mock.json"
)

# Flip to True only once a real fetch_seo_data(kind="live") branch exists
# below. Checked by src.ingestion.etl_pipeline before folding a scan into
# routine ingestion — see that module's seed_demo_seo_scan().
LIVE_API_CONFIGURED = False


def fetch_seo_data(kind: str = "fixture") -> list[dict]:
    """
    Return a list of rank-tracking rows:
    [{"keyword": str, "position": int, "previous_position": int | None,
      "search_volume": int | None, "url": str, "checked_at": str (ISO8601)}, ...]

    kind="fixture" reads FIXTURE_PATH directly.
    kind="live" is not implemented — raises NotImplementedError until a real
    SEO API is configured (see module docstring).
    """
    if kind == "fixture":
        logger.info("reading SEO fixture file directly", extra={"path": str(FIXTURE_PATH)})
        with open(FIXTURE_PATH, encoding="utf-8") as f:
            return json.load(f)

    raise NotImplementedError(
        "Live SEO API fetch is not implemented yet — no real SEO API is configured. "
        "See src/seo/api_client.py module docstring."
    )
