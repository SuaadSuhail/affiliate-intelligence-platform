"""
Site configuration for the promo leakage detector.

Defines which sites to scan, how to locate voucher codes in their markup,
and whether a site is a local fixture or a live target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

FIXTURES_DIR: Path = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class SiteConfig:
    name: str                       # unique slug, e.g. "voucherslug-mock"
    kind: str                       # "live" | "fixture"
    url: str                        # file:// path for fixtures, https:// for live
    code_selectors: list[str]       # CSS selectors that resolve to the code text
    merchant_selectors: list[str]   # CSS selectors for the merchant/store name


SITES: list[SiteConfig] = [
    SiteConfig(
        name="voucherslug-mock",
        kind="fixture",
        url=(FIXTURES_DIR / "voucherslug_mock.html").as_uri(),
        # Each .voucher-card holds one code in <code class="voucher-code">
        code_selectors=[".voucher-card .voucher-code"],
        merchant_selectors=[".voucher-card .merchant-name"],
    ),
    SiteConfig(
        name="dealsden-mock",
        kind="fixture",
        url=(FIXTURES_DIR / "dealsden_mock.html").as_uri(),
        code_selectors=["li.deal .deal-code"],
        merchant_selectors=["li.deal .store"],
    ),
    SiteConfig(
        name="csr-shell-mock",
        # kind="live" even though this is a local file: fetch_html(kind="fixture")
        # reads the file directly without a browser, which would return only the empty
        # <div id="app"></div> shell. Playwright can navigate file:// URLs just as well
        # as http:// ones, so kind="live" routes this through the browser render path
        # where csr_shell_mock.js executes and injects CSRLEAK99 into #app.
        kind="live",
        url=(FIXTURES_DIR / "csr_shell_mock.html").as_uri(),
        code_selectors=["#app .voucher-card .voucher-code"],
        merchant_selectors=["#app .voucher-card .merchant-name"],
    ),

    # ── Live targets ──────────────────────────────────────────────────────────
    # Before enabling any live site:
    #   1. Verify robots.txt allows automated access to the voucher listing path.
    #   2. Confirm the specific target URL and selector structure with the user.
    #   3. Add rate-limiting / polite delays in the fetcher.
    #
    # SiteConfig(
    #     name="example-live-site",
    #     kind="live",
    #     url="https://www.example.com/voucher-codes/",
    #     code_selectors=[".offer-code", ".promo-code span"],
    #     merchant_selectors=[".merchant-title"],
    # ),
]
