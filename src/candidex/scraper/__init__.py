r"""Contain code to scrape the papers."""

from __future__ import annotations

__all__ = ["BasePaperScraper", "CVFPaperScraper"]

from candidex.scraper.base import BasePaperScraper
from candidex.scraper.cvf import CVFPaperScraper
