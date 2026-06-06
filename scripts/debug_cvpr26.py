# noqa: INP001
r"""Define the code for CVPR 2026."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from candidex.scraper import CVFPaperScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger: logging.Logger = logging.getLogger(__name__)

logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)


def main() -> None:
    r"""Define the main function."""
    data_path = Path(__file__).parent.parent.joinpath("data/v1").joinpath("cvpr26").resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    paper_dir = data_path.joinpath("papers")
    author_dir = data_path.joinpath("authors")
    logger.info(f"paper_dir: {paper_dir} | author_dir: {author_dir}")

    scraper = CVFPaperScraper(venue="CVPR", year=2026, cache_dir=paper_dir)
    logger.info(scraper)
    df_papers = scraper.scrape()
    logger.info(df_papers)


if __name__ == "__main__":
    load_dotenv()
    main()
