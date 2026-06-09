# noqa: INP001
r"""Define the code for CVPR 2026."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from coola.utils.format import str_indent, str_mapping
from dotenv import load_dotenv
from rich.logging import RichHandler

from candidex.paper import dataframe_to_papers, download_pdfs
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


@dataclass(frozen=True)
class PathConfig:
    """Centralises all directory paths used by the candidex pipeline.

    All paths are derived from a single root directory, making it easy
    to relocate the entire pipeline output by changing one path.

    Attributes:
        paper_dir: Directory where paper metadata is stored.
        author_dir: Directory where author data is stored.
        paper_pdf_dir: Directory where downloaded paper PDFs are stored.
            Nested under `paper_dir`.

    Example:
        >>> from pathlib import Path
        >>> config = PathConfig.from_base_dir(Path("data"))
        >>> config.paper_dir
        PosixPath('data/papers')
        >>> config.paper_pdf_dir
        PosixPath('data/papers/pdf')
    """

    paper_dir: Path
    author_dir: Path
    paper_pdf_dir: Path

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(asdict(self)))}\n)"

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> PathConfig:
        """Construct a `PathConfig` from a single root directory.

        All subdirectories are derived from `base_dir`. The directories
        are not created by this method — use `mkdir` on each path as needed.

        Args:
            base_dir: Root directory for all pipeline outputs.

        Returns:
            A `PathConfig` instance with all paths derived from `base_dir`.

        Example:
            >>> from pathlib import Path
            >>> config = PathConfig.from_base_dir(Path("data"))
            >>> config.paper_pdf_dir
            PosixPath('data/papers/pdf')
        """
        base_dir.mkdir(parents=True, exist_ok=True)
        paper_dir = base_dir / "papers"
        return cls(
            paper_dir=paper_dir,
            author_dir=base_dir / "authors",
            paper_pdf_dir=paper_dir / "pdf",
        )


def main() -> None:
    r"""Define the main function."""
    base_dir = Path(__file__).parent.parent.joinpath("data/v1").joinpath("cvpr26").resolve()
    path_config = PathConfig.from_base_dir(base_dir)
    logger.info(path_config)

    scraper = CVFPaperScraper(venue="CVPR", year=2026, cache_dir=path_config.paper_dir)
    logger.info(scraper)
    papers = dataframe_to_papers(scraper.scrape().head(10))
    logger.info(papers)

    download_pdfs(papers=papers, pdf_dir=path_config.paper_pdf_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
