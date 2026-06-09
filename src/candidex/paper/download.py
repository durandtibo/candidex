r"""Contain utilities for downloading papers."""

from __future__ import annotations

__all__ = ["download_pdfs"]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from candidex.pdf.download import download_pdf
from candidex.utils.http import create_session
from candidex.utils.progressbar import make_progressbar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from candidex.paper import Paper

logger: logging.Logger = logging.getLogger(__name__)


def download_pdfs(
    papers: Sequence[Paper],
    pdf_dir: Path,
    timeout: int = 30,
    chunk_size: int = 8192,
    max_retries: int = 3,
    max_workers: int = 4,
) -> dict[Paper, bool]:
    """Download PDFs for a sequence of papers concurrently.

    Downloads each paper's PDF to `pdf_dir`, using the paper's hash as the
    filename. Papers whose PDF URL is None or whose file already exists are
    skipped. Downloads are performed concurrently using a thread pool with
    a shared session for connection pooling.

    Args:
        papers:      Sequence of `Paper` objects to download PDFs for.
                     Papers with `pdf_url=None` are skipped and mapped to
                     False in the result.
        pdf_dir:     Directory where PDF files will be saved. Created
                     automatically if it does not exist. Each file is named
                     `{paper.hash()}.pdf`.
        timeout:     Request timeout in seconds per download. Defaults to 30.
        chunk_size:  Number of bytes per chunk when streaming. Defaults to
                     8192.
        max_retries: Maximum number of retry attempts on transient failures.
                     Defaults to 3.
        max_workers: Maximum number of concurrent download threads. Defaults
                     to 4.

    Returns:
        A dictionary mapping each `Paper` to a bool indicating whether its
            PDF was downloaded successfully (or already existed). Papers with no
            `pdf_url` are mapped to False.

    Example:
        ```pycon
        >>> from pathlib import Path
        >>> from candidex.paper import download_pdfs, Paper
        >>> papers = [
        ...     Paper.from_raw(
        ...         title="Attention Is All You Need",
        ...         authors=["Ashish Vaswani"],
        ...         venue="NeurIPS",
        ...         year=2017,
        ...         pdf_url="https://arxiv.org/pdf/1706.03762",
        ...     )
        ... ]
        >>> results = download_pdfs(papers, pdf_dir=Path("data/pdfs"))  # doctest: +SKIP

        ```
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    results: dict[Paper, bool] = {}

    def _download(paper: Paper) -> tuple[Paper, bool]:
        if paper.pdf_url is None:
            logger.debug("Skipping %s — no PDF URL.", paper)
            return paper, False
        return paper, download_pdf(
            url=paper.pdf_url,
            pdf_path=pdf_dir / paper.to_filename(".pdf"),
            timeout=timeout,
            chunk_size=chunk_size,
            session=session,
        )

    session = create_session(max_retries=max_retries)
    try:
        with make_progressbar() as progress:
            task = progress.add_task("Downloading PDFs", total=len(papers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_download, paper): paper for paper in papers}
                for future in as_completed(futures):
                    paper, success = future.result()
                    results[paper] = success
                    progress.advance(task)
    finally:
        session.close()

    total = len(results)
    successful = sum(1 for s in results.values() if s)
    skipped = sum(1 for p in papers if p.pdf_url is None)
    failed = total - successful - skipped
    logger.info(
        "PDF download complete: %d/%d successful, %d skipped (no URL), %d failed.",
        successful,
        total,
        skipped,
        failed,
    )
    return results
