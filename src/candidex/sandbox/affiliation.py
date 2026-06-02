r"""Contain functionalities to extract author affiliations from PDF."""

from __future__ import annotations

__all__ = ["AFFILIATION_SYSTEM_PROMPT", "extract_and_save_affiliations", "load_affiliations"]

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from iden.io import load_json, save_json
from langchain_core.messages import HumanMessage, SystemMessage

from candidex.columns import PAPER_STEM
from candidex.sandbox.pdf import PDFReadError, extract_first_page_text
from candidex.sandbox.progressbar import make_progressbar
from candidex.schemas import PaperAffiliations

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from langchain_core.language_models import BaseChatModel

logger: logging.Logger = logging.getLogger(__name__)


AFFILIATION_SYSTEM_PROMPT = """
You are an expert at parsing academic paper metadata.

TASK:
Extract every author and their full institutional affiliations and email address from the first page of an academic paper.

AUTHOR LIST:
A known list of authors will be provided alongside the text. Use it to:
- Verify you have extracted every author without omission.
- Resolve ambiguous name formats — if the text contains 'J. Smith' but the author list contains 'Jane Smith', use 'Jane Smith'.
- Anchor affiliation markers (superscripts, symbols) to the correct author when the text layout is ambiguous.
- Preserve the exact order from the author list, which reflects the order on the paper.

AUTHOR EXTRACTION RULES:
- Extract every author exactly as their name appears on the paper. Do not normalise, expand initials, or infer missing name parts.
- Preserve the exact order in which authors appear on the paper.
- Do not omit any author, including those with no detectable affiliation or email.

AFFILIATION EXTRACTION RULES:
- An author may have multiple affiliations, typically indicated by superscript numbers, letters, or symbols (e.g. ¹, ², *, †) placed after the author name and before the corresponding affiliation entry.
- Extract the full affiliation string as it appears on the paper, including department, institution, city, country, and postal code where available (e.g. 'Department of Computer Science, MIT, Cambridge, MA 02139, USA').
- If an author shares an affiliation with another author, still list it explicitly for each author — do not reference other authors.
- If no affiliation can be determined for an author, return an empty list for that author.
- Do not infer or guess affiliations that are not explicitly stated in the text.

EMAIL EXTRACTION RULES:
- Extract the email address only if it is explicitly stated on the paper for that author.
- Emails may appear inline next to the author name, in a footnote, or in a correspondence block (e.g. 'Correspondence to: jane.smith@mit.edu').
- If an email is listed for a group of authors without individual attribution (e.g. '{alice,bob}@mit.edu'), expand it to each individual address (e.g. 'alice@mit.edu', 'bob@mit.edu').
- Set email to None if no email is present or if it cannot be unambiguously attributed to a specific author.
- Do not infer, construct, or guess email addresses from author names or affiliations.

COMMON PATTERNS TO HANDLE:
- Superscript markers: "Alice Smith¹, Bob Jones¹²" where ¹ and ² map to listed institutions.
- Footnote-style: affiliations listed at the bottom of the author block with matching markers.
- Inline style: affiliation written directly after the author name in parentheses or on the next line.
- Equal contribution markers (e.g. *, †): these denote contribution, not affiliation — ignore them unless they also map to an institution.
- Corresponding author markers (e.g. ✉): use to attribute an email to a specific author if accompanied by an email address.

OUTPUT:
Return a structured result containing every author, their affiliations, and their email address (if available) in the order they appear on the paper.
"""


def extract_affiliations(
    pdf_path: Path,
    llm: BaseChatModel,
    authors: list[str] | None = None,
) -> PaperAffiliations:
    """Extract author affiliations from the first page of a research
    paper PDF.

    Reads the first page of the PDF, then uses an LLM with structured output
    to parse each author's name and their associated affiliations. When a list
    of known author names is provided, it is passed to the LLM alongside the
    text to improve accuracy — particularly useful for resolving ambiguous name
    formats and anchoring affiliation markers to the correct author.

    Uses LangChain's `with_structured_output` to enforce a typed Pydantic response,
    avoiding brittle string parsing of the LLM output.

    Args:
        pdf_path: Path to the PDF file to extract affiliations from.
        llm:      Any LangChain-compatible chat model. The caller is responsible
                  for initialising and configuring it, e.g.:
                      ChatAnthropic(model="claude-3-5-sonnet-20241022")
                      ChatOpenAI(model="gpt-4o")
                      ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        authors:  Optional list of known author names for this paper, typically
                  taken from the `authors` column of the papers DataFrame. When
                  provided, the LLM uses it to verify completeness, resolve
                  ambiguous name formats, and anchor affiliation markers more
                  accurately. Defaults to None.

    Returns:
        A `PaperAffiliations` object containing a list of `AuthorAffiliation`
        entries, each with the author's full name, list of affiliations, and
        email address if found. Authors with no detectable affiliation will
        have an empty affiliations list.

    Raises:
        FileNotFoundError: If the PDF does not exist at `pdf_path`.

    Example:
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> result = extract_affiliations(
        ...     Path("papers/attention.pdf"),
        ...     llm,
        ...     authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        ... )
        >>> for entry in result.authors:
        ...     print(entry.author, entry.affiliations, entry.email)
    """
    text = extract_first_page_text(pdf_path)
    if not text:
        logger.warning("No text extracted from first page of %s.", pdf_path)
        return PaperAffiliations(authors=[])

    logger.debug("Extracting affiliations from %s.", pdf_path)

    author_block = (
        "\nKnown authors for this paper (in order):\n" + "\n".join(f"- {a}" for a in authors)
        if authors
        else ""
    )

    structured_llm = llm.with_structured_output(PaperAffiliations)
    messages = [
        SystemMessage(content=AFFILIATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Extract the author affiliations from this text:{author_block}\n\n"
                f"Paper text:\n{text}"
            )
        ),
    ]

    result = structured_llm.invoke(messages)
    logger.debug("Extracted affiliations for %d authors from %s.", len(result.authors), pdf_path)
    return result


def extract_and_save_paper_affiliations(
    row: dict[str, Any],
    pdf_dir: Path,
    affiliation_dir: Path,
    llm: BaseChatModel,
) -> None:
    """Extract and save author affiliations for a single paper.

    Extracts author affiliations from the first page of the PDF using an
    LLM and writes the result to a JSON file in `affiliation_dir`. Skips
    the paper if the affiliation file already exists or the PDF is not found.
    Handles `PDFReadError` gracefully by logging a warning and returning
    rather than raising.

    This function is the single-paper counterpart to
    `extract_and_save_affiliations`, which processes a full DataFrame of
    papers concurrently by submitting this function to a thread pool.

    Args:
        row:             A single row from the papers DataFrame as a dict.
                         Must contain a key matching `PAPER_STEM` with the
                         PDF filename stem, and an `authors` key with the
                         list of author names.
        pdf_dir:         Directory where the PDF files are stored. The PDF
                         must be named `{stem}.pdf`.
        affiliation_dir: Directory where the affiliation JSON file will be
                         written. Must already exist.
        llm:             Any LangChain-compatible chat model. The caller is
                         responsible for initialising and configuring it.

    Raises:
        Exception: Unexpected errors from the LLM call are logged and
                   suppressed. Only truly unexpected errors propagate.

    Example:
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> extract_and_save_paper_affiliations(
        ...     row={"stem": "attention", "authors": ["Vaswani", "Shazeer"]},
        ...     pdf_dir=Path("data/cvpr2024/pdfs"),
        ...     affiliation_dir=Path("data/cvpr2024/affiliations"),
        ...     llm=llm,
        ... )
    """
    stem = row[PAPER_STEM]
    pdf_path = pdf_dir / f"{stem}.pdf"
    affiliation_path = affiliation_dir / f"{stem}.json"

    if affiliation_path.is_file():
        logger.debug("Skipping %s, affiliation file already exists.", stem)
        return

    if not pdf_path.is_file():
        logger.warning("PDF not found, skipping: %s.", pdf_path.name)
        return

    try:
        affiliations = extract_affiliations(pdf_path, llm=llm, authors=row["authors"])
        logger.info(f"Extracted author affiliations for {stem}\n{affiliations}")
        save_json(affiliations.model_dump(), affiliation_path)
        logger.debug("Saved affiliations to %s.", affiliation_path.name)
    except PDFReadError as e:
        logger.warning("Skipping %s — PDF could not be read: %s", pdf_path.name, e)


def extract_and_save_affiliations(
    papers: pl.DataFrame,
    pdf_dir: Path,
    affiliation_dir: Path,
    llm: BaseChatModel,
    max_workers: int = 4,
) -> None:
    """Extract author affiliations for each paper and save them as
    individual JSON files.

    Processes papers concurrently using a thread pool by delegating each
    paper to `extract_and_save_paper_affiliations`. Papers whose JSON file
    already exists are skipped, making the function safe to call repeatedly
    and resilient to interruptions.

    The JSON file for a paper with stem `paper` will be saved as `paper.json`
    in `affiliation_dir`. Each file contains a serialised `PaperAffiliations`
    object with `authors`, `affiliations`, and `email` keys.

    Args:
        papers:          Polars DataFrame produced by `scrape_cvpr_papers` or
                         equivalent. Must contain a column named by `PAPER_STEM`
                         with the PDF filename stem (i.e. without the `.pdf`
                         extension), and an `authors` column with the list of
                         author names for each paper.
        pdf_dir:         Directory where the PDF files are stored. Each PDF must
                         be named `{stem}.pdf` where `stem` matches the value in
                         the `PAPER_STEM` column.
        affiliation_dir: Directory where affiliation JSON files will be written.
                         Created automatically if it does not exist. Output files
                         are named `{stem}.json` to match their source PDF.
        llm:             Any LangChain-compatible chat model. The caller is
                         responsible for initialising and configuring it, e.g.:
                             ChatAnthropic(model="claude-3-5-sonnet-20241022")
                             ChatOpenAI(model="gpt-4o")
        max_workers:     Maximum number of concurrent threads for LLM calls.
                         Defaults to 4. Reduce if hitting API rate limits.

    Example:
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> extract_and_save_affiliations(
        ...     papers=df,
        ...     pdf_dir=Path("data/cvpr2024/pdfs"),
        ...     affiliation_dir=Path("data/cvpr2024/affiliations"),
        ...     llm=llm,
        ...     max_workers=4,
        ... )
    """
    affiliation_dir.mkdir(parents=True, exist_ok=True)
    rows = list(papers.iter_rows(named=True))

    logger.info(
        "Extracting author affiliations from %d files to %s with %d workers...",
        len(rows),
        affiliation_dir,
        max_workers,
    )

    with make_progressbar() as progress:
        task = progress.add_task("Extracting author affiliations", total=len(rows))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extract_and_save_paper_affiliations, row, pdf_dir, affiliation_dir, llm
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                future.result()
                progress.advance(task)


def load_affiliations(papers: pl.DataFrame, affiliation_dir: Path) -> dict[str, PaperAffiliations]:
    """Load author affiliations from JSON files into a dictionary.

    Iterates over a DataFrame of papers and loads the corresponding affiliation
    JSON file for each paper. Papers whose JSON file does not exist are logged
    and skipped. Useful for downstream processing without re-running the
    extraction step.

    Args:
        papers:          Polars DataFrame produced by `scrape_cvpr_papers` or
                         equivalent. Must contain a column named by `PAPER_STEM`
                         with the PDF filename stem (i.e. without the `.pdf`
                         extension) for each paper.
        affiliation_dir: Directory containing the affiliation JSON files produced
                         by `extract_and_save_affiliations`. Each file must be
                         named `{stem}.json` where `stem` matches the value in
                         the `PAPER_STEM` column.

    Returns:
        A dictionary mapping each paper stem to its `PaperAffiliations` object.
        Papers with missing or unreadable JSON files are omitted from the result.

    Example:
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> affiliations = load_affiliations(df, Path("data/cvpr2024/affiliations"))
        >>> affiliations["attention_is_all_you_need"].authors
    """
    logger.info("Loading author affiliations from %s...", affiliation_dir)

    rows = list(papers.iter_rows(named=True))
    affiliations = {}

    with make_progressbar() as progress:
        task = progress.add_task("Loading author affiliations", total=len(rows))
        for row in rows:
            stem = row[PAPER_STEM]
            affiliation_path = affiliation_dir.joinpath(f"{stem}.json")

            if not affiliation_path.is_file():
                logger.warning("Affiliation file not found, skipping: %s.", affiliation_path.name)
                progress.advance(task)
                continue

            affiliations[stem] = PaperAffiliations.model_validate(load_json(affiliation_path))
            progress.advance(task)

    logger.info("Loaded author affiliations for %d/%d papers.", len(affiliations), len(papers))
    return affiliations
