r"""Contain functionalities to extract author affiliations from PDF."""

from __future__ import annotations

__all__ = ["AFFILIATION_SYSTEM_PROMPT", "extract_and_save_affiliations"]

import logging
from typing import TYPE_CHECKING

import pdfplumber
from iden.io import save_json
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from candidex.columns import PAPER_STEM

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from langchain_core.language_models import BaseChatModel

logger: logging.Logger = logging.getLogger(__name__)


class AuthorAffiliation(BaseModel):
    """Represents a single author and their institutional affiliations.

    An author may be affiliated with multiple institutions simultaneously,
    for example a university department and an associated research lab.
    Affiliations are typically listed on the first page of an academic paper,
    often linked to author names via superscript numbers or symbols.

    Attributes:
        author:       Full name of the author exactly as it appears on the paper
                      (e.g. 'Jane Smith', 'J. Smith'). Do not normalise or infer
                      missing name parts.
        affiliations: List of institutional affiliations for this author. Each
                      entry should be a complete affiliation string as it appears
                      on the paper (e.g. 'MIT CSAIL, Cambridge, MA, USA').
                      Use an empty list if no affiliation can be determined.
    """

    author: str = Field(description="Full name of the author exactly as it appears on the paper.")
    affiliations: list[str] = Field(
        description=(
            "List of institutional affiliations for this author. Each entry is a complete "
            "affiliation string as it appears on the paper. Empty list if none can be determined."
        )
    )


class PaperAffiliations(BaseModel):
    """Contains the full list of authors and their affiliations for a
    single paper.

    Extracted from the first page of an academic paper where author names and
    their corresponding institutional affiliations are listed. Each author is
    represented as an `AuthorAffiliation` entry. Preserve the order of authors
    exactly as they appear on the paper, as author order carries meaning in
    academic publishing (e.g. first author, last/senior author).

    Attributes:
        authors: Ordered list of all authors and their affiliations. Must
                 include every author listed on the paper without omission.
                 Each entry maps one author to their affiliations.
    """

    authors: list[AuthorAffiliation] = Field(
        description=(
            "Ordered list of all authors and their affiliations, preserving the order "
            "in which they appear on the paper. Must include every author without omission."
        )
    )


AFFILIATION_SYSTEM_PROMPT = """
You are an expert at parsing academic paper metadata.

TASK:
Extract every author and their full institutional affiliations from the first page of an academic paper.

AUTHOR EXTRACTION RULES:
- Extract every author exactly as their name appears on the paper. Do not normalise, expand initials, or infer missing name parts.
- Preserve the exact order in which authors appear on the paper.
- Do not omit any author, including those with no detectable affiliation.

AFFILIATION EXTRACTION RULES:
- An author may have multiple affiliations, typically indicated by superscript numbers, letters, or symbols (e.g. ¹, ², *, †) placed after the author name and before the corresponding affiliation entry.
- Extract the full affiliation string as it appears on the paper, including department, institution, city, country, and postal code where available (e.g. 'Department of Computer Science, MIT, Cambridge, MA 02139, USA').
- If an author shares an affiliation with another author, still list it explicitly for each author — do not reference other authors.
- If no affiliation can be determined for an author, return an empty list for that author.
- Do not infer or guess affiliations that are not explicitly stated in the text.

COMMON PATTERNS TO HANDLE:
- Superscript markers: "Alice Smith¹, Bob Jones¹²" where ¹ and ² map to listed institutions.
- Footnote-style: affiliations listed at the bottom of the author block with matching markers.
- Inline style: affiliation written directly after the author name in parentheses or on the next line.
- Equal contribution markers (e.g. *, †): these denote contribution, not affiliation — ignore them unless they also map to an institution.
- Corresponding author markers (e.g. ✉): ignore unless they also map to an institution.

OUTPUT:
Return a structured result containing every author and their affiliations in the order they appear on the paper.
"""


def extract_first_page_text(pdf_path: Path) -> str:
    """Extract raw text from the first page of a PDF.

    Only the first page is read since author affiliations are always
    listed there, avoiding the overhead of parsing the full document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Raw text content of the first page.

    Raises:
        FileNotFoundError: If the PDF does not exist at `pdf_path`.
        pdfplumber.exceptions.PDFSyntaxError: If the file is not a valid PDF.
    """
    logger.debug("Extracting first page text from %s.", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        return first_page.extract_text() or ""


def extract_affiliations(pdf_path: Path, llm: BaseChatModel) -> PaperAffiliations:
    """Extract author affiliations from the first page of a research
    paper PDF.

    Reads the first page of the PDF, then uses an LLM with structured output
    to parse each author's name and their associated affiliations. Affiliations
    are typically denoted by superscript numbers or symbols next to author names
    on the first page of academic papers.

    Uses LangChain's `with_structured_output` to enforce a typed Pydantic response,
    avoiding brittle string parsing of the LLM output.

    Args:
        pdf_path: Path to the PDF file to extract affiliations from.
        llm:      Any LangChain-compatible chat model. The caller is responsible
                  for initialising and configuring it, e.g.:
                      ChatAnthropic(model="claude-3-5-sonnet-20241022")
                      ChatOpenAI(model="gpt-4o")
                      ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    Returns:
        A `PaperAffiliations` object containing a list of `AuthorAffiliation`
        entries, each with the author's full name and their list of affiliations.
        Authors with no detectable affiliation will have an empty list.

    Raises:
        FileNotFoundError: If the PDF does not exist at `pdf_path`.

    Example:
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> result = extract_affiliations(Path("papers/attention.pdf"), llm)
        >>> for entry in result.authors:
        ...     print(entry.author, entry.affiliations)
    """
    text = extract_first_page_text(pdf_path)
    if not text:
        logger.warning("No text extracted from first page of %s.", pdf_path)
        return PaperAffiliations(authors=[])

    logger.debug("Extracting affiliations from %s.", pdf_path)

    structured_llm = llm.with_structured_output(PaperAffiliations)

    messages = [
        SystemMessage(content=AFFILIATION_SYSTEM_PROMPT),
        HumanMessage(content=f"Extract the author affiliations from this text:\n\n{text}"),
    ]

    result = structured_llm.invoke(messages)
    logger.debug("Extracted affiliations for %d authors from %s.", len(result.authors), pdf_path)
    return result


def extract_and_save_affiliations(
    papers: pl.DataFrame,
    pdf_dir: Path,
    affiliation_dir: Path,
    llm: BaseChatModel,
) -> None:
    """Extract author affiliations for each paper and save them as
    individual JSON files.

    Iterates over a DataFrame of papers, extracts author affiliations from the
    first page of each PDF using an LLM, and writes the result to a JSON file
    named after the PDF. Papers whose JSON file already exists are skipped,
    making the function safe to call repeatedly and resilient to interruptions.

    The JSON file for a paper named `paper.pdf` will be saved as `paper.json`
    in the same `pdf_dir`. Each file contains a list of objects with `author`
    and `affiliations` keys, matching the `PaperAffiliations` schema.

    Args:
        papers:     Polars DataFrame produced by `scrape_cvpr_papers` or equivalent,
                    must contain a column named `filename` with the PDF filename
                    (not the full path) for each paper.
        pdf_dir: Directory where both the PDFs and output JSON files reside.
        llm:        Any LangChain-compatible chat model. The caller is responsible
                    for initialising and configuring it, e.g.:
                        ChatAnthropic(model="claude-3-5-sonnet-20241022")
                        ChatOpenAI(model="gpt-4o")

    Example:
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> extract_and_save_affiliations(df, Path("papers/cvpr2024"), llm)
    """
    rows = list(papers.iter_rows(named=True))
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Extracting affiliations", total=len(rows))
        for row in rows:
            stem = row[PAPER_STEM]
            pdf_path = pdf_dir.joinpath(f"{stem}.pdf")
            affiliation_path = affiliation_dir.joinpath(rf"{stem}.json")

            if affiliation_path.is_file():
                logger.debug("Skipping %s, affiliation file already exists.", pdf_path.name)
                progress.advance(task)
                continue

            if not pdf_path.is_file():
                logger.warning("PDF not found, skipping: %s.", pdf_path)
                progress.advance(task)
                continue

            affiliations = extract_affiliations(pdf_path, llm=llm)
            save_json(affiliations.model_dump(), affiliation_path)

            progress.advance(task)
