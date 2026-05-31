r"""Contain functionalities to find the role of the authors."""

from __future__ import annotations

__all__ = ["find_authors_status"]

import logging
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from iden.io import load_json, save_json, load_text
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from candidex.columns import PAPER_FILENAME
from candidex.sandbox.affiliation import PaperAffiliations

if TYPE_CHECKING:
    import polars as pl
    from langchain_core.language_models import BaseChatModel

    from candidex.sandbox.affiliation import AuthorAffiliation

logger: logging.Logger = logging.getLogger(__name__)


class AcademicRole(StrEnum):
    BACHELOR_STUDENT = "Bachelor Student"
    MASTER_STUDENT = "Master Student"
    PHD_STUDENT = "PhD Student"
    POSTDOC = "Postdoc"
    RESEARCH_SCIENTIST = "Research Scientist"
    ASSISTANT_PROFESSOR = "Assistant Professor"
    ASSOCIATE_PROFESSOR = "Associate Professor"
    FULL_PROFESSOR = "Full Professor"
    INDUSTRY_RESEARCHER = "Industry Researcher"
    UNKNOWN = "Unknown"


class AuthorStatus(BaseModel):
    name: str = Field(description="Full name of the author.")
    affiliation: str = Field(description="Known affiliation of the author.")
    role: AcademicRole = Field(
        description=(
            "Current role of the author. Must be one of the predefined categories:\n"
            "- Bachelor Student: undergraduate student\n"
            "- Master Student: postgraduate student in a Master's program\n"
            "- PhD Student: doctoral student\n"
            "- Postdoc: postdoctoral researcher\n"
            "- Research Scientist: researcher at a university or industry lab without a faculty title\n"
            "- Assistant Professor: early-career faculty member\n"
            "- Associate Professor: mid-career tenured or tenure-track faculty member\n"
            "- Full Professor: senior tenured faculty member\n"
            "- Industry Researcher: researcher or engineer working in industry outside of a research lab\n"
            "- Unknown: role cannot be determined from available information"
        )
    )
    phd_start_year: str | None = Field(
        description=(
            "Year the author started their PhD (e.g. '2019'). "
            "Set to 'UNKNOWN' if they have or are pursuing a PhD but the start year cannot be determined. "
            "Set to 'NO PhD' if the author has no PhD and is not pursuing one. "
            "None only if role is 'Unknown' and no information is available."
        )
    )
    phd_end_year: str | None = Field(
        description=(
            "Year the author completed their PhD (e.g. '2023'). "
            "Set to 'UNKNOWN' if they have or are pursuing a PhD but the end year cannot be determined. "
            "Set to 'NO PhD' if the author has no PhD and is not pursuing one. "
            "Set to 'In Progress' if the author is currently a PhD student. "
            "None only if role is 'Unknown' and no information is available."
        )
    )
    phd_domain: str | None = Field(
        description=(
            "Domain or field of the PhD (e.g. 'Computer Vision', 'Machine Learning', 'Robotics'). "
            "Set to 'UNKNOWN' if they have or are pursuing a PhD but the domain cannot be determined. "
            "Set to 'NO PhD' if the author has no PhD and is not pursuing one. "
            "None only if role is 'Unknown' and no information is available."
        )
    )
    phd_university: str | None = Field(
        description=(
            "University where the author completed or is pursuing their PhD (e.g. 'MIT', 'Stanford University'). "
            "Set to 'UNKNOWN' if they have or are pursuing a PhD but the university cannot be determined. "
            "Set to 'NO PhD' if the author has no PhD and is not pursuing one. "
            "None only if role is 'Unknown' and no information is available."
        )
    )
    details: str | None = Field(
        description=(
            "Any additional relevant details such as lab, research group, supervisor, "
            "or university department."
        )
    )
    source: str | None = Field(description="URL of the source where the information was found.")


AUTHOR_SYSTEM_PROMPT = """
You are an expert at finding information about academic researchers.
Given an author's name and affiliation, determine their current academic role
and PhD information.

SEARCH STRATEGY:
Use the following search strategies in order of priority to find the most
accurate and up-to-date information:

1. Resume or CV search — most reliable source for PhD details and career history:
   "{author_name}" (intitle:resume OR intitle:cv OR inurl:resume OR inurl:cv) filetype:pdf
   "{author_name}" "{affiliation}" (resume OR cv OR curriculum vitae)

2. GitHub profile — useful for current role, affiliations listed in bio, and linked personal sites:
   "{author_name}" "{affiliation}" site:github.com
   "{author_name}" site:github.com machine learning

3. Academic profile pages:
   "{author_name}" "{affiliation}" site:scholar.google.com
   "{author_name}" "{affiliation}" (professor OR researcher OR "PhD student" OR postdoc)

4. University and lab pages:
   "{author_name}" site:{affiliation_domain} (e.g. site:mit.edu, site:stanford.edu)

5. General web search as a fallback:
   "{author_name}" "{affiliation}" academic position

ROLE ASSIGNMENT:
Assign exactly one of the following roles:
- Bachelor Student: enrolled in an undergraduate degree program
- Master Student: enrolled in a postgraduate Master's program
- PhD Student: enrolled in a doctoral program
- Postdoc: holds a PhD and is in a temporary postdoctoral research position
- Research Scientist: conducting research at a university or industry lab without a faculty title (e.g. Research Engineer, Staff Scientist)
- Assistant Professor: early-career faculty, often pre-tenure
- Associate Professor: mid-career faculty, often tenured
- Full Professor: senior faculty, full tenure
- Industry Researcher: working in industry in a non-research or engineering role (e.g. Software Engineer, ML Engineer)
- Unknown: insufficient information to determine the role

PHD INFORMATION RULES:
- If the author has completed or is currently pursuing a PhD, populate phd_start_year,
  phd_end_year, phd_domain, and phd_university with the best available information.
- Use 'UNKNOWN' for any PhD field that cannot be determined from the search results.
- Use 'In Progress' for phd_end_year if the author is currently a PhD student.
- Use 'NO PhD' for all PhD fields if the author has never pursued a PhD.
- Only use None for PhD fields if the role itself is 'Unknown' and no information
  is available at all.

SOURCE RULES:
- The source field must contain exactly one URL — the single most authoritative
  source used to determine the role and PhD information.
- Choose the URL according to this priority:
  1. Personal CV or resume (PDF)
  2. GitHub profile bio and linked personal website
  3. University or lab profile page
  4. Google Scholar profile
  5. LinkedIn profile
  6. Any other credible web source
- Never include multiple URLs, comma-separated links, or prose descriptions in
  the source field. Only a single raw URL (e.g. 'https://example.com/cv.pdf').
- If no credible source was found, set source to None.

Only report years if explicitly stated in the source. Do not infer or estimate years. Do not be lazy.
Always record the URL of the source used in the source field.
"""


def _run_searches(author: AuthorAffiliation, search: DuckDuckGoSearchRun) -> str:
    """Run multiple targeted DuckDuckGo searches for an author and
    return combined results.

    Executes two queries in sequence: one targeting CV or resume documents,
    and one targeting academic profile pages. Results from both queries are
    combined into a single formatted string for the LLM to reason over.

    Unlike `GoogleSearchAPIWrapper`, `DuckDuckGoSearchRun.run()` returns a
    plain text string rather than a list of structured dicts, so results are
    concatenated directly without per-field formatting.

    Args:
        author: An `AuthorAffiliation` object containing the author's name
                and known affiliations.
        search: A configured `DuckDuckGoSearchRun` instance.

    Returns:
        A formatted string of search results from both queries, separated by
        a divider. Returns an empty string if all queries return no results.
    """
    affiliation_str = ", ".join(author.affiliations) if author.affiliations else "Unknown"

    queries = [
        f'"{author.author}" (intitle:resume OR intitle:cv OR inurl:resume OR inurl:cv)',
        f'"{author.author}" "{affiliation_str}" PhD student professor researcher',
    ]

    sections = []
    for query in queries:
        result = search.run(query)
        if result:
            sections.append(result)

    return "\n\n---\n\n".join(sections)


def find_author_status(
    author: AuthorAffiliation,
    llm: BaseChatModel,
    search: DuckDuckGoSearchRun,
) -> AuthorStatus:
    """Find the current academic role of an author using DuckDuckGo
    search and an LLM.

    Runs two targeted DuckDuckGo searches — one for CV/resume documents, one for
    academic profile pages — combines the results, and passes them to the LLM
    to extract structured role and PhD information.

    Note: DuckDuckGo may raise `DDGSException` on DNS or connectivity errors in
    restricted network environments. Consider using `GoogleSearchAPIWrapper` if
    reliability is a concern.

    Args:
        author: An `AuthorAffiliation` object containing the author's name and
                known affiliations from the paper.
        llm:    Any LangChain-compatible chat model with structured output support.
                The caller is responsible for initialising and configuring it.
        search: A configured `DuckDuckGoSearchRun` instance.

    Returns:
        An `AuthorStatus` object with the author's current role, PhD details,
        and source URL. If no information can be found, `role` is set to
        `AcademicRole.UNKNOWN` and all PhD fields are None.

    Raises:
        langchain_core.exceptions.LangChainException: If the LLM call fails.
        duckduckgo_search.exceptions.DDGSException:   On DuckDuckGo connectivity errors.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> search = DuckDuckGoSearchRun()
        >>> author = AuthorAffiliation(author="Jane Smith", affiliations=["MIT CSAIL"])
        >>> status = find_author_status(author, llm, search)
        >>> print(status.role, status.phd_start_year)
    """
    affiliation_str = ", ".join(author.affiliations) if author.affiliations else "Unknown"
    logger.debug("Searching for author status: %s (%s).", author.author, affiliation_str)

    search_text = _run_searches(author, search)
    logger.debug("Search results retrieved for %s.", author.author)

    structured_llm = llm.with_structured_output(AuthorStatus)
    messages = [
        SystemMessage(content=AUTHOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Find the current academic role for this author:\n\n"
                f"Name: {author.author}\n"
                f"Affiliation: {affiliation_str}\n\n"
                f"Search results:\n{search_text}"
            )
        ),
    ]

    result = structured_llm.invoke(messages)
    logger.debug("Determined role for %s: %s.", author.author, result.role)
    return result


def find_authors_status(
    affiliations: PaperAffiliations,
    llm: BaseChatModel,
    search: DuckDuckGoSearchRun | None = None,
) -> list[AuthorStatus]:
    """Find the current academic role for all authors in a paper.

    Iterates over each author in the provided `PaperAffiliations`, performs
    DuckDuckGo searches for their current position, and uses an LLM to extract
    structured role and PhD information. Failures on individual authors are
    logged and skipped so that a single lookup failure does not abort the
    entire batch.

    Note: DuckDuckGo may raise `DDGSException` on DNS or connectivity errors in
    restricted network environments. Consider using `GoogleSearchAPIWrapper` if
    reliability is a concern.

    Args:
        affiliations: A `PaperAffiliations` object as returned by
                      `extract_affiliations`, containing each author's name
                      and known affiliations.
        llm:          Any LangChain-compatible chat model with structured output
                      support. The caller is responsible for initialising and
                      configuring it.
        search:       A `DuckDuckGoSearchRun` instance. Defaults to a new
                      instance if not provided. Override to inject a mock
                      for testing.

    Returns:
        A list of `AuthorStatus` objects, one per author for whom a lookup was
        attempted. Authors that raised an exception are omitted from the list
        rather than surfaced as partial results.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> affiliations = extract_affiliations(Path("papers/attention.pdf"), llm)
        >>> statuses = find_authors_status(affiliations, llm)
        >>> for s in statuses:
        ...     print(s.name, s.role, s.phd_start_year, s.phd_end_year)
    """
    search = search or DuckDuckGoSearchRun()
    results = []
    total = len(affiliations.authors)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Finding author roles", total=total)
        for author in affiliations.authors:
            try:
                status = find_author_status(author, llm=llm, search=search)
                results.append(status)
            except Exception as e:
                logger.warning("Failed to find status for %s: %s", author.author, e)
            finally:
                progress.advance(task)

    found = sum(1 for r in results if r.role != AcademicRole.UNKNOWN)
    failed = total - len(results)
    logger.info(
        "Author status lookup complete. %d/%d resolved, %d unknown, %d failed.",
        found,
        total,
        len(results) - found,
        failed,
    )
    return results


def find_and_save_authors_status(
    papers: pl.DataFrame,
    affiliations_dir: Path,
    role_dir: Path,
    llm: BaseChatModel,
    search: DuckDuckGoSearchRun | None = None,
) -> None:
    """Find and save the academic status of all authors for each paper
    to a JSON file.

    For each paper in the DataFrame, loads the corresponding affiliations JSON
    file, looks up the current academic role of each author, and writes the
    results to a JSON file named after the paper. Papers whose status JSON file
    already exists are skipped, making the function safe to call repeatedly and
    resilient to interruptions.

    The JSON file for a paper named `paper.pdf` will be saved as `paper.json`
    in `role_dir`. Each file contains a list of serialised `AuthorStatus`
    objects.

    Args:
        papers:           Polars DataFrame produced by `scrape_cvpr_papers` or
                          equivalent, must contain a column named `filename`
                          with the PDF filename (not the full path) for each paper.
        affiliations_dir: Directory containing the affiliation JSON files produced
                          by `extract_and_save_affiliations`. Each file must be
                          named after its corresponding PDF (e.g. `paper.json`
                          for `paper.pdf`).
        role_dir:         Directory where author status JSON files will be saved.
                          Created automatically if it does not exist.
        llm:              Any LangChain-compatible chat model with structured output
                          support. The caller is responsible for initialising and
                          configuring it.
        search:           A `DuckDuckGoSearchRun` instance. Defaults to a new
                          instance if not provided. Override to inject a mock
                          for testing.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> find_and_save_authors_status(
        ...     papers=df,
        ...     affiliations_dir=Path("papers/cvpr2024/affiliations"),
        ...     role_dir=Path("papers/cvpr2024/status"),
        ...     llm=llm,
        ... )
    """
    role_dir.mkdir(parents=True, exist_ok=True)
    search = search or DuckDuckGoSearchRun()
    rows = list(papers.iter_rows(named=True))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing papers", total=len(rows))
        for row in rows:
            stem = Path(row[PAPER_FILENAME]).stem
            affiliation_path = affiliations_dir.joinpath(f"{stem}.json")
            status_path = role_dir.joinpath(f"{stem}.json")

            if status_path.exists():
                logger.debug("Skipping %s, status file already exists.", stem)
                progress.advance(task)
                continue

            if not affiliation_path.exists():
                logger.warning("Affiliation file not found, skipping: %s.", affiliation_path.name)
                progress.advance(task)
                continue

            affiliations = PaperAffiliations.model_validate_json(load_text(affiliation_path))
            statuses = find_authors_status(affiliations, llm=llm, search=search)
            save_json([s.model_dump() for s in statuses], status_path)
            logger.debug("Saved author statuses to %s.", status_path.name)

            progress.advance(task)
