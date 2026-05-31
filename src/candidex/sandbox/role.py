r"""Contain functionalities to find the current role of the authors."""

from __future__ import annotations

__all__ = ["find_and_save_authors_role", "load_author_roles"]

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from typing import TYPE_CHECKING

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from iden.io import load_json, load_text, save_json
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from candidex.columns import PAPER_STEM
from candidex.sandbox.affiliation import AuthorAffiliation, PaperAffiliations
from candidex.sandbox.progressbar import make_progressbar

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    from langchain_core.language_models import BaseChatModel


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
    MISSING = "Missing"


class AuthorRole(BaseModel):
    """Represents the current academic role and PhD history of a single
    author.

    PhD fields use the following sentinel strings instead of None:
    - 'UNKNOWN':     The author has a PhD but the information is unavailable.
    - 'NO PhD':      The author has never pursued a PhD.
    - 'In Progress': The author is currently a PhD student (phd_end_year only).
    - None:          Reserved for when role is AcademicRole.UNKNOWN or
                     AcademicRole.MISSING only.

    Attributes:
        name:           Full name of the author as it appears on the paper.
        affiliation:    Known institutional affiliation taken from the paper.
        role:           Current academic role from the `AcademicRole` enum.
                        Use AcademicRole.UNKNOWN if the role cannot be determined
                        from search results. AcademicRole.MISSING is reserved for
                        system use only and must never be assigned by the LLM.
        phd_start_year: 4-digit year the PhD started (e.g. '2019').
        phd_end_year:   4-digit year the PhD was completed (e.g. '2023').
        phd_domain:     Research field of the PhD (e.g. 'Computer Vision').
        phd_university: Institution where the PhD was or is being completed.
        details:        Additional context such as lab, supervisor, or department.
        source:         Single raw URL of the most authoritative source used.
    """

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
            "- Unknown: role cannot be determined from available information\n"
            "- Missing: reserved for system use only — never assign this value"
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


ROLE_SYSTEM_PROMPT = """
You are an expert at finding information about academic researchers.
Given an author's name and affiliation, determine their current academic role
and PhD information.

SEARCH STRATEGY:
Use the following search strategies in order of priority to find the most
accurate and up-to-date information:

1. Resume or CV search — most reliable source for PhD details and career history:
   "{author_name}" "{affiliation}" resume OR CV OR curriculum vitae

2. GitHub profile — useful for current role, affiliations listed in bio, and linked personal sites:
   "{author_name}" "{affiliation}" site:github.com

3. OpenReview — lists author affiliation at time of paper submission, useful for confirming
   institutional affiliation and academic role:
   "{author_name}" "{affiliation}" site:openreview.net

4. University or lab profile page — most reliable for current faculty and student roles:
   "{author_name}" "{affiliation}" (Master student OR PhD student OR postdoc OR professor OR researcher) -linkedin

5. Google Scholar — confirms academic role and institutional affiliation:
   "{author_name}" "{affiliation}" site:scholar.google.com

6. LinkedIn — current job title and employment history, useful for industry roles:
   "{author_name}" "{affiliation}" site:linkedin.com

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
- Unknown: insufficient information to determine the role — use this as a last resort only

Do NOT assign the 'Missing' role — it is reserved for system use only and will
be assigned automatically when a lookup fails entirely.

PHD INFORMATION RULES:
- If the author has completed or is currently pursuing a PhD, populate phd_start_year,
  phd_end_year, phd_domain, and phd_university with the best available information.
- Use 'UNKNOWN' for any PhD field that cannot be determined from the search results.
- Use 'In Progress' for phd_end_year if the author is currently a PhD student.
- Use 'NO PhD' for all PhD fields if the author has never pursued a PhD.
- Only use None for PhD fields if the role itself is 'Unknown' and no information
  is available at all.

DETAILS RULES:
- Use the details field to capture any additional context that does not fit in
  other fields, such as lab name, research group, supervisor, or university department.
- Keep it concise — 200 characters maximum.
- Write as a short phrase, not a full sentence (e.g. 'CSAIL, advised by Prof. Smith').
- Set to None if no additional context is available.

SOURCE RULES:
- The source field must contain exactly one URL — the single most authoritative
  source used to determine the role and PhD information.
- Choose the URL according to this priority:
  1. Personal CV or resume (PDF)
  2. GitHub profile bio and linked personal website
  3. OpenReview profile page
  4. University or lab profile page
  5. Google Scholar profile
  6. LinkedIn profile
  7. Any other credible web source
- Never include multiple URLs, comma-separated links, or prose descriptions in
  the source field. Only a single raw URL (e.g. 'https://example.com/cv.pdf').
- If no credible source was found, set source to None.

Only report years if explicitly stated in the source. Do not infer or estimate years. Do not be lazy.
Always record the URL of the source used in the source field.
"""


def _run_single_query(query: str, max_retries: int, backoff_factor: float) -> str | None:
    """Run a single DuckDuckGo query with retries.

    Returns formatted results or None.
    """
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=3)
            if results:
                return "\n\n".join(
                    f"URL: {r['href']}\nTitle: {r['title']}\n{r['body']}" for r in results
                )
            return None
        except DDGSException as e:
            wait = backoff_factor**attempt
            if attempt < max_retries - 1:
                logger.debug(
                    "Search failed for query '%s' (attempt %d/%d): %s. Retrying in %.0fs...",
                    query,
                    attempt + 1,
                    max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Search failed for query '%s' after %d attempts: %s.",
                    query,
                    max_retries,
                    e,
                )
    return None


def _run_searches(
    author: AuthorAffiliation,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    max_workers: int = 3,
) -> str:
    """Run multiple targeted DuckDuckGo searches for an author in
    parallel and return combined results.

    Executes six queries concurrently using a thread pool, each targeting
    a different source type in order of reliability for academic role and
    PhD information:

    1. CV/resume PDF — most reliable for PhD details and career history.
    2. GitHub profile — bio and linked personal site often contain current role.
    3. OpenReview — confirms academic affiliation at time of paper submission.
    4. University or lab profile page — role and department details.
    5. Google Scholar — confirms academic role and institutional affiliation.
    6. LinkedIn — current job title and employment history.

    Results are reassembled in query order regardless of completion order,
    so the LLM always receives results from more reliable sources first.
    Each query is retried up to `max_retries` times with exponential backoff
    on timeout or connectivity errors.

    Args:
        author:         An `AuthorAffiliation` object containing the author's
                        name and known affiliations.
        max_retries:    Maximum number of retry attempts per query on timeout
                        or connectivity errors. Defaults to 3.
        backoff_factor: Multiplier for the wait time between retries.
                        Wait times are: 2s, 4s, 8s, ... Defaults to 2.0.
        max_workers:    Maximum number of concurrent threads. Defaults to 3
                        to avoid rate limiting from DuckDuckGo.

    Returns:
        A formatted string of search results from all queries, separated by
        a divider, in query priority order. Returns an empty string if all
        queries return no results after retries.
    """
    affiliation_str = ", ".join(author.affiliations) if author.affiliations else "Unknown"
    name = author.author

    queries = [
        # 1. CV/resume — most reliable for PhD details and career history
        f'"{name}" "{affiliation_str}" resume OR CV OR curriculum vitae',
        # 2. GitHub — bio and linked personal site often contain current role
        f'"{name}" "{affiliation_str}" site:github.com',
        # 3. OpenReview — confirms academic affiliation at time of submission
        f'"{name}" "{affiliation_str}" site:openreview.net',
        # 4. University or lab profile page
        f'"{name}" "{affiliation_str}" (Master student OR PhD student OR postdoc OR professor OR researcher) -linkedin',
        # 5. Google Scholar — confirms academic role and institutional affiliation
        f'"{name}" "{affiliation_str}" site:scholar.google.com',
        # 6. LinkedIn — current job title and employment history
        f'"{name}" "{affiliation_str}" site:linkedin.com',
    ]

    # Map future -> original index to reassemble results in priority order
    index_map = {}
    sections = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, query in enumerate(queries):
            future = executor.submit(_run_single_query, query, max_retries, backoff_factor)
            index_map[future] = i

        for future in as_completed(index_map):
            i = index_map[future]
            result = future.result()
            if result:
                sections[i] = result

    return "\n\n---\n\n".join(s for s in sections if s)


def find_author_role(author: AuthorAffiliation, llm: BaseChatModel) -> AuthorRole:
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

    Returns:
        An `AuthorRole` object with the author's current role, PhD details,
        and source URL. If no information can be found, `role` is set to
        `AcademicRole.UNKNOWN` and all PhD fields are None.

    Raises:
        langchain_core.exceptions.LangChainException: If the LLM call fails.
        duckduckgo_search.exceptions.DDGSException:   On DuckDuckGo connectivity errors.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> author = AuthorAffiliation(author="Jane Smith", affiliations=["MIT CSAIL"])
        >>> role = find_author_role(author, llm)
        >>> print(role.role, role.phd_start_year)
    """
    affiliation_str = ", ".join(author.affiliations) if author.affiliations else "Unknown"
    logger.debug("Searching for author role: %s (%s).", author.author, affiliation_str)

    search_text = _run_searches(author)
    logger.debug("Search results retrieved for %s.", author.author)

    structured_llm = llm.with_structured_output(AuthorRole)
    messages = [
        SystemMessage(content=ROLE_SYSTEM_PROMPT),
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


def find_authors_role(
    affiliations: PaperAffiliations,
    llm: BaseChatModel,
) -> list[AuthorRole]:
    """Find the current academic role for all authors in a paper.

    Iterates over each author in the provided `PaperAffiliations`, performs
    DuckDuckGo searches for their current position, and uses an LLM to extract
    structured role and PhD information. If the lookup fails for an author, an
    `AuthorRole` with `role` set to `AcademicRole.MISSING` is returned for that
    author so the full author list is always preserved in the output.

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
        A list of `AuthorRole` objects, one per author. Authors for whom the
        lookup failed have `role` set to `AcademicRole.MISSING` and all PhD
        fields set to None.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> affiliations = extract_affiliations(Path("papers/attention.pdf"), llm)
        >>> roles = find_authors_role(affiliations, llm)
        >>> for r in roles:
        ...     print(r.name, r.role, r.phd_start_year, r.phd_end_year)
    """
    results = []
    total = len(affiliations.authors)

    with make_progressbar() as progress:
        task = progress.add_task("Finding author roles", total=total)
        for author in affiliations.authors:
            try:
                role = find_author_role(author, llm=llm)
            except Exception as e:
                logger.warning("Failed to find role for %s: %s", author.author, e)
                affiliation_str = (
                    ", ".join(author.affiliations) if author.affiliations else "Unknown"
                )
                role = AuthorRole(
                    name=author.author,
                    affiliation=affiliation_str,
                    role=AcademicRole.MISSING,
                    phd_start_year=None,
                    phd_end_year=None,
                    phd_domain=None,
                    phd_university=None,
                    details=None,
                    source=None,
                )
            results.append(role)
            progress.advance(task)

    resolved = sum(1 for r in results if r.role not in {AcademicRole.UNKNOWN, AcademicRole.MISSING})
    unknown = sum(1 for r in results if r.role == AcademicRole.UNKNOWN)
    missing = sum(1 for r in results if r.role == AcademicRole.MISSING)
    logger.info(
        "Author role lookup complete. %d/%d resolved, %d unknown, %d missing.",
        resolved,
        total,
        unknown,
        missing,
    )
    return results


def find_and_save_authors_role(
    papers: pl.DataFrame,
    affiliations_dir: Path,
    role_dir: Path,
    llm: BaseChatModel,
    max_authors: int = 20,
) -> None:
    """Find and save the academic role of all authors for each paper to
    a JSON file.

    For each paper in the DataFrame, loads the corresponding affiliations JSON
    file, looks up the current academic role of each author, and writes the
    results to a JSON file named after the paper. Papers whose role JSON file
    already exists are skipped, making the function safe to call repeatedly and
    resilient to interruptions.

    Only the first `max_authors` authors per paper are processed. This is
    useful for large papers with many authors where processing all of them
    would be too slow or costly.

    The JSON file for a paper with stem `paper` will be saved as `paper.json`
    in `role_dir`. Each file contains a list of serialised `AuthorRole` objects.

    Args:
        papers:           Polars DataFrame produced by `scrape_cvpr_papers` or
                          equivalent. Must contain a column named by `PAPER_STEM`
                          with the PDF filename stem for each paper.
        affiliations_dir: Directory containing the affiliation JSON files produced
                          by `extract_and_save_affiliations`. Each file must be
                          named `{stem}.json` where `stem` matches `PAPER_STEM`.
        role_dir:         Directory where author role JSON files will be saved.
                          Created automatically if it does not exist.
        llm:              Any LangChain-compatible chat model with structured output
                          support. The caller is responsible for initialising and
                          configuring it.
        max_authors:      Maximum number of authors to process per paper, taking
                          the first N authors in the order they appear. Defaults
                          to 20. Set to None to process all authors.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> find_and_save_authors_role(
        ...     papers=df,
        ...     affiliations_dir=Path("papers/cvpr2024/affiliations"),
        ...     role_dir=Path("papers/cvpr2024/roles"),
        ...     llm=llm,
        ...     max_authors=10,
        ... )
    """
    logger.info("Finding authors roles and saving them at %s...", role_dir)
    role_dir.mkdir(parents=True, exist_ok=True)
    rows = list(papers.iter_rows(named=True))

    with make_progressbar() as progress:
        task = progress.add_task("Processing papers", total=len(rows))
        for row in rows:
            stem = row[PAPER_STEM]
            affiliation_path = affiliations_dir.joinpath(f"{stem}.json")
            role_path = role_dir.joinpath(f"{stem}.json")

            if role_path.exists():
                logger.debug("Skipping %s, role file already exists.", stem)
                progress.advance(task)
                continue

            if not affiliation_path.exists():
                logger.warning("Affiliation file not found, skipping: %s.", affiliation_path.name)
                progress.advance(task)
                continue

            affiliations = PaperAffiliations.model_validate_json(load_text(affiliation_path))
            affiliations.authors = affiliations.authors[:max_authors]

            if len(affiliations.authors) < len(
                PaperAffiliations.model_validate_json(load_text(affiliation_path)).authors
            ):
                logger.debug(
                    "Processing first %d of %d authors for %s.",
                    len(affiliations.authors),
                    len(PaperAffiliations.model_validate_json(load_text(affiliation_path)).authors),
                    stem,
                )

            roles = find_authors_role(affiliations, llm=llm)
            save_json([r.model_dump() for r in roles], role_path)
            logger.debug("Saved author roles to %s.", role_path.name)

            progress.advance(task)


def retry_authors_role(
    papers: pl.DataFrame,
    role_dir: Path,
    llm: BaseChatModel,
    retry_role: AcademicRole = AcademicRole.MISSING,
    max_authors: int = 20,
) -> None:
    """Retry finding the academic role for authors with a specific role
    value.

    For each paper in the DataFrame, loads the existing role JSON file, finds
    all authors whose role matches `retry_role`, re-runs the role lookup for
    those authors, and saves the updated results back to the same file. Authors
    whose role does not match `retry_role` are left unchanged.

    Only the first `max_authors` authors per paper are considered for retry.
    Authors beyond this limit are left unchanged regardless of their role.

    Useful for recovering from failed lookups (`AcademicRole.MISSING`) or
    refining ambiguous results (`AcademicRole.UNKNOWN`) without re-processing
    the entire dataset.

    Args:
        papers:      Polars DataFrame produced by `scrape_cvpr_papers` or
                     equivalent. Must contain a column named by `PAPER_STEM`
                     with the PDF filename stem for each paper.
        role_dir:    Directory containing the role JSON files produced by
                     `find_and_save_authors_role`. Each file must be named
                     `{stem}.json` where `stem` matches `PAPER_STEM`. Files
                     are updated in place.
        llm:         Any LangChain-compatible chat model with structured output
                     support. The caller is responsible for initialising and
                     configuring it.
        retry_role:  The `AcademicRole` value that triggers a retry. Only
                     authors whose current role matches this value will be
                     re-processed. Defaults to `AcademicRole.MISSING`.
        max_authors: Maximum number of authors to consider per paper, taking
                     the first N authors in the order they appear. Authors
                     beyond this limit are left unchanged. Defaults to 20.
                     Set to None to consider all authors.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        >>> retry_authors_role(df, Path("data/cvpr2024/roles"), llm)
        >>> retry_authors_role(
        ...     df,
        ...     Path("data/cvpr2024/roles"),
        ...     llm,
        ...     retry_role=AcademicRole.UNKNOWN,
        ...     max_authors=10,
        ... )
    """
    logger.info("Retrying author role lookup for role '%s'...", retry_role)
    rows = list(papers.iter_rows(named=True))

    with make_progressbar() as progress:
        task = progress.add_task(f"Retrying '{retry_role}' authors", total=len(rows))
        for row in rows:
            stem = row[PAPER_STEM]
            role_path = role_dir.joinpath(f"{stem}.json")

            if not role_path.is_file():
                logger.debug("Role file not found, skipping: %s.", role_path.name)
                progress.advance(task)
                continue

            try:
                roles = [AuthorRole.model_validate(r) for r in load_json(role_path)]
            except Exception as e:
                logger.warning("Failed to load role file %s: %s", role_path.name, e)
                progress.advance(task)
                continue

            to_retry = [r for r in roles[:max_authors] if r.role == retry_role]
            if not to_retry:
                progress.advance(task)
                continue

            logger.debug(
                "Retrying %d/%d authors in %s.",
                len(to_retry),
                len(roles),
                role_path.name,
            )

            role_map = {r.name: r for r in roles}
            for author_role in to_retry:
                author = AuthorAffiliation(
                    author=author_role.name,
                    affiliations=[author_role.affiliation],
                )
                try:
                    updated = find_author_role(author, llm=llm)
                except Exception as e:
                    logger.warning("Failed to retry role for %s: %s", author_role.name, e)
                    updated = author_role
                role_map[author_role.name] = updated

            updated_roles = [role_map[r.name] for r in roles]
            save_json([r.model_dump() for r in updated_roles], role_path)
            logger.debug("Updated role file %s.", role_path.name)

            progress.advance(task)

    logger.info("Retry complete for role '%s'.", retry_role)


def load_author_roles(papers: pl.DataFrame, role_dir: Path) -> dict[str, list[AuthorRole]]:
    """Load author roles from JSON files into a dictionary.

    Iterates over a DataFrame of papers and loads the corresponding role JSON
    file for each paper. Papers whose JSON file does not exist are logged and
    skipped. Useful for downstream processing without re-running the role
    lookup step.

    Args:
        papers:   Polars DataFrame produced by `scrape_cvpr_papers` or
                  equivalent. Must contain a column named by `PAPER_STEM`
                  with the PDF filename stem (i.e. without the `.pdf`
                  extension) for each paper.
        role_dir: Directory containing the role JSON files produced by
                  `find_and_save_authors_role`. Each file must be named
                  `{stem}.json` where `stem` matches the value in the
                  `PAPER_STEM` column.

    Returns:
        A dictionary mapping each paper stem to its list of `AuthorRole`
        objects. Papers with missing or unreadable JSON files are omitted
        from the result.

    Example:
        >>> df = scrape_cvpr_papers("https://openaccess.thecvf.com/CVPR2024?day=all")
        >>> roles = load_author_roles(df, Path("data/cvpr2024/roles"))
        >>> roles["attention_is_all_you_need"][0].role
    """
    logger.info("Loading author roles from %s...", role_dir)

    rows = list(papers.iter_rows(named=True))
    roles: dict[str, list[AuthorRole]] = {}

    with make_progressbar() as progress:
        task = progress.add_task("Loading author roles", total=len(rows))
        for row in rows:
            stem = row[PAPER_STEM]
            role_path = role_dir.joinpath(f"{stem}.json")

            if not role_path.is_file():
                logger.warning("Role file not found, skipping: %s.", role_path.name)
                progress.advance(task)
                continue

            roles[stem] = [AuthorRole.model_validate(r) for r in load_json(role_path)]
            progress.advance(task)

    total_authors = sum(len(r) for r in roles.values())
    resolved = sum(
        1
        for paper_roles in roles.values()
        for r in paper_roles
        if r.role not in {AcademicRole.UNKNOWN, AcademicRole.MISSING}
    )
    unknown = sum(
        1 for paper_roles in roles.values() for r in paper_roles if r.role == AcademicRole.UNKNOWN
    )
    missing = sum(
        1 for paper_roles in roles.values() for r in paper_roles if r.role == AcademicRole.MISSING
    )
    logger.info(
        "Loaded roles for %d/%d papers — %d/%d authors resolved, %d unknown, %d missing.",
        len(roles),
        len(papers),
        resolved,
        total_authors,
        unknown,
        missing,
    )
    return roles
