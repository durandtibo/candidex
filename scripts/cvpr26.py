# noqa: INP001
r"""Define the code for CVPR 2026."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from rich.logging import RichHandler
from rich.pretty import pprint

from candidex.affiliation import flatten_authors
from candidex.author import Author, authors_to_dataframe, deduplicate_authors
from candidex.author.frame import (
    add_openreview_profile_ids_to_dataframe,
    add_openreview_profiles_to_dataframe,
)
from candidex.chat_model import create_chat_model
from candidex.columns import AUTHOR_NAME, PAPER_PDF_URL
from candidex.config import ChatModelConfig
from candidex.openreview import (
    create_client,
    extract_profile_ids_by_author,
    extract_profiles_by_author,
    log_profile_ids_stats,
    log_profiles_by_author_stats,
    parse_names_and_history_profile,
)
from candidex.sandbox.affiliation import (
    AFFILIATION_SYSTEM_PROMPT,
    extract_and_save_affiliations,
    load_affiliations,
)
from candidex.sandbox.cvpr import find_and_save_papers
from candidex.sandbox.download import download_papers
from candidex.sandbox.role import (
    ROLE_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    import polars as pl


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


def extract_openreview_profile_ids(
    papers: pl.DataFrame, affiliation_dir: Path, profile_ids_dir: Path
) -> dict[Author, list[str] | None]:
    affiliations = load_affiliations(papers=papers, affiliation_dir=affiliation_dir)
    authors = deduplicate_authors(
        [x.to_author() for x in flatten_authors(list(affiliations.values()))]
    )

    profile_ids_by_author = extract_profile_ids_by_author(
        authors=authors, profile_ids_dir=profile_ids_dir
    )
    log_profile_ids_stats(profile_ids_by_author)
    return profile_ids_by_author


def main() -> None:
    r"""Define the main function."""
    data_path = Path(__file__).parent.parent.joinpath("data/v0").joinpath("cvpr26").resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    llm_affiliation_config = ChatModelConfig(
        model="ollama:gemma3:4b",
        # model="ollama:gemma3:12b",
        # model="ollama:gemma4:latest",
        # model="anthropic:claude-haiku-4-5-20251001",
        system_prompt=AFFILIATION_SYSTEM_PROMPT,
        temperature=0.0,
    )

    ChatModelConfig(
        model="ollama:gemma3:4b",
        # model="ollama:gemma3:12b",
        # model="ollama:gemma4:latest",
        # model="anthropic:claude-haiku-4-5-20251001",
        system_prompt=ROLE_SYSTEM_PROMPT,
        temperature=0.0,
    )

    pdf_dir = data_path.joinpath("pdf")
    affiliation_dir = data_path.joinpath("affiliation").joinpath(llm_affiliation_config.hash())
    openreview_dir = (
        data_path.joinpath("openreview")
        .joinpath("20260602")
        .joinpath(llm_affiliation_config.hash())
    )
    openreview_ids_dir = openreview_dir.joinpath("ids")
    openreview_profiles_dir = openreview_dir.joinpath("profiles")

    # role_dir = data_path.joinpath("role").joinpath(
    #     combine_hashes([llm_affiliation_config.hash(), llm_role_config.hash()])
    # )

    papers = find_and_save_papers(
        url="https://openaccess.thecvf.com/CVPR2026?day=all",
        filepath=data_path.joinpath("papers.parquet"),
    ).head(10)

    # remove_unreadable_pdfs(papers=papers, pdf_dir=pdf_dir)
    download_papers(urls=papers[PAPER_PDF_URL].to_list(), output_path=pdf_dir)

    extract_and_save_affiliations(
        papers=papers,
        llm=create_chat_model(llm_affiliation_config),
        pdf_dir=pdf_dir,
        affiliation_dir=affiliation_dir,
    )
    # affiliations = load_affiliations(papers=papers, affiliation_dir=affiliation_dir)
    # logger.info(affiliations)

    client = create_client()

    profile_ids_by_author = extract_openreview_profile_ids(
        papers=papers, affiliation_dir=affiliation_dir, profile_ids_dir=openreview_ids_dir
    )

    profiles_by_author = extract_profiles_by_author(
        profile_ids_by_author, profiles_dir=openreview_profiles_dir, client=client
    )
    log_profiles_by_author_stats(profiles_by_author)
    for profiles in profiles_by_author.values():
        if profiles:
            pprint(parse_names_and_history_profile(profiles[0]))
        break

    affiliations = load_affiliations(papers=papers, affiliation_dir=affiliation_dir)
    authors = deduplicate_authors(
        [x.to_author() for x in flatten_authors(list(affiliations.values()))]
    )
    frame = authors_to_dataframe(authors, include_id=True).sort(by=AUTHOR_NAME)
    frame = add_openreview_profile_ids_to_dataframe(frame, profile_ids_by_author)
    frame = add_openreview_profiles_to_dataframe(frame, profiles_by_author)
    pprint(frame)


if __name__ == "__main__":
    load_dotenv()
    main()
