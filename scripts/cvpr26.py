r"""Define the code for CVPR 2026."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from candidex.chat_model import create_chat_model
from candidex.columns import PAPER_PDF_URL
from candidex.config import ChatModelConfig
from candidex.sandbox.affiliation import (
    AFFILIATION_SYSTEM_PROMPT,
    extract_and_save_affiliations,
    load_affiliations,
)
from candidex.sandbox.cvpr import find_and_save_papers
from candidex.sandbox.download import download_papers
from candidex.sandbox.role import (
    ROLE_SYSTEM_PROMPT,
    find_and_save_authors_role,
    load_author_roles,
)
from candidex.utils.hashing import combine_hashes

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger: logging.Logger = logging.getLogger(__name__)


logging.getLogger("ddgs").setLevel(logging.WARNING)
logging.getLogger("duckduckgo_search").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)


def main() -> None:
    r"""Define the main function."""
    data_path = Path(__file__).parent.parent.joinpath("data/v0").joinpath("cvpr26").resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    llm_affiliation_config = ChatModelConfig(
        # model="ollama:gemma3:12b",
        model="ollama:gemma4:latest",
        system_prompt=AFFILIATION_SYSTEM_PROMPT,
        temperature=0.0,
    )

    llm_role_config = ChatModelConfig(
        # model="ollama:gemma3:12b",
        model="ollama:gemma4:latest",
        system_prompt=ROLE_SYSTEM_PROMPT,
        temperature=0.0,
    )

    pdf_dir = data_path.joinpath("pdf")
    affiliation_dir = data_path.joinpath("affiliation").joinpath(llm_affiliation_config.hash())
    role_dir = data_path.joinpath("role").joinpath(
        combine_hashes([llm_affiliation_config.hash(), llm_role_config.hash()])
    )

    papers = find_and_save_papers(
        url="https://openaccess.thecvf.com/CVPR2026?day=all",
        filepath=data_path.joinpath("papers.parquet"),
    ).head(10)
    logger.info(papers)

    download_papers(urls=papers[PAPER_PDF_URL].to_list(), output_path=pdf_dir)

    extract_and_save_affiliations(
        papers=papers,
        llm=create_chat_model(llm_affiliation_config),
        pdf_dir=pdf_dir,
        affiliation_dir=affiliation_dir,
    )
    affiliations = load_affiliations(papers=papers, affiliation_dir=affiliation_dir)
    logger.info(affiliations)

    find_and_save_authors_role(
        papers=papers,
        affiliations_dir=affiliation_dir,
        role_dir=role_dir,
        llm=create_chat_model(llm_role_config),
    )
    roles = load_author_roles(papers=papers, role_dir=role_dir)
    logger.info(roles)


if __name__ == "__main__":
    load_dotenv()
    main()
