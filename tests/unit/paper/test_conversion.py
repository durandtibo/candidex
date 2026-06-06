from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from candidex.columns import (
    PAPER_AUTHORS,
    PAPER_ID,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    PAPER_YEAR,
)
from candidex.paper import Paper, papers_to_dataframe

# --- Helpers ---


def make_paper(
    title: str = "Attention Is All You Need",
    authors: list[str] | None = None,
    venue: str = "NeurIPS",
    year: int = 2017,
    pdf_url: str = "https://arxiv.org/pdf/1706.03762",
) -> Paper:
    return Paper.from_raw(
        title=title,
        authors=authors or [],
        venue=venue,
        year=year,
        pdf_url=pdf_url,
    )


#########################################
#     Tests for papers_to_dataframe     #
#########################################


def test_papers_to_dataframe_empty_sequence() -> None:
    assert_frame_equal(
        papers_to_dataframe([]),
        pl.DataFrame(
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
            }
        ),
    )


def test_papers_to_dataframe_single_paper() -> None:
    paper = make_paper(authors=["Ashish Vaswani", "Noam Shazeer"])
    assert_frame_equal(
        papers_to_dataframe([paper]),
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need"],
                PAPER_AUTHORS: [["Ashish Vaswani", "Noam Shazeer"]],
                PAPER_VENUE: ["NeurIPS"],
                PAPER_YEAR: [2017],
                PAPER_URL: ["https://arxiv.org/pdf/1706.03762"],
            },
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
            },
        ),
    )


def test_papers_to_dataframe_empty_authors() -> None:
    paper = make_paper(authors=[])
    assert_frame_equal(
        papers_to_dataframe([paper]),
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need"],
                PAPER_AUTHORS: [[]],
                PAPER_VENUE: ["NeurIPS"],
                PAPER_YEAR: [2017],
                PAPER_URL: ["https://arxiv.org/pdf/1706.03762"],
            },
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
            },
        ),
    )


def test_papers_to_dataframe_multiple_papers() -> None:
    paper_a = make_paper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
    )
    paper_b = make_paper(
        title="BERT",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        venue="NAACL",
        year=2019,
        pdf_url="https://arxiv.org/pdf/1810.04805",
    )
    assert_frame_equal(
        papers_to_dataframe([paper_a, paper_b]),
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need", "BERT"],
                PAPER_AUTHORS: [
                    ["Ashish Vaswani", "Noam Shazeer"],
                    ["Jacob Devlin", "Ming-Wei Chang"],
                ],
                PAPER_VENUE: ["NeurIPS", "NAACL"],
                PAPER_YEAR: [2017, 2019],
                PAPER_URL: ["https://arxiv.org/pdf/1706.03762", "https://arxiv.org/pdf/1810.04805"],
            },
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
            },
        ),
    )


def test_papers_to_dataframe_preserves_order() -> None:
    paper_a = make_paper(title="Charlie Paper", authors=["Charlie"], venue="CVPR", year=2020)
    paper_b = make_paper(
        title="Alice Paper",
        authors=["Alice"],
        venue="ICLR",
        year=2021,
        pdf_url="https://arxiv.org/pdf/2.pdf",
    )
    paper_c = make_paper(
        title="Bob Paper",
        authors=["Bob"],
        venue="NeurIPS",
        year=2022,
        pdf_url="https://arxiv.org/pdf/3.pdf",
    )
    assert_frame_equal(
        papers_to_dataframe([paper_a, paper_b, paper_c]),
        pl.DataFrame(
            {
                PAPER_TITLE: ["Charlie Paper", "Alice Paper", "Bob Paper"],
                PAPER_AUTHORS: [["Charlie"], ["Alice"], ["Bob"]],
                PAPER_VENUE: ["CVPR", "ICLR", "NeurIPS"],
                PAPER_YEAR: [2020, 2021, 2022],
                PAPER_URL: [
                    "https://arxiv.org/pdf/1706.03762",
                    "https://arxiv.org/pdf/2.pdf",
                    "https://arxiv.org/pdf/3.pdf",
                ],
            },
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
            },
        ),
    )


def test_papers_to_dataframe_exclude_id_by_default() -> None:
    assert PAPER_ID not in papers_to_dataframe([make_paper()]).columns


def test_papers_to_dataframe_include_id_full_output() -> None:
    paper_a = make_paper(
        title="Attention Is All You Need", authors=["Ashish Vaswani"], venue="NeurIPS", year=2017
    )
    paper_b = make_paper(
        title="BERT",
        authors=["Jacob Devlin"],
        venue="NAACL",
        year=2019,
        pdf_url="https://arxiv.org/pdf/1810.04805",
    )
    assert_frame_equal(
        papers_to_dataframe([paper_a, paper_b], include_id=True),
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need", "BERT"],
                PAPER_AUTHORS: [["Ashish Vaswani"], ["Jacob Devlin"]],
                PAPER_VENUE: ["NeurIPS", "NAACL"],
                PAPER_YEAR: [2017, 2019],
                PAPER_URL: ["https://arxiv.org/pdf/1706.03762", "https://arxiv.org/pdf/1810.04805"],
                PAPER_ID: [paper_a.hash(), paper_b.hash()],
            },
            schema={
                PAPER_TITLE: pl.String,
                PAPER_AUTHORS: pl.List(pl.String),
                PAPER_VENUE: pl.String,
                PAPER_YEAR: pl.Int32,
                PAPER_URL: pl.String,
                PAPER_ID: pl.String,
            },
        ),
    )
