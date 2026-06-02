from __future__ import annotations

import pytest

from candidex.columns import (
    AUTHOR_AFFILIATION,
    AUTHOR_NAME,
    AUTHORS,
    PAPER_PDF_URL,
    PAPER_STEM,
    PAPER_TITLE,
    PAPER_URL,
)

ALL_COLUMNS = [
    AUTHORS,
    AUTHOR_AFFILIATION,
    AUTHOR_NAME,
    PAPER_PDF_URL,
    PAPER_STEM,
    PAPER_TITLE,
    PAPER_URL,
]


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        pytest.param(AUTHORS, "authors", id="authors"),
        pytest.param(AUTHOR_AFFILIATION, "author_affiliation", id="author_affiliation"),
        pytest.param(AUTHOR_NAME, "author_name", id="author_name"),
        pytest.param(PAPER_PDF_URL, "paper_pdf_url", id="paper_pdf_url"),
        pytest.param(PAPER_STEM, "paper_stem", id="paper_stem"),
        pytest.param(PAPER_TITLE, "paper_title", id="paper_title"),
        pytest.param(PAPER_URL, "paper_url", id="paper_url"),
    ],
)
def test_column_value(column: str, expected: str) -> None:
    assert column == expected


def test_all_columns_are_unique() -> None:
    assert len(ALL_COLUMNS) == len(set(ALL_COLUMNS))


def test_all_columns_are_strings() -> None:
    assert all(isinstance(c, str) for c in ALL_COLUMNS)


def test_all_columns_are_lowercase() -> None:
    assert all(c == c.lower() for c in ALL_COLUMNS)


def test_all_columns_use_underscores() -> None:
    assert all(" " not in c for c in ALL_COLUMNS)
