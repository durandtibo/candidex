from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from candidex.paper import Paper

# --- Fixtures ---


@pytest.fixture
def paper() -> Paper:
    return Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


###########################
#     Tests for Paper     #
###########################

# --- Construction ---


def test_paper_from_raw_title_only() -> None:
    paper = Paper.from_raw(title="Attention Is All You Need")
    assert paper.title == "Attention Is All You Need"
    assert paper.authors is None
    assert paper.venue is None
    assert paper.year is None
    assert paper.pdf_url is None


def test_paper_from_raw_all_fields(paper: Paper) -> None:
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ("Ashish Vaswani", "Noam Shazeer")
    assert paper.venue == "NeurIPS"
    assert paper.year == 2017
    assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762"


def test_paper_from_raw_raises_on_empty_title() -> None:
    with pytest.raises(ValueError, match="title"):
        Paper.from_raw(title="   ")


# --- Optional fields ---


def test_paper_from_raw_none_authors() -> None:
    assert Paper.from_raw(title="My Paper", authors=None).authors is None


def test_paper_from_raw_empty_authors() -> None:
    assert Paper.from_raw(title="My Paper", authors=[]).authors == ()


def test_paper_from_raw_none_venue() -> None:
    assert Paper.from_raw(title="My Paper", venue=None).venue is None


def test_paper_from_raw_none_year() -> None:
    assert Paper.from_raw(title="My Paper", year=None).year is None


def test_paper_from_raw_none_pdf_url() -> None:
    assert Paper.from_raw(title="My Paper", pdf_url=None).pdf_url is None


# --- Normalization ---


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        pytest.param(
            "  Attention Is All You Need  ", "Attention Is All You Need", id="strips_whitespace"
        ),
        pytest.param(
            "Réseaux de Neurones Profonds", "Reseaux de Neurones Profonds", id="normalizes_unicode"
        ),
        pytest.param("  Réseaux de Neurones  ", "Reseaux de Neurones", id="strips_and_normalizes"),
    ],
)
def test_paper_from_raw_normalizes_title(title: str, expected: str) -> None:
    assert Paper.from_raw(title=title).title == expected


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        pytest.param(["  Ashish Vaswani  "], ("Ashish Vaswani",), id="strips_whitespace"),
        pytest.param(["Université Smith"], ("Universite Smith",), id="normalizes_unicode"),
    ],
)
def test_paper_from_raw_normalizes_authors(authors: list[str], expected: tuple[str, ...]) -> None:
    assert Paper.from_raw(title="My Paper", authors=authors).authors == expected


@pytest.mark.parametrize(
    ("venue", "expected"),
    [
        pytest.param("  NeurIPS  ", "NeurIPS", id="strips_whitespace"),
        pytest.param("École Normale", "Ecole Normale", id="normalizes_unicode"),
    ],
)
def test_paper_from_raw_normalizes_venue(venue: str, expected: str) -> None:
    assert Paper.from_raw(title="My Paper", venue=venue).venue == expected


def test_paper_from_raw_normalizes_pdf_url() -> None:
    assert (
        Paper.from_raw(title="My Paper", pdf_url="  https://arxiv.org/pdf/1706.03762  ").pdf_url
        == "https://arxiv.org/pdf/1706.03762"
    )


# --- Hashability and equality ---


def test_paper_is_hashable(paper: Paper) -> None:
    assert hash(paper) is not None


def test_paper_can_be_used_as_dict_key(paper: Paper) -> None:
    assert {paper: "value"}[paper] == "value"


def test_paper_can_be_used_in_set() -> None:
    a = Paper.from_raw(title="My Paper")
    b = Paper.from_raw(title="My Paper")
    assert len({a, b}) == 1


def test_paper_equality_same_fields(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper == other


def test_paper_inequality_different_title(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="BERT",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_different_authors(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_none_vs_authors(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=None,
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_different_venue(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="ICML",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_none_vs_venue(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue=None,
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_different_year(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2018,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_none_vs_year(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=None,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


def test_paper_inequality_different_pdf_url(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/other.pdf",
    )


def test_paper_inequality_none_vs_pdf_url(paper: Paper) -> None:
    assert paper != Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url=None,
    )


# --- Immutability ---


def test_paper_is_frozen(paper: Paper) -> None:
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'title'"):
        paper.title = "BERT"


# --- hash method ---


def test_paper_hash_returns_string(paper: Paper) -> None:
    assert isinstance(paper.hash(), str)


def test_paper_hash_is_64_char_lowercase_hex(paper: Paper) -> None:
    digest = paper.hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_paper_hash_same_paper_same_hash(paper: Paper) -> None:
    assert paper.hash() == paper.hash()


def test_paper_hash_equal_papers_same_hash(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper.hash() == other.hash()


def test_paper_hash_none_authors_different_from_empty_authors() -> None:
    a = Paper.from_raw(title="My Paper", authors=None)
    b = Paper.from_raw(title="My Paper", authors=[])
    assert a.hash() != b.hash()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            Paper.from_raw("My Paper"), Paper.from_raw("Other Paper"), id="different_title"
        ),
        pytest.param(
            Paper.from_raw("My Paper", authors=["Jane"]),
            Paper.from_raw("My Paper", authors=["John"]),
            id="different_authors",
        ),
        pytest.param(
            Paper.from_raw("My Paper", authors=["Jane"]),
            Paper.from_raw("My Paper", authors=None),
            id="authors_none_vs_value",
        ),
        pytest.param(
            Paper.from_raw("My Paper", venue="NeurIPS"),
            Paper.from_raw("My Paper", venue="ICML"),
            id="different_venue",
        ),
        pytest.param(
            Paper.from_raw("My Paper", venue="NeurIPS"),
            Paper.from_raw("My Paper", venue=None),
            id="venue_none_vs_value",
        ),
        pytest.param(
            Paper.from_raw("My Paper", year=2017),
            Paper.from_raw("My Paper", year=2018),
            id="different_year",
        ),
        pytest.param(
            Paper.from_raw("My Paper", year=2017),
            Paper.from_raw("My Paper", year=None),
            id="year_none_vs_value",
        ),
        pytest.param(
            Paper.from_raw("My Paper", pdf_url="https://a.pdf"),
            Paper.from_raw("My Paper", pdf_url="https://b.pdf"),
            id="different_pdf_url",
        ),
        pytest.param(
            Paper.from_raw("My Paper", pdf_url="https://a.pdf"),
            Paper.from_raw("My Paper", pdf_url=None),
            id="pdf_url_none_vs_value",
        ),
    ],
)
def test_paper_hash_different_papers_different_hash(a: Paper, b: Paper) -> None:
    assert a.hash() != b.hash()
