from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from candidex.paper import Paper

# --- Fixtures ---


@pytest.fixture
def paper() -> Paper:
    return Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


###########################
#     Tests for Paper     #
###########################

# --- Construction ---


def test_paper_from_raw_basic(paper: Paper) -> None:
    assert paper.title == "Attention Is All You Need"
    assert paper.venue == "NeurIPS"
    assert paper.year == 2017
    assert paper.pdf_url == "https://arxiv.org/pdf/1706.03762"


def test_paper_from_raw_raises_on_empty_title() -> None:
    with pytest.raises(ValueError, match="Paper title cannot be empty"):
        Paper.from_raw(
            title="   ", venue="NeurIPS", year=2017, pdf_url="https://arxiv.org/pdf/1706.03762"
        )


def test_paper_from_raw_raises_on_empty_venue() -> None:
    with pytest.raises(ValueError, match="Paper venue cannot be empty"):
        Paper.from_raw(
            title="Attention Is All You Need",
            venue="   ",
            year=2017,
            pdf_url="https://arxiv.org/pdf/1706.03762",
        )


def test_paper_from_raw_raises_on_empty_pdf_url() -> None:
    with pytest.raises(ValueError, match="Paper PDF URL cannot be empty"):
        Paper.from_raw(title="Attention Is All You Need", venue="NeurIPS", year=2017, pdf_url="   ")


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
    paper = Paper.from_raw(
        title=title, venue="NeurIPS", year=2017, pdf_url="https://arxiv.org/pdf/1706.03762"
    )
    assert paper.title == expected


@pytest.mark.parametrize(
    ("venue", "expected"),
    [
        pytest.param("  NeurIPS  ", "NeurIPS", id="strips_whitespace"),
        pytest.param("École Normale", "Ecole Normale", id="normalizes_unicode"),
    ],
)
def test_paper_from_raw_normalizes_venue(venue: str, expected: str) -> None:
    paper = Paper.from_raw(
        title="Attention Is All You Need",
        venue=venue,
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper.venue == expected


@pytest.mark.parametrize(
    ("pdf_url", "expected"),
    [
        pytest.param(
            "  https://arxiv.org/pdf/1706.03762  ",
            "https://arxiv.org/pdf/1706.03762",
            id="strips_whitespace",
        ),
        pytest.param(
            "https://arxiv.org/pdf/1706.03762", "https://arxiv.org/pdf/1706.03762", id="unchanged"
        ),
    ],
)
def test_paper_from_raw_normalizes_pdf_url(pdf_url: str, expected: str) -> None:
    paper = Paper.from_raw(
        title="Attention Is All You Need", venue="NeurIPS", year=2017, pdf_url=pdf_url
    )
    assert paper.pdf_url == expected


# --- Hashability and equality ---


def test_paper_is_hashable(paper: Paper) -> None:
    assert hash(paper) is not None


def test_paper_can_be_used_as_dict_key(paper: Paper) -> None:
    d = {paper: "value"}
    assert d[paper] == "value"


def test_paper_can_be_used_in_set() -> None:
    a = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    b = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert len({a, b}) == 1


def test_paper_equality_same_fields(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper == other


def test_paper_inequality_different_title(paper: Paper) -> None:
    other = Paper.from_raw(
        title="BERT", venue="NeurIPS", year=2017, pdf_url="https://arxiv.org/pdf/1706.03762"
    )
    assert paper != other


def test_paper_inequality_different_venue(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        venue="ICML",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper != other


def test_paper_inequality_different_year(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2018,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper != other


def test_paper_inequality_different_pdf_url(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/other.pdf",
    )
    assert paper != other


# --- Immutability ---


def test_paper_is_frozen(paper: Paper) -> None:
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field 'title'"):
        paper.title = "BERT"  # type: ignore[misc]


# --- hash method ---


def test_paper_hash_returns_string(paper: Paper) -> None:
    assert isinstance(paper.hash(), str)


def test_paper_hash_is_128_char_lowercase_hex(paper: Paper) -> None:
    digest = paper.hash()
    assert len(digest) == 128
    assert all(c in "0123456789abcdef" for c in digest)


def test_paper_hash_same_paper_same_hash(paper: Paper) -> None:
    assert paper.hash() == paper.hash()


def test_paper_hash_equal_papers_same_hash(paper: Paper) -> None:
    other = Paper.from_raw(
        title="Attention Is All You Need",
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )
    assert paper.hash() == other.hash()


def test_paper_hash_matches_manual_blake2b(paper: Paper) -> None:
    canonical = json.dumps(
        {
            "title": paper.title,
            "venue": paper.venue,
            "year": paper.year,
            "pdf_url": paper.pdf_url,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    expected = hashlib.blake2b(canonical.encode()).hexdigest()
    assert paper.hash() == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2017, "https://arxiv.org/pdf/1706.03762"
            ),
            Paper.from_raw("BERT", "NeurIPS", 2017, "https://arxiv.org/pdf/1706.03762"),
            id="different_title",
        ),
        pytest.param(
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2017, "https://arxiv.org/pdf/1706.03762"
            ),
            Paper.from_raw(
                "Attention Is All You Need", "ICML", 2017, "https://arxiv.org/pdf/1706.03762"
            ),
            id="different_venue",
        ),
        pytest.param(
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2017, "https://arxiv.org/pdf/1706.03762"
            ),
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2018, "https://arxiv.org/pdf/1706.03762"
            ),
            id="different_year",
        ),
        pytest.param(
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2017, "https://arxiv.org/pdf/1706.03762"
            ),
            Paper.from_raw(
                "Attention Is All You Need", "NeurIPS", 2017, "https://arxiv.org/pdf/other.pdf"
            ),
            id="different_pdf_url",
        ),
    ],
)
def test_paper_hash_different_papers_different_hash(a: Paper, b: Paper) -> None:
    assert a.hash() != b.hash()
