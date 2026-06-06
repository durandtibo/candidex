from __future__ import annotations

from unittest.mock import Mock, patch

import polars as pl
import pytest
from bs4 import BeautifulSoup, Tag
from polars.testing import assert_frame_equal

from candidex.columns import (
    PAPER_AUTHORS,
    PAPER_ID,
    PAPER_TITLE,
    PAPER_URL,
    PAPER_VENUE,
    PAPER_YEAR,
)
from candidex.paper import Paper
from candidex.scraper.cvf import (
    BASE_URL,
    build_listing_url,
    load_or_scrape_papers,
    parse_paper,
    parse_paper_entries,
    resolve_url,
    scrape_papers,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "candidex.scraper.cvf"

# --- Helpers ---


def make_cvf_html(papers: list[dict]) -> str:
    """Build a minimal CVF listing page HTML with the given papers.

    Each paper dict should have keys: title, authors, pdf_url.
    """
    entries = ""
    for paper in papers:
        entries += f"""
        <dt class="ptitle">
            <a href="{paper["url"]}">{paper["title"]}</a>
        </dt>
        <dd>
            <form>
                <input type="hidden" name="papername" value="{paper["title"]}">
            </form>
            <div class="bibref">{paper["authors"]}</div>
            <a href="{paper["pdf_url"]}">pdf</a>
        </dd>
        """
    return f"<html><body><dl>{entries}</dl></body></html>"


PAPER_A = {
    "title": "Attention Is All You Need",
    "url": "/content/CVPR2024/html/paper_a.html",
    "authors": "Jane Smith, John Doe",
    "pdf_url": "/content/CVPR2024/papers/paper_a.pdf",
}

PAPER_B = {
    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
    "url": "/content/CVPR2024/html/paper_b.html",
    "authors": "Alice Brown, Bob Jones",
    "pdf_url": "/content/CVPR2024/papers/paper_b.pdf",
}


def make_dt(html: str) -> Tag:
    """Parse a HTML snippet and return the first <dt> tag."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.find("dt")


def make_paper_html(
    title: str = "My Paper",
    paper_href: str = "/content/CVPR2024/html/paper.html",
    authors: str = "Jane Smith, John Doe",
    pdf_href: str = "/content/CVPR2024/papers/paper.pdf",
) -> str:
    return f"""
    <dl>
        <dt class="ptitle">
            <a href="{paper_href}">{title}</a>
        </dt>
        <dd>
            <div class="bibref">{authors}</div>
        </dd>
        <dd>
            <a href="{pdf_href}">pdf</a>
        </dd>
    </dl>
    """


#######################################
#     Tests for build_listing_url     #
#######################################


@pytest.mark.parametrize(
    ("venue", "year", "expected"),
    [
        pytest.param(
            "CVPR", 2024, "https://openaccess.thecvf.com/CVPR2024?day=all", id="cvpr_2024"
        ),
        pytest.param(
            "CVPR", 2023, "https://openaccess.thecvf.com/CVPR2023?day=all", id="cvpr_2023"
        ),
        pytest.param(
            "ICCV", 2023, "https://openaccess.thecvf.com/ICCV2023?day=all", id="iccv_2023"
        ),
        pytest.param(
            "WACV", 2022, "https://openaccess.thecvf.com/WACV2022?day=all", id="wacv_2022"
        ),
        pytest.param(
            "CVPR", 2020, "https://openaccess.thecvf.com/CVPR2020?day=all", id="cvpr_2020"
        ),
    ],
)
def test_build_listing_url(venue: str, year: int, expected: str) -> None:
    assert build_listing_url(venue, year) == expected


def test_build_listing_url_contains_day_all_query_param() -> None:
    assert "day=all" in build_listing_url("CVPR", 2024)


def test_build_listing_url_contains_venue() -> None:
    assert "CVPR" in build_listing_url("CVPR", 2024)


def test_build_listing_url_contains_year() -> None:
    assert "2024" in build_listing_url("CVPR", 2024)


def test_build_listing_url_starts_with_base_url() -> None:

    assert build_listing_url("CVPR", 2024).startswith(BASE_URL)


#########################################
#     Tests for parse_paper_entries     #
#########################################


# --- Empty and malformed HTML ---


def test_parse_paper_entries_empty_html() -> None:
    assert parse_paper_entries("") == []


def test_parse_paper_entries_no_paper_entries() -> None:
    assert parse_paper_entries("<html><body><p>No papers here.</p></body></html>") == []


def test_parse_paper_entries_malformed_html() -> None:
    assert parse_paper_entries("<html><body><dt>no class</dt></body></html>") == []


# --- Single paper ---


def test_parse_paper_entries_single_paper_count() -> None:
    html = make_cvf_html([PAPER_A])
    assert len(parse_paper_entries(html)) == 1


def test_parse_paper_entries_single_paper_tag_name() -> None:
    html = make_cvf_html([PAPER_A])
    entries = parse_paper_entries(html)
    assert entries[0].name == "dt"


def test_parse_paper_entries_single_paper_class() -> None:
    html = make_cvf_html([PAPER_A])
    entries = parse_paper_entries(html)
    assert "ptitle" in entries[0].get("class", [])


def test_parse_paper_entries_single_paper_title() -> None:
    html = make_cvf_html([PAPER_A])
    entries = parse_paper_entries(html)
    assert PAPER_A["title"] in entries[0].get_text()


# --- Multiple papers ---


def test_parse_paper_entries_multiple_papers_count() -> None:
    html = make_cvf_html([PAPER_A, PAPER_B])
    assert len(parse_paper_entries(html)) == 2


def test_parse_paper_entries_multiple_papers_preserves_order() -> None:
    html = make_cvf_html([PAPER_A, PAPER_B])
    entries = parse_paper_entries(html)
    assert PAPER_A["title"] in entries[0].get_text()
    assert PAPER_B["title"] in entries[1].get_text()


def test_parse_paper_entries_multiple_papers_all_have_ptitle_class() -> None:
    html = make_cvf_html([PAPER_A, PAPER_B])
    entries = parse_paper_entries(html)
    assert all("ptitle" in entry.get("class", []) for entry in entries)


# --- Does not match unrelated dt tags ---


def test_parse_paper_entries_ignores_dt_without_ptitle_class() -> None:
    html = """
    <html><body><dl>
        <dt class="ptitle"><a href="/paper">My Paper</a></dt>
        <dt class="other">Not a paper</dt>
        <dt>No class at all</dt>
    </dl></body></html>
    """
    assert len(parse_paper_entries(html)) == 1


#############################################
#         Tests for resolve_url             #
#############################################


@pytest.mark.parametrize(
    ("href", "base_url", "expected"),
    [
        pytest.param(
            "/content/CVPR2024/papers/paper.pdf",
            "https://openaccess.thecvf.com",
            "https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf",
            id="relative_href",
        ),
        pytest.param(
            "https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf",
            "https://openaccess.thecvf.com",
            "https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf",
            id="absolute_href_unchanged",
        ),
        pytest.param(
            "http://openaccess.thecvf.com/content/paper.pdf",
            "https://openaccess.thecvf.com",
            "http://openaccess.thecvf.com/content/paper.pdf",
            id="http_absolute_href_unchanged",
        ),
        pytest.param(
            "/paper.pdf",
            "https://example.com",
            "https://example.com/paper.pdf",
            id="relative_href_custom_base_url",
        ),
    ],
)
def test_resolve_url(href: str, base_url: str, expected: str) -> None:
    assert resolve_url(href, base_url) == expected


#############################################
#           Tests for parse_paper           #
#############################################


#############################################
#           Tests for parse_paper           #
#############################################


# --- Returns Paper object ---


def test_parse_paper_returns_paper_instance() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert isinstance(parse_paper(dt), Paper)


# --- Title ---


def test_parse_paper_title() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">Attention Is All You Need</a></dt>'
        "<dd>Jane Smith, John Doe</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).title == "Attention Is All You Need"


def test_parse_paper_title_stripped() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">  Attention Is All You Need  </a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).title == "Attention Is All You Need"


def test_parse_paper_title_fallback_when_no_a_tag() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle">Fallback Title</dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).title == "Fallback Title"


# --- Venue ---


def test_parse_paper_venue_none_by_default() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).venue is None


def test_parse_paper_venue() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt, venue="CVPR").venue == "CVPR"


# --- Year ---


def test_parse_paper_year_none_by_default() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).year is None


def test_parse_paper_year() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt, year=2024).year == 2024


# --- Authors ---


def test_parse_paper_authors_single() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).authors == ("Jane Smith",)


def test_parse_paper_authors_multiple() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith, John Doe, Alice Brown</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).authors == ("Jane Smith", "John Doe", "Alice Brown")


def test_parse_paper_authors_strips_whitespace() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>  Jane Smith  ,  John Doe  </dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).authors == ("Jane Smith", "John Doe")


def test_parse_paper_authors_collapses_internal_whitespace() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane  Smith,John   Doe</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).authors == ("Jane Smith", "John Doe")


def test_parse_paper_authors_none_when_no_dd() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>',
        "html.parser",
    ).dt
    assert parse_paper(dt).authors is None


# --- PDF URL ---


def test_parse_paper_pdf_url_relative() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).pdf_url == f"{BASE_URL}/content/CVPR2024/papers/paper.pdf"


def test_parse_paper_pdf_url_absolute() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="https://openaccess.thecvf.com/content/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).pdf_url == "https://openaccess.thecvf.com/content/paper.pdf"


def test_parse_paper_pdf_url_custom_base_url() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert (
        parse_paper(dt, base_url="https://custom.example.com").pdf_url
        == "https://custom.example.com/content/paper.pdf"
    )


def test_parse_paper_pdf_url_none_when_no_pdf_link() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/paper.html">html</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).pdf_url is None


def test_parse_paper_pdf_url_case_insensitive() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">My Paper</a></dt>'
        "<dd>Jane Smith</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.PDF">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt).pdf_url == f"{BASE_URL}/content/CVPR2024/papers/paper.PDF"


# --- Full Paper object ---


def test_parse_paper_full_output_with_venue_and_year() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">Attention Is All You Need</a></dt>'
        "<dd>Jane Smith, John Doe</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt, venue="CVPR", year=2024) == Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith", "John Doe"],
        venue="CVPR",
        year=2024,
        pdf_url=f"{BASE_URL}/content/CVPR2024/papers/paper.pdf",
    )


def test_parse_paper_full_output_without_venue_and_year() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">Attention Is All You Need</a></dt>'
        "<dd>Jane Smith, John Doe</dd>"
        '<dd><a href="/content/CVPR2024/papers/paper.pdf">pdf</a></dd>',
        "html.parser",
    ).dt
    assert parse_paper(dt) == Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith", "John Doe"],
        venue=None,
        year=None,
        pdf_url=f"{BASE_URL}/content/CVPR2024/papers/paper.pdf",
    )


def test_parse_paper_full_output_missing_authors_and_pdf() -> None:
    dt = BeautifulSoup(
        '<dt class="ptitle"><a href="/content/CVPR2024/html/paper.html">Attention Is All You Need</a></dt>',
        "html.parser",
    ).dt
    assert parse_paper(dt, venue="CVPR", year=2024) == Paper.from_raw(
        title="Attention Is All You Need",
        authors=None,
        venue="CVPR",
        year=2024,
        pdf_url=None,
    )


###################################
#     Tests for scrape_papers     #
###################################


# --- Fixtures ---


@pytest.fixture
def paper_a() -> Paper:
    return Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith", "John Doe"],
        venue="CVPR",
        year=2024,
        pdf_url="https://openaccess.thecvf.com/content/CVPR2024/papers/paper_a.pdf",
    )


@pytest.fixture
def paper_b() -> Paper:
    return Paper.from_raw(
        title="BERT",
        authors=["Alice Brown"],
        venue="CVPR",
        year=2024,
        pdf_url="https://openaccess.thecvf.com/content/CVPR2024/papers/paper_b.pdf",
    )


# --- URL building ---


def test_scrape_papers_builds_correct_url() -> None:
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>") as mock_fetch,
        patch(f"{MODULE}.parse_paper_entries", return_value=[]),
    ):
        scrape_papers(venue="CVPR", year=2024)
        mock_fetch.assert_called_once_with("https://openaccess.thecvf.com/CVPR2024?day=all")


def test_scrape_papers_builds_correct_url_different_venue() -> None:
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>") as mock_fetch,
        patch(f"{MODULE}.parse_paper_entries", return_value=[]),
    ):
        scrape_papers(venue="ICCV", year=2023)
        mock_fetch.assert_called_once_with("https://openaccess.thecvf.com/ICCV2023?day=all")


# --- Passes venue and year to parse_paper ---


def test_scrape_papers_passes_venue_and_year_to_parse_paper(paper_a: Paper) -> None:
    mock_dt = Mock()
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[mock_dt]),
        patch(f"{MODULE}.parse_paper", return_value=paper_a) as mock_parse,
    ):
        scrape_papers(venue="CVPR", year=2024)
        mock_parse.assert_called_once_with(mock_dt, venue="CVPR", year=2024)


# --- Empty listing ---


def test_scrape_papers_returns_empty_dataframe_when_no_entries() -> None:
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[]),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert result.is_empty()


def test_scrape_papers_empty_dataframe_has_correct_schema() -> None:
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[]),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert result.schema == {
        PAPER_TITLE: pl.String,
        PAPER_AUTHORS: pl.List(pl.String),
        PAPER_VENUE: pl.String,
        PAPER_YEAR: pl.Int32,
        PAPER_URL: pl.String,
        PAPER_ID: pl.String,
    }


# --- Single paper ---


def test_scrape_papers_single_paper(paper_a: Paper) -> None:
    mock_dt = Mock()
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[mock_dt]),
        patch(f"{MODULE}.parse_paper", return_value=paper_a),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need"],
                PAPER_AUTHORS: [["Jane Smith", "John Doe"]],
                PAPER_VENUE: ["CVPR"],
                PAPER_YEAR: [2024],
                PAPER_URL: ["https://openaccess.thecvf.com/content/CVPR2024/papers/paper_a.pdf"],
                PAPER_ID: [paper_a.hash()],
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


# --- Multiple papers ---


def test_scrape_papers_multiple_papers(paper_a: Paper, paper_b: Paper) -> None:
    mock_dt_a = Mock()
    mock_dt_b = Mock()

    def side_effect(
        dt: Tag,
        venue: str,  # noqa: ARG001
        year: int,  # noqa: ARG001
    ) -> Paper:
        return paper_a if dt is mock_dt_a else paper_b

    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[mock_dt_a, mock_dt_b]),
        patch(f"{MODULE}.parse_paper", side_effect=side_effect),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                PAPER_TITLE: ["Attention Is All You Need", "BERT"],
                PAPER_AUTHORS: [["Jane Smith", "John Doe"], ["Alice Brown"]],
                PAPER_VENUE: ["CVPR", "CVPR"],
                PAPER_YEAR: [2024, 2024],
                PAPER_URL: [
                    "https://openaccess.thecvf.com/content/CVPR2024/papers/paper_a.pdf",
                    "https://openaccess.thecvf.com/content/CVPR2024/papers/paper_b.pdf",
                ],
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


# --- Papers with None fields ---


def test_scrape_papers_paper_with_null_pdf_url() -> None:
    paper = Paper.from_raw(
        title="My Paper", authors=["Jane Smith"], venue="CVPR", year=2024, pdf_url=None
    )
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[Mock()]),
        patch(f"{MODULE}.parse_paper", return_value=paper),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert result[PAPER_URL][0] is None


def test_scrape_papers_paper_with_null_authors() -> None:
    paper = Paper.from_raw(
        title="My Paper",
        authors=None,
        venue="CVPR",
        year=2024,
        pdf_url="https://example.com/paper.pdf",
    )
    with (
        patch(f"{MODULE}.fetch_html", return_value="<html></html>"),
        patch(f"{MODULE}.parse_paper_entries", return_value=[Mock()]),
        patch(f"{MODULE}.parse_paper", return_value=paper),
    ):
        result = scrape_papers(venue="CVPR", year=2024)
    assert result[PAPER_AUTHORS][0] is None


# --- Network error propagates ---


def test_scrape_papers_propagates_network_error() -> None:
    import requests

    with (
        patch(
            f"{MODULE}.fetch_html", side_effect=requests.exceptions.ConnectionError("unreachable")
        ),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        scrape_papers(venue="CVPR", year=2024)


###########################################
#     Tests for load_or_scrape_papers     #
###########################################


# --- Fixtures ---


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "papers"


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    paper = Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith", "John Doe"],
        venue="CVPR",
        year=2024,
        pdf_url="https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf",
    )
    return pl.DataFrame(
        {
            PAPER_TITLE: ["Attention Is All You Need"],
            PAPER_AUTHORS: [["Jane Smith", "John Doe"]],
            PAPER_VENUE: ["CVPR"],
            PAPER_YEAR: [2024],
            PAPER_URL: ["https://openaccess.thecvf.com/content/CVPR2024/papers/paper.pdf"],
            PAPER_ID: [paper.hash()],
        },
        schema={
            PAPER_TITLE: pl.String,
            PAPER_AUTHORS: pl.List(pl.String),
            PAPER_VENUE: pl.String,
            PAPER_YEAR: pl.Int32,
            PAPER_URL: pl.String,
            PAPER_ID: pl.String,
        },
    )


# --- Default cache_dir is None ---


def test_load_or_scrape_papers_default_cache_dir_is_none(
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame) as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024)
        mock_scrape.assert_called_once_with(venue="CVPR", year=2024)


# --- cache_dir=None: no caching ---


def test_load_or_scrape_papers_scrapes_when_cache_dir_is_none(
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame) as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
        mock_scrape.assert_called_once_with(venue="CVPR", year=2024)


def test_load_or_scrape_papers_returns_frame_when_cache_dir_is_none(
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        result = load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
    assert_frame_equal(result, sample_frame)


def test_load_or_scrape_papers_does_not_write_file_when_cache_dir_is_none(
    tmp_path: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
    assert not any(tmp_path.glob("*.parquet"))


def test_load_or_scrape_papers_scrapes_every_call_when_cache_dir_is_none(
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame) as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
        assert mock_scrape.call_count == 2


# --- cache_dir provided: directory creation ---


def test_load_or_scrape_papers_creates_cache_dir(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    assert not cache_dir.exists()
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
    assert cache_dir.exists()


# --- cache_dir provided: cache miss ---


def test_load_or_scrape_papers_scrapes_when_no_cache(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame) as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
        mock_scrape.assert_called_once_with(venue="CVPR", year=2024)


def test_load_or_scrape_papers_saves_parquet_when_no_cache(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
    assert (cache_dir / "CVPR_2024.parquet").is_file()


def test_load_or_scrape_papers_returns_scraped_frame_when_no_cache(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        result = load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
    assert_frame_equal(result, sample_frame)


# --- cache_dir provided: cache hit ---


def test_load_or_scrape_papers_does_not_scrape_when_cache_exists(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    cache_dir.mkdir(parents=True)
    sample_frame.write_parquet(cache_dir / "CVPR_2024.parquet")
    with patch(f"{MODULE}.scrape_papers") as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
        mock_scrape.assert_not_called()


def test_load_or_scrape_papers_returns_cached_frame(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    cache_dir.mkdir(parents=True)
    sample_frame.write_parquet(cache_dir / "CVPR_2024.parquet")
    result = load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
    assert_frame_equal(result, sample_frame)


def test_load_or_scrape_papers_scrapes_only_once_across_multiple_calls(
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame) as mock_scrape:
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)
        mock_scrape.assert_called_once()


# --- Cache filename ---


@pytest.mark.parametrize(
    ("venue", "year", "expected_filename"),
    [
        pytest.param("CVPR", 2024, "CVPR_2024.parquet", id="cvpr_2024"),
        pytest.param("ICCV", 2023, "ICCV_2023.parquet", id="iccv_2023"),
        pytest.param("WACV", 2022, "WACV_2022.parquet", id="wacv_2022"),
    ],
)
def test_load_or_scrape_papers_cache_filename(
    venue: str,
    year: int,
    expected_filename: str,
    cache_dir: Path,
    sample_frame: pl.DataFrame,
) -> None:
    with patch(f"{MODULE}.scrape_papers", return_value=sample_frame):
        load_or_scrape_papers(venue=venue, year=year, cache_dir=cache_dir)
    assert (cache_dir / expected_filename).is_file()


# --- Network error propagates ---


def test_load_or_scrape_papers_propagates_network_error_when_no_cache(
    cache_dir: Path,
) -> None:
    import requests

    with (
        patch(
            f"{MODULE}.scrape_papers",
            side_effect=requests.exceptions.ConnectionError("unreachable"),
        ),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=cache_dir)


def test_load_or_scrape_papers_propagates_network_error_when_cache_dir_is_none() -> None:
    import requests

    with (
        patch(
            f"{MODULE}.scrape_papers",
            side_effect=requests.exceptions.ConnectionError("unreachable"),
        ),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        load_or_scrape_papers(venue="CVPR", year=2024, cache_dir=None)
