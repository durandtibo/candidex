from __future__ import annotations

import pytest
from bs4 import BeautifulSoup, Tag

from candidex.paper import Paper
from candidex.scraper.cvf import (
    BASE_URL,
    build_listing_url,
    parse_paper,
    parse_paper_entries,
    resolve_url,
)

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
