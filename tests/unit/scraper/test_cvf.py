from __future__ import annotations


from candidex.scraper.cvf import parse_paper_entries

#########################################
#     Tests for parse_paper_entries     #
#########################################

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
