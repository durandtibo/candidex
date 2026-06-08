from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from candidex.pdf.extraction import extract_text_pypdf

MODULE = "candidex.pdf.extraction._pypdf"


# --- Helpers ---


def make_page(text: str | None) -> Mock:
    page = Mock()
    page.extract_text.return_value = text
    return page


def make_reader(pages: list[Mock]) -> Mock:
    return Mock(pages=pages)


########################################
#     Tests for extract_text_pypdf     #
########################################

# --- Empty PDF ---


def test_extract_text_pypdf_empty_pdf_returns_empty_string() -> None:
    with patch(f"{MODULE}.PdfReader", return_value=make_reader([])):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


def test_extract_text_pypdf_all_pages_empty_returns_empty_string() -> None:
    reader = make_reader([make_page(None), make_page(None), make_page(None)])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


# --- Single page ---


def test_extract_text_pypdf_single_page_returns_text() -> None:
    reader = make_reader([make_page("Hello world.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Hello world."


def test_extract_text_pypdf_single_page_none_returns_empty_string() -> None:
    reader = make_reader([make_page(None)])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


# --- Multiple pages ---


def test_extract_text_pypdf_multiple_pages_separated_by_form_feed() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Page one.\fPage two."


def test_extract_text_pypdf_skips_pages_with_no_text() -> None:
    reader = make_reader([make_page("Page one."), make_page(None), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Page one.\fPage three."


def test_extract_text_pypdf_all_pages_extracted_by_default() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    assert result.count("\f") == 2


# --- max_pages ---


def test_extract_text_pypdf_max_pages_one() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=1) == "Page one."


def test_extract_text_pypdf_max_pages_two() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=2) == "Page one.\fPage two."


def test_extract_text_pypdf_max_pages_none_extracts_all() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result_all = extract_text_pypdf(Path("paper.pdf"), max_pages=None)
        result_default = extract_text_pypdf(Path("paper.pdf"))
    assert result_all == result_default


def test_extract_text_pypdf_max_pages_exceeds_total_pages() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=10) == "Page one.\fPage two."


def test_extract_text_pypdf_max_pages_zero_returns_empty_string() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=0) == ""


# --- PdfReader called with correct path ---


def test_extract_text_pypdf_passes_path_to_pdf_reader() -> None:
    with patch(f"{MODULE}.PdfReader", return_value=make_reader([])) as mock_reader:
        extract_text_pypdf(Path("paper.pdf"))
        mock_reader.assert_called_once_with(Path("paper.pdf"))


# --- Page separator ---


def test_extract_text_pypdf_uses_form_feed_as_separator() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    assert "\f" in result


def test_extract_text_pypdf_result_splittable_by_form_feed() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    pages = result.split("\f")
    assert pages == ["Page one.", "Page two.", "Page three."]
