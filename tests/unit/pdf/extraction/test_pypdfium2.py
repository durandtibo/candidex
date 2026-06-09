from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from candidex.pdf.extraction import PyPdfium2Extractor, extract_text_pypdfium2
from candidex.testing.fixtures import pypdfium2_available

MODULE = "candidex.pdf.extraction.pypdfium2"


# --- Helpers ---


def make_text_page(text: str | None) -> Mock:
    text_page = Mock()
    text_page.get_text_range.return_value = text or ""
    return text_page


def make_page(text: str | None) -> Mock:
    page = Mock()
    page.get_textpage.return_value = make_text_page(text)
    return page


def make_pdf(pages: list[Mock]) -> Mock:
    pdf = Mock()
    pdf.__len__ = Mock(return_value=len(pages))
    pdf.get_page.side_effect = lambda idx: pages[idx]
    return pdf


########################################
#     Tests for PyPdfium2Extractor     #
########################################


# --- Construction ---


@pypdfium2_available
def test_pypdfium2_extractor_calls_check_pypdfium2_on_init() -> None:
    with patch(f"{MODULE}.check_pypdfium2") as mock_check:
        PyPdfium2Extractor()
        mock_check.assert_called_once()


@pypdfium2_available
def test_pypdfium2_extractor_raises_when_pypdfium2_not_available() -> None:
    with (
        patch(f"{MODULE}.check_pypdfium2", side_effect=RuntimeError("pypdfium2 not installed")),
        pytest.raises(RuntimeError, match="pypdfium2 not installed"),
    ):
        PyPdfium2Extractor()


@pypdfium2_available
def test_pypdfium2_extractor_default_max_pages_is_none() -> None:
    assert PyPdfium2Extractor()._max_pages is None


@pypdfium2_available
def test_pypdfium2_extractor_stores_max_pages() -> None:
    assert PyPdfium2Extractor(max_pages=2)._max_pages == 2


# --- __repr__ ---


@pypdfium2_available
def test_pypdfium2_extractor_repr_no_max_pages() -> None:
    assert repr(PyPdfium2Extractor()) == "PyPdfium2Extractor(max_pages=None)"


@pypdfium2_available
def test_pypdfium2_extractor_repr_with_max_pages() -> None:
    assert repr(PyPdfium2Extractor(max_pages=2)) == "PyPdfium2Extractor(max_pages=2)"


# --- extract ---


@pypdfium2_available
def test_pypdfium2_extractor_calls_extract_text_pypdfium2() -> None:
    with patch(f"{MODULE}.extract_text_pypdfium2", return_value="extracted text") as mock_extract:
        PyPdfium2Extractor().extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=None)


@pypdfium2_available
def test_pypdfium2_extractor_passes_max_pages_to_extract_text_pypdfium2() -> None:
    with patch(f"{MODULE}.extract_text_pypdfium2", return_value="extracted text") as mock_extract:
        PyPdfium2Extractor(max_pages=2).extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=2)


@pypdfium2_available
def test_pypdfium2_extractor_returns_extracted_text() -> None:
    with patch(f"{MODULE}.extract_text_pypdfium2", return_value="Hello world"):
        assert PyPdfium2Extractor().extract(Path("paper.pdf")) == "Hello world"


@pypdfium2_available
def test_pypdfium2_extractor_returns_empty_string_when_no_text() -> None:
    with patch(f"{MODULE}.extract_text_pypdfium2", return_value=""):
        assert PyPdfium2Extractor().extract(Path("paper.pdf")) == ""


@pypdfium2_available
def test_pypdfium2_extractor_passes_pdf_path_correctly() -> None:
    pdf_path = Path("papers/attention.pdf")
    with patch(f"{MODULE}.extract_text_pypdfium2", return_value="text") as mock_extract:
        PyPdfium2Extractor().extract(pdf_path)
        _, kwargs = mock_extract.call_args
        assert kwargs["pdf_path"] == pdf_path


@pypdfium2_available
def test_pypdfium2_extractor_integration_file_not_found(tmp_path: Path) -> None:
    extractor = PyPdfium2Extractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(tmp_path / "nonexistent.pdf")


############################################
#     Tests for extract_text_pypdfium2     #
############################################

# --- Empty PDF ---


@pypdfium2_available
def test_extract_text_pypdfium2_empty_pdf_returns_empty_string() -> None:
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=make_pdf([])):
        assert extract_text_pypdfium2(Path("paper.pdf")) == ""


@pypdfium2_available
def test_extract_text_pypdfium2_all_pages_empty_returns_empty_string() -> None:
    pdf = make_pdf([make_page(None), make_page(None), make_page(None)])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == ""


# --- Single page ---


@pypdfium2_available
def test_extract_text_pypdfium2_single_page_returns_text() -> None:
    pdf = make_pdf([make_page("Hello world.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == "Hello world."


@pypdfium2_available
def test_extract_text_pypdfium2_single_page_none_returns_empty_string() -> None:
    pdf = make_pdf([make_page(None)])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == ""


@pypdfium2_available
def test_extract_text_pypdfium2_strips_whitespace() -> None:
    pdf = make_pdf([make_page("  Hello world.  ")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == "Hello world."


# --- Multiple pages ---


@pypdfium2_available
def test_extract_text_pypdfium2_multiple_pages_separated_by_form_feed() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == "Page one.\fPage two."


@pypdfium2_available
def test_extract_text_pypdfium2_skips_pages_with_no_text() -> None:
    pdf = make_pdf([make_page("Page one."), make_page(None), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf")) == "Page one.\fPage three."


@pypdfium2_available
def test_extract_text_pypdfium2_all_pages_extracted_by_default() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        result = extract_text_pypdfium2(Path("paper.pdf"))
    assert result.count("\f") == 2


# --- max_pages ---


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_one() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf"), max_pages=1) == "Page one."


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_two() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf"), max_pages=2) == "Page one.\fPage two."


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_none_extracts_all() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        result_all = extract_text_pypdfium2(Path("paper.pdf"), max_pages=None)
        result_default = extract_text_pypdfium2(Path("paper.pdf"))
    assert result_all == result_default


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_exceeds_total_pages() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf"), max_pages=10) == "Page one.\fPage two."


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_zero_returns_empty_string() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert extract_text_pypdfium2(Path("paper.pdf"), max_pages=0) == ""


# --- Resource management ---


@pypdfium2_available
def test_extract_text_pypdfium2_closes_pdf_after_extraction() -> None:
    pdf = make_pdf([make_page("Page one.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        extract_text_pypdfium2(Path("paper.pdf"))
    pdf.close.assert_called_once()


@pypdfium2_available
def test_extract_text_pypdfium2_closes_pdf_on_error() -> None:
    pdf = make_pdf([make_page("Page one.")])
    pdf.get_page.side_effect = RuntimeError("unexpected error")
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf), pytest.raises(RuntimeError):
        extract_text_pypdfium2(Path("paper.pdf"))
    pdf.close.assert_called_once()


@pypdfium2_available
def test_extract_text_pypdfium2_closes_page_after_extraction() -> None:
    page = make_page("Page one.")
    pdf = make_pdf([page])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        extract_text_pypdfium2(Path("paper.pdf"))
    page.close.assert_called_once()


@pypdfium2_available
def test_extract_text_pypdfium2_closes_text_page_after_extraction() -> None:
    page = make_page("Page one.")
    text_page = page.get_textpage.return_value
    pdf = make_pdf([page])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        extract_text_pypdfium2(Path("paper.pdf"))
    text_page.close.assert_called_once()


@pypdfium2_available
def test_extract_text_pypdfium2_closes_text_page_on_error() -> None:
    page = make_page("Page one.")
    text_page = page.get_textpage.return_value
    text_page.get_text_range.side_effect = RuntimeError("unexpected error")
    pdf = make_pdf([page])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf), pytest.raises(RuntimeError):
        extract_text_pypdfium2(Path("paper.pdf"))
    text_page.close.assert_called_once()


@pypdfium2_available
def test_extract_text_pypdfium2_closes_all_pages() -> None:
    pages = [make_page("Page one."), make_page("Page two."), make_page("Page three.")]
    pdf = make_pdf(pages)
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        extract_text_pypdfium2(Path("paper.pdf"))
    for page in pages:
        page.close.assert_called_once()


# --- Path passed to PdfDocument ---


@pypdfium2_available
def test_extract_text_pypdfium2_passes_path_to_pdf_document() -> None:
    pdf = make_pdf([])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf) as mock_doc:
        extract_text_pypdfium2(Path("paper.pdf"))
        mock_doc.assert_called_once_with(Path("paper.pdf"))


# --- Page separator ---


@pypdfium2_available
def test_extract_text_pypdfium2_uses_form_feed_as_separator() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        assert "\f" in extract_text_pypdfium2(Path("paper.pdf"))


@pypdfium2_available
def test_extract_text_pypdfium2_result_splittable_by_form_feed() -> None:
    pdf = make_pdf([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.pypdfium2.PdfDocument", return_value=pdf):
        result = extract_text_pypdfium2(Path("paper.pdf"))
    assert result.split("\f") == ["Page one.", "Page two.", "Page three."]


@pypdfium2_available
def test_extract_text_pypdfium2_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_text_pypdfium2(tmp_path / "nonexistent.pdf")
