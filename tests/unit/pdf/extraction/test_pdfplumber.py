from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from candidex.pdf.extraction import PdfPlumberExtractor, extract_text_pdfplumber
from candidex.testing.fixtures import pdfplumber_available

MODULE = "candidex.pdf.extraction.pdfplumber"


# --- Helpers ---


def make_page(text: str | None, page_number: int = 1) -> Mock:
    page = Mock()
    page.page_number = page_number
    page.extract_text.return_value = text
    return page


def make_pdf(pages: list[Mock]) -> MagicMock:
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = Mock(return_value=pdf)
    pdf.__exit__ = Mock(return_value=False)
    return pdf


#########################################
#     Tests for PdfPlumberExtractor     #
#########################################


# --- Construction ---


@pdfplumber_available
def test_pdfplumber_extractor_calls_check_pdfplumber_on_init() -> None:
    with patch(f"{MODULE}.check_pdfplumber") as mock_check:
        PdfPlumberExtractor()
        mock_check.assert_called_once()


@pdfplumber_available
def test_pdfplumber_extractor_raises_when_pdfplumber_not_available() -> None:
    with (
        patch(f"{MODULE}.check_pdfplumber", side_effect=RuntimeError("pdfplumber not installed")),
        pytest.raises(RuntimeError, match="pdfplumber not installed"),
    ):
        PdfPlumberExtractor()


@pdfplumber_available
def test_pdfplumber_extractor_default_max_pages_is_none() -> None:
    assert PdfPlumberExtractor()._max_pages is None


@pdfplumber_available
def test_pdfplumber_extractor_stores_max_pages() -> None:
    assert PdfPlumberExtractor(max_pages=2)._max_pages == 2


# --- __repr__ ---


@pdfplumber_available
def test_pdfplumber_extractor_repr_no_max_pages() -> None:
    assert repr(PdfPlumberExtractor()) == "PdfPlumberExtractor(max_pages=None)"


@pdfplumber_available
def test_pdfplumber_extractor_repr_with_max_pages() -> None:
    assert repr(PdfPlumberExtractor(max_pages=2)) == "PdfPlumberExtractor(max_pages=2)"


# --- extract ---


@pdfplumber_available
def test_pdfplumber_extractor_calls_extract_text_pdfplumber() -> None:
    with patch(f"{MODULE}.extract_text_pdfplumber", return_value="extracted text") as mock_extract:
        PdfPlumberExtractor().extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=None)


@pdfplumber_available
def test_pdfplumber_extractor_passes_max_pages_to_extract_text_pdfplumber() -> None:
    with patch(f"{MODULE}.extract_text_pdfplumber", return_value="extracted text") as mock_extract:
        PdfPlumberExtractor(max_pages=2).extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=2)


@pdfplumber_available
def test_pdfplumber_extractor_returns_extracted_text() -> None:
    with patch(f"{MODULE}.extract_text_pdfplumber", return_value="Hello world"):
        assert PdfPlumberExtractor().extract(Path("paper.pdf")) == "Hello world"


@pdfplumber_available
def test_pdfplumber_extractor_returns_empty_string_when_no_text() -> None:
    with patch(f"{MODULE}.extract_text_pdfplumber", return_value=""):
        assert PdfPlumberExtractor().extract(Path("paper.pdf")) == ""


@pdfplumber_available
def test_pdfplumber_extractor_passes_pdf_path_correctly() -> None:
    pdf_path = Path("papers/attention.pdf")
    with patch(f"{MODULE}.extract_text_pdfplumber", return_value="text") as mock_extract:
        PdfPlumberExtractor().extract(pdf_path)
        _, kwargs = mock_extract.call_args
        assert kwargs["pdf_path"] == pdf_path


@pdfplumber_available
def test_pdfplumber_extractor_integration_file_not_found(tmp_path: Path) -> None:
    extractor = PdfPlumberExtractor()
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        extractor.extract(tmp_path / "nonexistent.pdf")


#############################################
#     Tests for extract_text_pdfplumber     #
#############################################

# --- Empty PDF ---


@pdfplumber_available
def test_extract_text_pdfplumber_empty_pdf_returns_empty_string() -> None:
    with patch(f"{MODULE}.pdfplumber.open", return_value=make_pdf([])):
        assert extract_text_pdfplumber(Path("paper.pdf")) == ""


@pdfplumber_available
def test_extract_text_pdfplumber_all_pages_empty_returns_empty_string() -> None:
    pdf = make_pdf([make_page(None, 1), make_page(None, 2), make_page(None, 3)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf")) == ""


# --- Single page ---


@pdfplumber_available
def test_extract_text_pdfplumber_single_page_returns_text() -> None:
    pdf = make_pdf([make_page("Hello world.", 1)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf")) == "Hello world."


@pdfplumber_available
def test_extract_text_pdfplumber_single_page_none_returns_empty_string() -> None:
    pdf = make_pdf([make_page(None, 1)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf")) == ""


# --- Multiple pages ---


@pdfplumber_available
def test_extract_text_pdfplumber_multiple_pages_separated_by_form_feed() -> None:
    pdf = make_pdf([make_page("Page one.", 1), make_page("Page two.", 2)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf")) == "Page one.\fPage two."


@pdfplumber_available
def test_extract_text_pdfplumber_skips_pages_with_no_text() -> None:
    pdf = make_pdf([make_page("Page one.", 1), make_page(None, 2), make_page("Page three.", 3)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf")) == "Page one.\fPage three."


@pdfplumber_available
def test_extract_text_pdfplumber_all_pages_extracted_by_default() -> None:
    pdf = make_pdf(
        [make_page("Page one.", 1), make_page("Page two.", 2), make_page("Page three.", 3)]
    )
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        result = extract_text_pdfplumber(Path("paper.pdf"))
    assert result.count("\f") == 2


# --- max_pages ---


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_one() -> None:
    pdf = make_pdf(
        [make_page("Page one.", 1), make_page("Page two.", 2), make_page("Page three.", 3)]
    )
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf"), max_pages=1) == "Page one."


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_two() -> None:
    pdf = make_pdf(
        [make_page("Page one.", 1), make_page("Page two.", 2), make_page("Page three.", 3)]
    )
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf"), max_pages=2) == "Page one.\fPage two."


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_none_extracts_all() -> None:
    pdf = make_pdf(
        [make_page("Page one.", 1), make_page("Page two.", 2), make_page("Page three.", 3)]
    )
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        result_all = extract_text_pdfplumber(Path("paper.pdf"), max_pages=None)
        result_default = extract_text_pdfplumber(Path("paper.pdf"))
    assert result_all == result_default


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_exceeds_total_pages() -> None:
    pdf = make_pdf([make_page("Page one.", 1), make_page("Page two.", 2)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf"), max_pages=10) == "Page one.\fPage two."


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_zero_returns_empty_string() -> None:
    pdf = make_pdf([make_page("Page one.", 1), make_page("Page two.", 2)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        assert extract_text_pdfplumber(Path("paper.pdf"), max_pages=0) == ""


# --- Context manager ---


@pdfplumber_available
def test_extract_text_pdfplumber_opens_file_with_correct_path() -> None:
    pdf = make_pdf([])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf) as mock_open:
        extract_text_pdfplumber(Path("paper.pdf"))
        mock_open.assert_called_once_with(Path("paper.pdf"))


@pdfplumber_available
def test_extract_text_pdfplumber_uses_context_manager() -> None:
    pdf = make_pdf([])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        extract_text_pdfplumber(Path("paper.pdf"))
    pdf.__exit__.assert_called_once()


# --- Page separator ---


@pdfplumber_available
def test_extract_text_pdfplumber_uses_form_feed_as_separator() -> None:
    pdf = make_pdf([make_page("Page one.", 1), make_page("Page two.", 2)])
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        result = extract_text_pdfplumber(Path("paper.pdf"))
    assert "\f" in result


@pdfplumber_available
def test_extract_text_pdfplumber_result_splittable_by_form_feed() -> None:
    pdf = make_pdf(
        [make_page("Page one.", 1), make_page("Page two.", 2), make_page("Page three.", 3)]
    )
    with patch(f"{MODULE}.pdfplumber.open", return_value=pdf):
        result = extract_text_pdfplumber(Path("paper.pdf"))
    assert result.split("\f") == ["Page one.", "Page two.", "Page three."]


@pdfplumber_available
def test_extract_text_pdfplumber_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        extract_text_pdfplumber(tmp_path / "nonexistent.pdf")
