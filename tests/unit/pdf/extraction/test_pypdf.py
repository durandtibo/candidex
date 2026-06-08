from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from candidex.pdf.extraction import PyPdfExtractor, extract_text_pypdf
from candidex.testing.fixtures import pypdf_available

MODULE = "candidex.pdf.extraction.pypdf"


# --- Helpers ---


def make_page(text: str | None) -> Mock:
    page = Mock()
    page.extract_text.return_value = text
    return page


def make_reader(pages: list[Mock]) -> Mock:
    return Mock(pages=pages)


####################################
#     Tests for PyPdfExtractor     #
####################################


# --- Construction ---


@pypdf_available
def test_pypdf_extractor_calls_check_pypdf_on_init() -> None:
    with patch(f"{MODULE}.check_pypdf") as mock_check:
        PyPdfExtractor()
        mock_check.assert_called_once()


@pypdf_available
def test_pypdf_extractor_raises_when_pypdf_not_available() -> None:
    with (
        patch(f"{MODULE}.check_pypdf", side_effect=RuntimeError("pypdf not installed")),
        pytest.raises(RuntimeError, match="pypdf not installed"),
    ):
        PyPdfExtractor()


@pypdf_available
def test_pypdf_extractor_default_max_pages_is_none() -> None:
    assert PyPdfExtractor()._max_pages is None


@pypdf_available
def test_pypdf_extractor_stores_max_pages() -> None:
    assert PyPdfExtractor(max_pages=2)._max_pages == 2


# --- __repr__ ---


@pypdf_available
def test_pypdf_extractor_repr_no_max_pages() -> None:
    assert repr(PyPdfExtractor()) == "PyPdfExtractor(max_pages=None)"


@pypdf_available
def test_pypdf_extractor_repr_with_max_pages() -> None:
    assert repr(PyPdfExtractor(max_pages=2)) == "PyPdfExtractor(max_pages=2)"


# --- extract ---


@pypdf_available
def test_pypdf_extractor_calls_extract_text_pypdf() -> None:
    with patch(f"{MODULE}.extract_text_pypdf", return_value="extracted text") as mock_extract:
        PyPdfExtractor().extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=None)


@pypdf_available
def test_pypdf_extractor_passes_max_pages_to_extract_text_pypdf() -> None:
    with patch(f"{MODULE}.extract_text_pypdf", return_value="extracted text") as mock_extract:
        PyPdfExtractor(max_pages=2).extract(Path("paper.pdf"))
        mock_extract.assert_called_once_with(pdf_path=Path("paper.pdf"), max_pages=2)


@pypdf_available
def test_pypdf_extractor_returns_extracted_text() -> None:
    with patch(f"{MODULE}.extract_text_pypdf", return_value="Hello world"):
        assert PyPdfExtractor().extract(Path("paper.pdf")) == "Hello world"


@pypdf_available
def test_pypdf_extractor_returns_empty_string_when_no_text() -> None:
    with patch(f"{MODULE}.extract_text_pypdf", return_value=""):
        assert PyPdfExtractor().extract(Path("paper.pdf")) == ""


@pypdf_available
def test_pypdf_extractor_passes_pdf_path_correctly() -> None:
    pdf_path = Path("papers/attention.pdf")
    with patch(f"{MODULE}.extract_text_pypdf", return_value="text") as mock_extract:
        PyPdfExtractor().extract(pdf_path)
        _, kwargs = mock_extract.call_args
        assert kwargs["pdf_path"] == pdf_path


@pypdf_available
def test_pypdf_extractor_integration_file_not_found(tmp_path: Path) -> None:
    extractor = PyPdfExtractor()
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        extractor.extract(tmp_path / "nonexistent.pdf")


########################################
#     Tests for extract_text_pypdf     #
########################################

# --- Empty PDF ---


@pypdf_available
def test_extract_text_pypdf_empty_pdf_returns_empty_string() -> None:
    with patch(f"{MODULE}.PdfReader", return_value=make_reader([])):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


@pypdf_available
def test_extract_text_pypdf_all_pages_empty_returns_empty_string() -> None:
    reader = make_reader([make_page(None), make_page(None), make_page(None)])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


# --- Single page ---


@pypdf_available
def test_extract_text_pypdf_single_page_returns_text() -> None:
    reader = make_reader([make_page("Hello world.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Hello world."


@pypdf_available
def test_extract_text_pypdf_single_page_none_returns_empty_string() -> None:
    reader = make_reader([make_page(None)])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == ""


# --- Multiple pages ---


@pypdf_available
def test_extract_text_pypdf_multiple_pages_separated_by_form_feed() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Page one.\fPage two."


@pypdf_available
def test_extract_text_pypdf_skips_pages_with_no_text() -> None:
    reader = make_reader([make_page("Page one."), make_page(None), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf")) == "Page one.\fPage three."


@pypdf_available
def test_extract_text_pypdf_all_pages_extracted_by_default() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    assert result.count("\f") == 2


# --- max_pages ---


@pypdf_available
def test_extract_text_pypdf_max_pages_one() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=1) == "Page one."


@pypdf_available
def test_extract_text_pypdf_max_pages_two() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=2) == "Page one.\fPage two."


@pypdf_available
def test_extract_text_pypdf_max_pages_none_extracts_all() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result_all = extract_text_pypdf(Path("paper.pdf"), max_pages=None)
        result_default = extract_text_pypdf(Path("paper.pdf"))
    assert result_all == result_default


@pypdf_available
def test_extract_text_pypdf_max_pages_exceeds_total_pages() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=10) == "Page one.\fPage two."


@pypdf_available
def test_extract_text_pypdf_max_pages_zero_returns_empty_string() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        assert extract_text_pypdf(Path("paper.pdf"), max_pages=0) == ""


# --- PdfReader called with correct path ---


@pypdf_available
def test_extract_text_pypdf_passes_path_to_pdf_reader() -> None:
    with patch(f"{MODULE}.PdfReader", return_value=make_reader([])) as mock_reader:
        extract_text_pypdf(Path("paper.pdf"))
        mock_reader.assert_called_once_with(Path("paper.pdf"))


# --- Page separator ---


@pypdf_available
def test_extract_text_pypdf_uses_form_feed_as_separator() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    assert "\f" in result


@pypdf_available
def test_extract_text_pypdf_result_splittable_by_form_feed() -> None:
    reader = make_reader([make_page("Page one."), make_page("Page two."), make_page("Page three.")])
    with patch(f"{MODULE}.PdfReader", return_value=reader):
        result = extract_text_pypdf(Path("paper.pdf"))
    pages = result.split("\f")
    assert pages == ["Page one.", "Page two.", "Page three."]


@pypdf_available
def test_extract_text_pypdf_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        extract_text_pypdf(tmp_path / "nonexistent.pdf")
