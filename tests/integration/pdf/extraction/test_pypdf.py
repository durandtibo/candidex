from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from candidex.pdf.extraction import PyPdfExtractor, extract_text_pypdf
from candidex.testing.fixtures import pypdf_available, pypdf_not_available
from tests.integration.pdf.extraction.helpers import make_pdf

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def single_page_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "single.pdf", ["Hello from page one."])


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    return make_pdf(
        tmp_path / "multi.pdf", ["Page one text.", "Page two text.", "Page three text."]
    )


####################################
#     Tests for PyPdfExtractor     #
####################################


@pypdf_available
def test_pypdf_extractor_integration_single_page(single_page_pdf: Path) -> None:
    extractor = PyPdfExtractor()
    assert extractor.extract(single_page_pdf) == "Hello from page one."


@pypdf_available
def test_pypdf_extractor_integration_multiple_pages(multi_page_pdf: Path) -> None:
    extractor = PyPdfExtractor()
    assert extractor.extract(multi_page_pdf) == "Page one text.\fPage two text.\fPage three text."


@pypdf_available
def test_pypdf_extractor_integration_max_pages_one(multi_page_pdf: Path) -> None:
    extractor = PyPdfExtractor(max_pages=1)
    assert extractor.extract(multi_page_pdf) == "Page one text."


@pypdf_available
def test_pypdf_extractor_integration_max_pages_two(multi_page_pdf: Path) -> None:
    extractor = PyPdfExtractor(max_pages=2)
    assert extractor.extract(multi_page_pdf) == "Page one text.\fPage two text."


@pypdf_not_available
def test_pypdf_extractor_without_pypdf() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."):
        PyPdfExtractor()


########################################
#     Tests for extract_text_pypdf     #
########################################


@pypdf_available
def test_extract_text_pypdf_single_page_exact_output(single_page_pdf: Path) -> None:
    assert extract_text_pypdf(single_page_pdf) == "Hello from page one."


@pypdf_available
def test_extract_text_pypdf_three_pages_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdf(multi_page_pdf) == "Page one text.\fPage two text.\fPage three text."


@pypdf_available
def test_extract_text_pypdf_max_pages_one_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdf(multi_page_pdf, max_pages=1) == "Page one text."


@pypdf_available
def test_extract_text_pypdf_max_pages_two_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdf(multi_page_pdf, max_pages=2) == "Page one text.\fPage two text."


@pypdf_available
def test_extract_text_pypdf_max_pages_zero_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdf(multi_page_pdf, max_pages=0) == ""


@pypdf_not_available
def test_extract_text_pypdf_without_pypdf(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."):
        extract_text_pypdf(tmp_path / "paper.pdf")
