from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from candidex.pdf.extraction import PyPdfium2Extractor, extract_text_pypdfium2
from candidex.testing.fixtures import pypdfium2_available, pypdfium2_not_available
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


########################################
#     Tests for PyPdfium2Extractor     #
########################################


@pypdfium2_available
def test_pypdfium2_extractor_integration_single_page(single_page_pdf: Path) -> None:
    extractor = PyPdfium2Extractor()
    assert extractor.extract(single_page_pdf) == "Hello from page one."


@pypdfium2_available
def test_pypdfium2_extractor_integration_multiple_pages(multi_page_pdf: Path) -> None:
    extractor = PyPdfium2Extractor()
    assert extractor.extract(multi_page_pdf) == "Page one text.\fPage two text.\fPage three text."


@pypdfium2_available
def test_pypdfium2_extractor_integration_max_pages_one(multi_page_pdf: Path) -> None:
    extractor = PyPdfium2Extractor(max_pages=1)
    assert extractor.extract(multi_page_pdf) == "Page one text."


@pypdfium2_available
def test_pypdfium2_extractor_integration_max_pages_two(multi_page_pdf: Path) -> None:
    extractor = PyPdfium2Extractor(max_pages=2)
    assert extractor.extract(multi_page_pdf) == "Page one text.\fPage two text."


@pypdfium2_not_available
def test_pypdfium2_extractor_without_pypdfium2() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdfium2' package is required but not installed."):
        PyPdfium2Extractor()


########################################
#     Tests for extract_text_pypdfium2     #
########################################


@pypdfium2_available
def test_extract_text_pypdfium2_single_page_exact_output(single_page_pdf: Path) -> None:
    assert extract_text_pypdfium2(single_page_pdf) == "Hello from page one."


@pypdfium2_available
def test_extract_text_pypdfium2_three_pages_exact_output(multi_page_pdf: Path) -> None:
    assert (
        extract_text_pypdfium2(multi_page_pdf) == "Page one text.\fPage two text.\fPage three text."
    )


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_one_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdfium2(multi_page_pdf, max_pages=1) == "Page one text."


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_two_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdfium2(multi_page_pdf, max_pages=2) == "Page one text.\fPage two text."


@pypdfium2_available
def test_extract_text_pypdfium2_max_pages_zero_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pypdfium2(multi_page_pdf, max_pages=0) == ""


@pypdfium2_not_available
def test_extract_text_pypdfium2_without_pypdfium2(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"'pypdfium2' package is required but not installed."):
        extract_text_pypdfium2(tmp_path / "paper.pdf")
