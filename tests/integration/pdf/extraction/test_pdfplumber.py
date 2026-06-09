from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from candidex.pdf.extraction import extract_text_pdfplumber
from candidex.testing.fixtures import pdfplumber_available, pdfplumber_not_available
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


#############################################
#     Tests for extract_text_pdfplumber     #
#############################################


@pdfplumber_available
def test_extract_text_pdfplumber_single_page_exact_output(single_page_pdf: Path) -> None:
    assert extract_text_pdfplumber(single_page_pdf) == "Hello from page one."


@pdfplumber_available
def test_extract_text_pdfplumber_three_pages_exact_output(multi_page_pdf: Path) -> None:
    assert (
        extract_text_pdfplumber(multi_page_pdf)
        == "Page one text.\fPage two text.\fPage three text."
    )


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_one_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pdfplumber(multi_page_pdf, max_pages=1) == "Page one text."


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_two_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pdfplumber(multi_page_pdf, max_pages=2) == "Page one text.\fPage two text."


@pdfplumber_available
def test_extract_text_pdfplumber_max_pages_zero_exact_output(multi_page_pdf: Path) -> None:
    assert extract_text_pdfplumber(multi_page_pdf, max_pages=0) == ""


@pdfplumber_not_available
def test_extract_text_pdfplumber_without_pdfplumber(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"'pdfplumber' package is required but not installed."):
        extract_text_pdfplumber(tmp_path / "paper.pdf")
