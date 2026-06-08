from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from candidex.pdf.extraction import extract_text_pypdf
from candidex.testing.fixtures import pypdf_available, pypdf_not_available
from tests.integration.pdf.extraction.helpers import make_pdf

if TYPE_CHECKING:
    from pathlib import Path


########################################
#     Tests for extract_text_pypdf     #
########################################


@pypdf_available
def test_extract_text_pypdf_single_page_exact_output(tmp_path: Path) -> None:
    pdf_path = make_pdf(tmp_path / "paper.pdf", ["Hello world"])
    assert extract_text_pypdf(pdf_path) == "Hello world"


@pypdf_available
def test_extract_text_pypdf_three_pages_exact_output(tmp_path: Path) -> None:
    pdf_path = make_pdf(
        tmp_path / "paper.pdf", ["Page one text.", "Page two text.", "Page three text."]
    )
    assert extract_text_pypdf(pdf_path) == "Page one text.\fPage two text.\fPage three text."


@pypdf_available
def test_extract_text_pypdf_max_pages_one_exact_output(tmp_path: Path) -> None:
    pdf_path = make_pdf(
        tmp_path / "paper.pdf", ["Page one text.", "Page two text.", "Page three text."]
    )
    assert extract_text_pypdf(pdf_path, max_pages=1) == "Page one text."


@pypdf_available
def test_extract_text_pypdf_max_pages_two_exact_output(tmp_path: Path) -> None:
    pdf_path = make_pdf(
        tmp_path / "paper.pdf", ["Page one text.", "Page two text.", "Page three text."]
    )
    assert extract_text_pypdf(pdf_path, max_pages=2) == "Page one text.\fPage two text."


@pypdf_available
def test_extract_text_pypdf_max_pages_zero_exact_output(tmp_path: Path) -> None:
    pdf_path = make_pdf(tmp_path / "paper.pdf", ["Page one text.", "Page two text."])
    assert extract_text_pypdf(pdf_path, max_pages=0) == ""


@pypdf_not_available
def test_extract_text_pypdf_without_pypdf(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."):
        extract_text_pypdf(tmp_path / "paper.pdf")
