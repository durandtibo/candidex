from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from candidex.pdf.extraction import extract_text_pypdf
from candidex.testing.fixtures import pypdf_available, pypdf_not_available

if TYPE_CHECKING:
    from pathlib import Path

# --- Helpers ---


def make_pdf(path: Path, texts: list[str]) -> Path:
    """Create a minimal valid PDF with one text string per page.

    Uses raw PDF construction with a Type1 font (Courier) to ensure text
    is extractable by pypdf without additional dependencies.
    """
    offsets = {}
    body = b""
    obj_num = 0
    page_ids = []
    stream_ids = []

    for text in texts:
        obj_num += 1
        stream_id = obj_num
        stream_ids.append(stream_id)
        stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        obj_bytes = (
            f"{stream_id} 0 obj\n<</Length {len(stream_content)}>>\nstream\n".encode()
            + stream_content
            + b"\nendstream\nendobj\n"
        )
        offsets[stream_id] = len(body) + len(b"%PDF-1.4\n")
        body += obj_bytes

        obj_num += 1
        page_id = obj_num
        page_ids.append(page_id)
        page_bytes = (
            f"{page_id} 0 obj\n"
            f"<</Type /Page /Parent {obj_num + 2} 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {stream_id} 0 R "
            f"/Resources <</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Courier>>>>>>"
            f">>\nendobj\n"
        ).encode()
        offsets[page_id] = len(body) + len(b"%PDF-1.4\n")
        body += page_bytes

    obj_num += 1
    pages_id = obj_num
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    pages_bytes = (
        f"{pages_id} 0 obj\n<</Type /Pages /Kids [{kids}] /Count {len(texts)}>>\nendobj\n"
    ).encode()
    offsets[pages_id] = len(body) + len(b"%PDF-1.4\n")
    body += pages_bytes

    obj_num += 1
    catalog_id = obj_num
    catalog_bytes = (
        f"{catalog_id} 0 obj\n<</Type /Catalog /Pages {pages_id} 0 R>>\nendobj\n"
    ).encode()
    offsets[catalog_id] = len(body) + len(b"%PDF-1.4\n")
    body += catalog_bytes

    raw = b"%PDF-1.4\n" + body
    xref_pos = len(raw)
    raw += f"xref\n0 {obj_num + 1}\n".encode()
    raw += b"0000000000 65535 f \n"
    for i in range(1, obj_num + 1):
        raw += f"{offsets.get(i, 0):010d} 00000 n \n".encode()
    raw += (
        f"trailer\n<</Size {obj_num + 1} /Root {catalog_id} 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    path.write_bytes(raw)
    return path


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
