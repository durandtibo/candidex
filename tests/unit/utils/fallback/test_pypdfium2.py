from __future__ import annotations

from types import ModuleType

import pytest

from candidex.utils.fallback.pypdfium2 import pypdfium2


def test_pypdfium2_is_module_type() -> None:
    assert isinstance(pypdfium2, ModuleType)


def test_pypdfium2_module_name() -> None:
    assert pypdfium2.__name__ == "pypdfium2"


def test_pypdfium2_pdf_document_exists() -> None:
    assert hasattr(pypdfium2, "PdfDocument")


def test_pypdfium2_pdf_document_is_class() -> None:
    assert isinstance(pypdfium2.PdfDocument, type)


def test_pypdfium2_pdf_document_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdfium2' package is required but not installed."):
        pypdfium2.PdfDocument()
