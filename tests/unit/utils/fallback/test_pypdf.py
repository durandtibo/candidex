from __future__ import annotations

from types import ModuleType

import pytest

from candidex.utils.fallback.pypdf import pypdf


def test_pypdf_is_module_type() -> None:
    assert isinstance(pypdf, ModuleType)


def test_pypdf_module_name() -> None:
    assert pypdf.__name__ == "pypdf"


def test_pypdf_pdf_reader_exists() -> None:
    assert hasattr(pypdf, "PdfReader")


def test_pypdf_pdf_reader_is_class() -> None:
    assert isinstance(pypdf.PdfReader, type)


def test_pypdf_pdf_reader_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."):
        pypdf.PdfReader()
