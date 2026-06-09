from __future__ import annotations

from types import ModuleType

import pytest

from candidex.utils.fallback.pdfplumber import pdfplumber


def test_pdfplumber_is_module_type() -> None:
    assert isinstance(pdfplumber, ModuleType)


def test_pdfplumber_module_name() -> None:
    assert pdfplumber.__name__ == "pdfplumber"


def test_pdfplumber_open_exists() -> None:
    assert hasattr(pdfplumber, "open")


def test_pdfplumber_open_is_function() -> None:
    assert callable(pdfplumber.open)


def test_pdfplumber_open_called() -> None:
    with pytest.raises(RuntimeError, match=r"'pdfplumber' package is required but not installed."):
        pdfplumber.open()
