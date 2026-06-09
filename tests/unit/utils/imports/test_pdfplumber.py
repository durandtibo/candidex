from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from candidex.utils.imports import (
    check_pdfplumber,
    is_pdfplumber_available,
    pdfplumber_available,
    raise_pdfplumber_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pdfplumber_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


######################
#     pdfplumber     #
######################


def test_check_pdfplumber_with_package() -> None:
    with patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: True):
        check_pdfplumber()


def test_check_pdfplumber_without_package() -> None:
    with (
        patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pdfplumber' package is required but not installed."),
    ):
        check_pdfplumber()


def test_is_pdfplumber_available() -> None:
    assert isinstance(is_pdfplumber_available(), bool)


def test_pdfplumber_available_with_package() -> None:
    with patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: True):
        fn = pdfplumber_available(my_function)
        assert fn(2) == 44


def test_pdfplumber_available_without_package() -> None:
    with patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: False):
        fn = pdfplumber_available(my_function)
        assert fn(2) is None


def test_pdfplumber_available_decorator_with_package() -> None:
    with patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: True):

        @pdfplumber_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pdfplumber_available_decorator_without_package() -> None:
    with patch("candidex.utils.imports.pdfplumber.is_pdfplumber_available", lambda: False):

        @pdfplumber_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pdfplumber_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pdfplumber' package is required but not installed."):
        raise_pdfplumber_missing_error()
