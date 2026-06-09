from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from candidex.utils.imports import (
    check_pdfminer,
    is_pdfminer_available,
    pdfminer_available,
    raise_pdfminer_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pdfminer_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


####################
#     pdfminer     #
####################


def test_check_pdfminer_with_package() -> None:
    with patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: True):
        check_pdfminer()


def test_check_pdfminer_without_package() -> None:
    with (
        patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pdfminer' package is required but not installed."),
    ):
        check_pdfminer()


def test_is_pdfminer_available() -> None:
    assert isinstance(is_pdfminer_available(), bool)


def test_pdfminer_available_with_package() -> None:
    with patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: True):
        fn = pdfminer_available(my_function)
        assert fn(2) == 44


def test_pdfminer_available_without_package() -> None:
    with patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: False):
        fn = pdfminer_available(my_function)
        assert fn(2) is None


def test_pdfminer_available_decorator_with_package() -> None:
    with patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: True):

        @pdfminer_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pdfminer_available_decorator_without_package() -> None:
    with patch("candidex.utils.imports.pdfminer.is_pdfminer_available", lambda: False):

        @pdfminer_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pdfminer_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pdfminer' package is required but not installed."):
        raise_pdfminer_missing_error()
