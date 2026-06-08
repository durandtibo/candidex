from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from candidex.utils.imports import (
    check_pypdf,
    is_pypdf_available,
    pypdf_available,
    raise_pypdf_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pypdf_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#################
#     pypdf     #
#################


def test_check_pypdf_with_package() -> None:
    with patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: True):
        check_pypdf()


def test_check_pypdf_without_package() -> None:
    with (
        patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."),
    ):
        check_pypdf()


def test_is_pypdf_available() -> None:
    assert isinstance(is_pypdf_available(), bool)


def test_pypdf_available_with_package() -> None:
    with patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: True):
        fn = pypdf_available(my_function)
        assert fn(2) == 44


def test_pypdf_available_without_package() -> None:
    with patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: False):
        fn = pypdf_available(my_function)
        assert fn(2) is None


def test_pypdf_available_decorator_with_package() -> None:
    with patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: True):

        @pypdf_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pypdf_available_decorator_without_package() -> None:
    with patch("candidex.utils.imports.pypdf.is_pypdf_available", lambda: False):

        @pypdf_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pypdf_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdf' package is required but not installed."):
        raise_pypdf_missing_error()
