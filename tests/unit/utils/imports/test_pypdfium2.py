from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from candidex.utils.imports import (
    check_pypdfium2,
    is_pypdfium2_available,
    pypdfium2_available,
    raise_pypdfium2_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pypdfium2_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     pypdfium2     #
#####################


def test_check_pypdfium2_with_package() -> None:
    with patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: True):
        check_pypdfium2()


def test_check_pypdfium2_without_package() -> None:
    with (
        patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pypdfium2' package is required but not installed."),
    ):
        check_pypdfium2()


def test_is_pypdfium2_available() -> None:
    assert isinstance(is_pypdfium2_available(), bool)


def test_pypdfium2_available_with_package() -> None:
    with patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: True):
        fn = pypdfium2_available(my_function)
        assert fn(2) == 44


def test_pypdfium2_available_without_package() -> None:
    with patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: False):
        fn = pypdfium2_available(my_function)
        assert fn(2) is None


def test_pypdfium2_available_decorator_with_package() -> None:
    with patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: True):

        @pypdfium2_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pypdfium2_available_decorator_without_package() -> None:
    with patch("candidex.utils.imports.pypdfium2.is_pypdfium2_available", lambda: False):

        @pypdfium2_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pypdfium2_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pypdfium2' package is required but not installed."):
        raise_pypdfium2_missing_error()
