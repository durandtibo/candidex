from __future__ import annotations

import pytest

from candidex.testing.fixtures import colorlog_available, colorlog_not_available
from candidex.utils.imports import check_colorlog, is_colorlog_available

####################
#     colorlog     #
####################


@colorlog_available
def test_check_colorlog_with_package() -> None:
    check_colorlog()


@colorlog_not_available
def test_check_colorlog_without_package() -> None:
    with pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."):
        check_colorlog()


@colorlog_available
def test_is_colorlog_available_true() -> None:
    assert is_colorlog_available()


@colorlog_not_available
def test_is_colorlog_available_false() -> None:
    assert not is_colorlog_available()
