r"""Utilities to work with the optional ``pdfplumber`` dependency."""

from __future__ import annotations

__all__ = [
    "check_pdfplumber",
    "is_pdfplumber_available",
    "pdfplumber_available",
    "raise_pdfplumber_missing_error",
]

from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from coola.utils.imports import (
    decorator_package_available,
    package_available,
    raise_package_missing_error,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_pdfplumber() -> None:
    r"""Check if the ``pdfplumber`` package is installed.

    Raises:
        RuntimeError: if the ``pdfplumber`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import check_pdfplumber
        >>> try:
        ...     check_pdfplumber()
        ... except RuntimeError:
        ...     pass
        ...

        ```
    """
    if not is_pdfplumber_available():
        raise_pdfplumber_missing_error()


@lru_cache
def is_pdfplumber_available() -> bool:
    r"""Indicate if the ``pdfplumber`` package is installed or not.

    Returns:
        ``True`` if ``pdfplumber`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import is_pdfplumber_available
        >>> is_pdfplumber_available()

        ```
    """
    return package_available("pdfplumber")


def pdfplumber_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if
    ``pdfplumber`` is installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn``. When ``pdfplumber`` is unavailable, calling
            the wrapper returns ``None``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import pdfplumber_available
        >>> @pdfplumber_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pdfplumber_available)


def raise_pdfplumber_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``pdfplumber`` package
    is missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``pdfplumber`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import raise_pdfplumber_missing_error
        >>> try:
        ...     raise_pdfplumber_missing_error()
        ... except RuntimeError as e:
        ...     "'pdfplumber' package is required" in str(e)
        ...
        True

        ```
    """
    raise_package_missing_error("pdfplumber", "pdfplumber")
