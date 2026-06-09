r"""Utilities to work with the optional ``pdfminer`` dependency."""

from __future__ import annotations

__all__ = [
    "check_pdfminer",
    "is_pdfminer_available",
    "pdfminer_available",
    "raise_pdfminer_missing_error",
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


def check_pdfminer() -> None:
    r"""Check if the ``pdfminer`` package is installed.

    Raises:
        RuntimeError: if the ``pdfminer`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import check_pdfminer
        >>> try:
        ...     check_pdfminer()
        ... except RuntimeError:
        ...     pass
        ...

        ```
    """
    if not is_pdfminer_available():
        raise_pdfminer_missing_error()


@lru_cache
def is_pdfminer_available() -> bool:
    r"""Indicate if the ``pdfminer`` package is installed or not.

    Returns:
        ``True`` if ``pdfminer`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import is_pdfminer_available
        >>> is_pdfminer_available()

        ```
    """
    return package_available("pdfminer")


def pdfminer_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``pdfminer``
    is installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn``. When ``pdfminer`` is unavailable, calling
            the wrapper returns ``None``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import pdfminer_available
        >>> @pdfminer_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pdfminer_available)


def raise_pdfminer_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``pdfminer`` package is
    missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``pdfminer`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import raise_pdfminer_missing_error
        >>> try:
        ...     raise_pdfminer_missing_error()
        ... except RuntimeError as e:
        ...     "'pdfminer' package is required" in str(e)
        ...
        True

        ```
    """
    raise_package_missing_error("pdfminer", "pdfminer.six")
