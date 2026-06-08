r"""Utilities to work with the optional ``pypdf`` dependency."""

from __future__ import annotations

__all__ = [
    "check_pypdf",
    "is_pypdf_available",
    "pypdf_available",
    "raise_pypdf_missing_error",
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


def check_pypdf() -> None:
    r"""Check if the ``pypdf`` package is installed.

    Raises:
        RuntimeError: if the ``pypdf`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import check_pypdf
        >>> try:
        ...     check_pypdf()
        ... except RuntimeError:
        ...     pass
        ...

        ```
    """
    if not is_pypdf_available():
        raise_pypdf_missing_error()


@lru_cache
def is_pypdf_available() -> bool:
    r"""Indicate if the ``pypdf`` package is installed or not.

    Returns:
        ``True`` if ``pypdf`` is available otherwise ``False``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import is_pypdf_available
        >>> is_pypdf_available()

        ```
    """
    return package_available("pypdf")


def pypdf_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``pypdf`` is
    installed.

    Args:
        fn: The function to conditionally execute.

    Returns:
        A wrapper around ``fn``. When ``pypdf`` is unavailable, calling
            the wrapper returns ``None``.

    Example:
        ```pycon
        >>> from candidex.utils.imports import pypdf_available
        >>> @pypdf_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_pypdf_available)


def raise_pypdf_missing_error() -> NoReturn:
    r"""Raise a ``RuntimeError`` to indicate the ``pypdf`` package is
    missing.

    Raises:
        RuntimeError: Always, with a message indicating that the
            ``pypdf`` package is not installed.

    Example:
        ```pycon
        >>> from candidex.utils.imports import raise_pypdf_missing_error
        >>> try:
        ...     raise_pypdf_missing_error()
        ... except RuntimeError as e:
        ...     "'pypdf' package is required" in str(e)
        ...
        True

        ```
    """
    raise_package_missing_error("pypdf", "pypdf")
