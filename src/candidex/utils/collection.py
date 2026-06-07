r"""Contain collection utilities."""

from __future__ import annotations

__all__ = ["flatten"]

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def flatten(items: Iterable[Any]) -> list[Any]:
    """Recursively flatten a nested iterable into a flat list.

    Flattens arbitrarily deep nested iterables (lists, tuples, generators,
    etc.) into a single flat list. Strings are treated as atomic items and
    not flattened character by character.

    Args:
        items: A nested iterable of arbitrary depth to flatten.

    Returns:
        A flat list containing all non-iterable items from the input,
            in depth-first order.

    Example:
        ```pycon
        >>> from candidex.utils.collection import flatten
        >>> flatten([[1, 2], [3, 4]])
        [1, 2, 3, 4]
        >>> flatten([[1, [2, 3]], [4, [5, [6]]]])
        [1, 2, 3, 4, 5, 6]
        >>> flatten(["MIT", ["Stanford", ["CMU"]]])
        ['MIT', 'Stanford', 'CMU']

        ```
    """
    result = []
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, str):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
