r"""Contain mapping utilities."""

from __future__ import annotations

__all__ = ["remove_keys"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def remove_keys(data: dict[Any, Any], keys: Sequence[Any]) -> dict[Any, Any]:
    """Return a copy of a dict with the specified keys removed.

    Does not modify the original dict. Keys that are not present in the
    dict are silently ignored.

    Args:
        data: The source dictionary to filter.
        keys: List of keys to remove from the dictionary.

    Returns:
        A new dictionary with the specified keys omitted.

    Example:
        ```pycon
        >>> from candidex.utils.mapping import remove_keys
        >>> remove_keys({"a": 1, "b": 2, "c": 3}, ["a", "c"])
        {'b': 2}
        >>> remove_keys({"a": 1}, ["x"])  # missing keys are ignored
        {'a': 1}

        ```
    """
    keys_to_remove = set(keys)
    return {k: v for k, v in data.items() if k not in keys_to_remove}
