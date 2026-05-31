r"""Contain utility functions for hashing."""

from __future__ import annotations

__all__ = ["combine_hashes"]

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def combine_hashes(hashes: Sequence[str]) -> str:
    """Return a single SHA-256 hex digest derived from a list of hashes.

    Concatenates the input hashes in order and hashes the result, producing
    a stable digest that uniquely represents the combination. Useful for
    generating a cache key or output filename that depends on multiple
    ``ChatModelConfig`` instances.

    Args:
        hashes: List of hex digest strings to combine, typically produced
                by ``ChatModelConfig.hash()``. Order matters — the same
                hashes in a different order produce a different result.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 digest string.

    Raises:
        ValueError: If `hashes` is empty.

    Example:
        >>> a = ChatModelConfig(model="openai:gpt-4o", system_prompt="A").hash()
        >>> b = ChatModelConfig(model="openai:gpt-4o", system_prompt="B").hash()
        >>> combine_hashes([a, b])
        'e3b0c4...'
    """
    if not hashes:
        msg = "Cannot combine an empty list of hashes."
        raise ValueError(msg)
    return hashlib.sha256("".join(hashes).encode()).hexdigest()
