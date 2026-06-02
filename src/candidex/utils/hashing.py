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
        >>> from candidex.utils.hashing import combine_hashes
        >>> a = '42db77cda5f9aed8970129d2a9237bdf'
        >>> b = '3524dbb6b14a5e280f2a5e6bd8c1f7a5'
        >>> combine_hashes([a, b])  # doctest: +SKIP
        'e3b0c4...'
    """
    if not hashes:
        msg = "Cannot combine an empty list of hashes."
        raise ValueError(msg)
    return hashlib.sha256("".join(hashes).encode()).hexdigest()
