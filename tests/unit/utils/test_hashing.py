from __future__ import annotations

import pytest

from candidex.utils.hashing import combine_hashes

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

####################################
#     Tests for combine_hashes     #
####################################


def test_combine_hashes_returns_64_char_lowercase_hex() -> None:
    digest = combine_hashes([HASH_A, HASH_B])
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_combine_hashes_same_input_same_output() -> None:
    assert combine_hashes([HASH_A, HASH_B]) == combine_hashes([HASH_A, HASH_B])


def test_combine_hashes_single_element_differs_from_original() -> None:
    """Combining a single hash must not be a no-op — the result differs
    from the input."""
    assert combine_hashes([HASH_A]) != HASH_A


def test_combine_hashes_order_matters() -> None:
    assert combine_hashes([HASH_A, HASH_B]) != combine_hashes([HASH_B, HASH_A])


def test_combine_hashes_different_inputs_different_output() -> None:
    assert combine_hashes([HASH_A, HASH_B]) != combine_hashes([HASH_A, HASH_C])


def test_combine_hashes_raises_on_empty_list() -> None:
    with pytest.raises(ValueError, match=r"Cannot combine an empty list of hashes."):
        combine_hashes([])
