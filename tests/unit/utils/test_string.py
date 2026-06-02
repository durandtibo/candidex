from __future__ import annotations

import pytest

from candidex.utils.string import is_substring_match

########################################
#     Tests for is_substring_match     #
########################################


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param("MIT", "MIT CSAIL, Cambridge, MA, USA", True, id="a_in_b"),
        pytest.param("MIT CSAIL, Cambridge, MA, USA", "MIT", True, id="b_in_a"),
        pytest.param("mit", "MIT CSAIL", True, id="case_insensitive_lower"),
        pytest.param("MIT", "mit csail", True, id="case_insensitive_upper"),
        pytest.param("Stanford", "MIT CSAIL", False, id="no_match"),
        pytest.param("MIT", "MIT", True, id="equal_strings"),
        pytest.param("", "MIT", True, id="empty_string_a"),
        pytest.param("MIT", "", True, id="empty_string_b"),
        pytest.param("", "", True, id="both_empty"),
    ],
)
def test_is_substring_match(a: str, b: str, expected: bool) -> None:
    assert is_substring_match(a, b) == expected
