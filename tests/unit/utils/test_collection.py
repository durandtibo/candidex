from __future__ import annotations

import pytest

from candidex.author import Author
from candidex.utils.collection import flatten

#############################
#     Tests for flatten     #
#############################


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        pytest.param([], [], id="empty"),
        pytest.param([[]], [], id="nested_empty"),
        pytest.param([[], []], [], id="multiple_empty"),
        pytest.param([[1, 2], [3, 4]], [1, 2, 3, 4], id="flat_lists"),
        pytest.param([[1, [2, 3]], [4]], [1, 2, 3, 4], id="two_levels"),
        pytest.param([[1, [2, [3, [4]]]]], [1, 2, 3, 4], id="deeply_nested"),
        pytest.param([1, 2, 3], [1, 2, 3], id="already_flat"),
        pytest.param(
            ["MIT", ["Stanford", ["CMU"]]], ["MIT", "Stanford", "CMU"], id="strings_not_split"
        ),
        pytest.param(["MIT", "Stanford"], ["MIT", "Stanford"], id="flat_strings"),
        pytest.param([[1, "MIT"], [2, "Stanford"]], [1, "MIT", 2, "Stanford"], id="mixed_types"),
        pytest.param((1, (2, (3,))), [1, 2, 3], id="nested_tuples"),
        pytest.param([[1, 2], (3, 4)], [1, 2, 3, 4], id="mixed_list_and_tuple"),
    ],
)
def test_flatten(items: list, expected: list) -> None:
    assert flatten(items) == expected


def test_flatten_does_not_modify_input() -> None:
    items = [[1, 2], [3, 4]]
    original = [[1, 2], [3, 4]]
    flatten(items)
    assert items == original


def test_flatten_generator_input() -> None:
    assert flatten(x for x in [[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_flatten_does_not_recurse_into_dataclasses() -> None:
    a = Author.from_raw("Jane Smith", ["MIT"])
    b = Author.from_raw("John Doe", ["Stanford"])
    assert flatten([[a, b]]) == [a, b]
