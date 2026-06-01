from __future__ import annotations

import pytest

from candidex.utils.mapping import remove_keys

#################################
#     Tests for remove_keys     #
#################################


@pytest.mark.parametrize(
    ("data", "keys", "expected"),
    [
        pytest.param({"a": 1, "b": 2, "c": 3}, ["a", "c"], {"b": 2}, id="remove_multiple"),
        pytest.param({"a": 1, "b": 2}, ["a"], {"b": 2}, id="remove_single"),
        pytest.param({"a": 1}, ["x"], {"a": 1}, id="missing_key_ignored"),
        pytest.param({"a": 1, "b": 2}, ["a", "b"], {}, id="remove_all"),
        pytest.param({"a": 1}, [], {"a": 1}, id="empty_keys"),
        pytest.param({}, ["a"], {}, id="empty_dict"),
    ],
)
def test_remove_keys(data: dict, keys: list[str], expected: dict) -> None:
    assert remove_keys(data, keys) == expected


def test_remove_keys_does_not_modify_original() -> None:
    data = {"a": 1, "b": 2}
    remove_keys(data, ["a"])
    assert data == {"a": 1, "b": 2}
