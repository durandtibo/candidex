from __future__ import annotations

import openreview
import pytest

from candidex.openreview import parse_names_and_history_profile

# --- Fixtures ---


@pytest.fixture
def profile() -> openreview.Profile:
    return openreview.Profile(
        id="~Jane_Smith1",
        content={
            "names": [{"fullname": "Jane Smith", "first": "Jane", "last": "Smith"}],
            "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
        },
    )


@pytest.fixture
def empty_profile() -> openreview.Profile:
    return openreview.Profile(id="~Jane_Smith1", content={})


#####################################################
#     Tests for parse_names_and_history_profile     #
#####################################################

# --- Keys ---


def test_parse_names_and_history_profile_returns_exactly_two_keys(
    profile: openreview.Profile,
) -> None:
    assert set(parse_names_and_history_profile(profile).keys()) == {"names", "history"}


def test_parse_names_and_history_profile_does_not_return_id(profile: openreview.Profile) -> None:
    assert "id" not in parse_names_and_history_profile(profile)


# --- names ---


def test_parse_names_and_history_profile_returns_names(profile: openreview.Profile) -> None:
    result = parse_names_and_history_profile(profile)
    assert result["names"] == [{"fullname": "Jane Smith", "first": "Jane", "last": "Smith"}]


def test_parse_names_and_history_profile_returns_empty_names_when_missing(
    empty_profile: openreview.Profile,
) -> None:
    assert parse_names_and_history_profile(empty_profile)["names"] == []


def test_parse_names_and_history_profile_returns_multiple_names() -> None:
    profile = openreview.Profile(
        id="~Jane_Smith1",
        content={
            "names": [
                {"fullname": "Jane Smith"},
                {"fullname": "J. Smith"},
            ],
            "history": [],
        },
    )
    assert len(parse_names_and_history_profile(profile)["names"]) == 2


# --- history ---


def test_parse_names_and_history_profile_returns_history(profile: openreview.Profile) -> None:
    result = parse_names_and_history_profile(profile)
    assert result["history"] == [{"position": "PhD Student", "institution": {"name": "MIT"}}]


def test_parse_names_and_history_profile_returns_empty_history_when_missing(
    empty_profile: openreview.Profile,
) -> None:
    assert parse_names_and_history_profile(empty_profile)["history"] == []


def test_parse_names_and_history_profile_returns_multiple_history_entries() -> None:
    profile = openreview.Profile(
        id="~Jane_Smith1",
        content={
            "names": [],
            "history": [
                {"position": "PhD Student", "institution": {"name": "MIT"}},
                {"position": "Postdoc", "institution": {"name": "Stanford"}},
            ],
        },
    )
    assert len(parse_names_and_history_profile(profile)["history"]) == 2
