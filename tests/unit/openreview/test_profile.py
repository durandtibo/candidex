from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openreview import OpenReviewException, Profile

from candidex.author import Author
from candidex.openreview import fetch_profile_by_id, get_unique_profiles

MODULE = "candidex.openreview.profile"

# --- Fixtures ---


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


@pytest.fixture
def author() -> Author:
    return Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")


def make_profile(profile_id: str) -> Profile:
    return Mock(id=profile_id, spec=Profile)


def make_author(name: str) -> Author:
    return Author.from_raw(name, ["MIT"])


#########################################
#     Tests for get_unique_profiles     #
#########################################


@pytest.mark.parametrize(
    ("profiles_list", "expected_ids"),
    [
        pytest.param([], [], id="empty_input"),
        pytest.param([[]], [], id="single_empty_list"),
        pytest.param([[], []], [], id="multiple_empty_lists"),
        pytest.param(
            [[make_profile("~Jane_Smith1")]],
            ["~Jane_Smith1"],
            id="single_profile",
        ),
        pytest.param(
            [[make_profile("~Jane_Smith1"), make_profile("~John_Doe1")]],
            ["~Jane_Smith1", "~John_Doe1"],
            id="multiple_profiles_single_list",
        ),
        pytest.param(
            [
                [make_profile("~Jane_Smith1")],
                [make_profile("~John_Doe1")],
            ],
            ["~Jane_Smith1", "~John_Doe1"],
            id="unique_profiles_across_lists",
        ),
        pytest.param(
            [
                [make_profile("~Jane_Smith1")],
                [make_profile("~Jane_Smith1")],
            ],
            ["~Jane_Smith1"],
            id="duplicate_across_lists",
        ),
        pytest.param(
            [
                [make_profile("~Jane_Smith1"), make_profile("~Jane_Smith1")],
            ],
            ["~Jane_Smith1"],
            id="duplicate_within_same_list",
        ),
        pytest.param(
            [
                [make_profile("~Jane_Smith1"), make_profile("~John_Doe1")],
                [make_profile("~Jane_Smith1"), make_profile("~Alice_Brown1")],
            ],
            ["~Jane_Smith1", "~John_Doe1", "~Alice_Brown1"],
            id="partial_overlap_across_lists",
        ),
    ],
)
def test_get_unique_profiles(profiles_list: list[list[Mock]], expected_ids: list[str]) -> None:
    result = get_unique_profiles(profiles_list)
    assert [p.id for p in result] == expected_ids


def test_get_unique_profiles_preserves_first_occurrence() -> None:
    """When duplicates exist, the first occurrence should be kept."""
    profile_a = make_profile("~Jane_Smith1")
    profile_a.name = "first"
    profile_b = make_profile("~Jane_Smith1")
    profile_b.name = "second"

    result = get_unique_profiles([[profile_a], [profile_b]])
    assert len(result) == 1
    assert result[0].name == "first"


#########################################
#     Tests for fetch_profile_by_id     #
#########################################

# --- Client unavailable ---


def test_fetch_profile_by_id_returns_none_when_no_client() -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result = fetch_profile_by_id("~Thibaut_Durand1")
    assert result is None


def test_fetch_profile_by_id_uses_provided_client(mock_client: Mock) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    with patch(f"{MODULE}.create_client") as mock_create:
        fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
        mock_create.assert_not_called()


def test_fetch_profile_by_id_creates_client_when_none_provided(mock_client: Mock) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    with patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create:
        fetch_profile_by_id("~Thibaut_Durand1")
        mock_create.assert_called_once()


# --- Successful fetch ---


def test_fetch_profile_by_id_returns_profile_on_success(mock_client: Mock) -> None:
    mock_profile = make_profile("~Thibaut_Durand1")
    mock_client.get_profile.return_value = mock_profile
    result = fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
    assert result is mock_profile


def test_fetch_profile_by_id_calls_client_with_correct_id(mock_client: Mock) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
    mock_client.get_profile.assert_called_once_with("~Thibaut_Durand1")


# --- Profile not found ---


def test_fetch_profile_by_id_returns_none_when_profile_not_found(
    mock_client: Mock,
) -> None:
    mock_client.get_profile.side_effect = OpenReviewException("Not found")
    result = fetch_profile_by_id("~Unknown_Person1", client=mock_client)
    assert result is None


# --- Various ID formats ---


@pytest.mark.parametrize(
    "profile_id",
    [
        pytest.param("~Thibaut_Durand1", id="tilde_prefix"),
        pytest.param("%7EThibaut_Durand1", id="encoded_tilde_prefix"),
        pytest.param("~Jane_Smith1", id="different_author"),
    ],
)
def test_fetch_profile_by_id_accepts_various_id_formats(profile_id: str, mock_client: Mock) -> None:
    mock_profile = make_profile("~Thibaut_Durand1")
    mock_client.get_profile.return_value = mock_profile
    result = fetch_profile_by_id(profile_id, client=mock_client)
    assert result is mock_profile
    mock_client.get_profile.assert_called_once_with(profile_id)
