from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openreview import Profile

from candidex.openreview import find_author_profile_ids, get_unique_profiles
from candidex.openreview.profile import FilterMode

MODULE = "candidex.openreview.profile"

# --- Fixtures ---


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


def make_profile(profile_id: str) -> Mock:
    return Mock(id=profile_id, spec=Profile)


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


#############################################
#     Tests for find_author_profile_ids     #
#############################################


# --- Client unavailable ---


def test_find_author_profile_ids_returns_none_when_no_client() -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result = find_author_profile_ids(name="Jane Smith", affiliation="MIT CSAIL")
    assert result is None


def test_find_author_profile_ids_uses_provided_client(mock_client: Mock) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.search_profiles_by_name", return_value=[]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
        assert result == []
        mock_create.assert_not_called()


def test_find_author_profile_ids_creates_client_when_none_provided(mock_client: Mock) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.search_profiles_by_name", return_value=[]),
    ):
        result = find_author_profile_ids(name="Jane Smith", affiliation="MIT CSAIL")
        assert result == []
        mock_create.assert_called_once()


# --- Search failure ---


def test_find_author_profile_ids_returns_none_when_search_fails(mock_client: Mock) -> None:
    with patch(f"{MODULE}.search_profiles_by_name", return_value=None):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result is None


# --- No profiles found ---


def test_find_author_profile_ids_returns_empty_list_when_no_profiles(mock_client: Mock) -> None:
    with patch(f"{MODULE}.search_profiles_by_name", return_value=[]):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == []


# --- Affiliation filtering ---


def test_find_author_profile_ids_returns_empty_list_when_no_affiliation_match(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[]),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == []


def test_find_author_profile_ids_returns_sorted_ids_when_affiliation_matches(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1"), make_profile("~Jane_Smith2")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
    ):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]


def test_find_author_profile_ids_returns_ids_sorted_alphabetically(mock_client: Mock) -> None:
    profiles = [make_profile("~Jane_Smith2"), make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
    ):
        result = find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]


# --- Email filtering: no email ---


def test_find_author_profile_ids_skips_email_filter_when_none(mock_client: Mock) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_email") as mock_email_filter,
    ):
        find_author_profile_ids(
            name="Jane Smith", affiliation="MIT CSAIL", email=None, client=mock_client
        )
        mock_email_filter.assert_not_called()


# --- Email filtering: FilterMode.ANY (default) ---


def test_find_author_profile_ids_any_mode_applies_email_filter_on_all_profiles(
    mock_client: Mock,
) -> None:
    """In ANY mode, email filter runs on all profiles, not just
    affiliation matches."""
    profiles = [make_profile("~Jane_Smith1"), make_profile("~Jane_Smith2")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[]),
        patch(
            f"{MODULE}.filter_profiles_by_email", return_value=[make_profile("~Jane_Smith2")]
        ) as mock_email_filter,
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            mode=FilterMode.ANY,
            client=mock_client,
        )
        mock_email_filter.assert_called_once_with(profiles, email="jane@mit.edu")
    assert result == ["~Jane_Smith2"]


def test_find_author_profile_ids_any_mode_returns_union_of_affiliation_and_email(
    mock_client: Mock,
) -> None:
    profile_a = make_profile("~Jane_Smith1")
    profile_b = make_profile("~Jane_Smith2")
    profiles = [profile_a, profile_b]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[profile_a]),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[profile_b]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            mode=FilterMode.ANY,
            client=mock_client,
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]


def test_find_author_profile_ids_any_mode_deduplicates_results(mock_client: Mock) -> None:
    profile = make_profile("~Jane_Smith1")
    profiles = [profile]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[profile]),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[profile]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            mode=FilterMode.ANY,
            client=mock_client,
        )
    assert result == ["~Jane_Smith1"]


# --- Email filtering: FilterMode.ALL ---


def test_find_author_profile_ids_all_mode_applies_email_filter_on_affiliation_matches(
    mock_client: Mock,
) -> None:
    """In ALL mode, email filter runs only on affiliation-matched
    profiles."""
    profiles = [make_profile("~Jane_Smith1"), make_profile("~Jane_Smith2")]
    affiliation_matches = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=affiliation_matches),
        patch(
            f"{MODULE}.filter_profiles_by_email", return_value=affiliation_matches
        ) as mock_email_filter,
    ):
        find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            mode=FilterMode.ALL,
            client=mock_client,
        )
        mock_email_filter.assert_called_once_with(affiliation_matches, email="jane@mit.edu")


def test_find_author_profile_ids_all_mode_returns_intersection(mock_client: Mock) -> None:
    profile_a = make_profile("~Jane_Smith1")
    profile_b = make_profile("~Jane_Smith2")
    profiles = [profile_a, profile_b]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[profile_a, profile_b]),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[profile_a]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            mode=FilterMode.ALL,
            client=mock_client,
        )
    assert result == ["~Jane_Smith1"]


def test_find_author_profile_ids_all_mode_returns_empty_when_email_excludes_all(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@google.com",
            mode=FilterMode.ALL,
            client=mock_client,
        )
    assert result == []


# --- Default mode ---


def test_find_author_profile_ids_default_mode_is_any(mock_client: Mock) -> None:
    """Default mode should behave identically to FilterMode.ANY."""
    profile_a = make_profile("~Jane_Smith1")
    profile_b = make_profile("~Jane_Smith2")
    profiles = [profile_a, profile_b]
    with (
        patch(f"{MODULE}.search_profiles_by_name", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[profile_a]),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[profile_b]),
    ):
        result = find_author_profile_ids(
            name="Jane Smith",
            affiliation="MIT CSAIL",
            email="jane@mit.edu",
            client=mock_client,
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]
