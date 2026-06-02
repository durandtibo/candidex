from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openreview import Profile

from candidex.openreview.profile import find_openreview_profile

MODULE = "candidex.openreview.profile"

# --- Fixtures ---


@pytest.fixture
def mock_client() -> Mock:
    return Mock()


def make_profile(profile_id: str) -> Mock:
    return Mock(id=profile_id, spec=Profile)


#############################################
#     Tests for find_openreview_profile     #
#############################################


# --- Client unavailable ---


def test_find_openreview_profile_returns_none_when_no_client() -> None:
    with patch(f"{MODULE}.create_openreview_client", return_value=None):
        result = find_openreview_profile(name="Jane Smith", affiliation="MIT CSAIL")
    assert result is None


def test_find_openreview_profile_uses_provided_client(mock_client: Mock) -> None:
    with (
        patch(f"{MODULE}.create_openreview_client") as mock_create,
        patch(f"{MODULE}.search_openreview_profiles", return_value=[]),
    ):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
        assert result == []
        mock_create.assert_not_called()


def test_find_openreview_profile_creates_client_when_none_provided(mock_client: Mock) -> None:
    with (
        patch(f"{MODULE}.create_openreview_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.search_openreview_profiles", return_value=[]),
    ):
        result = find_openreview_profile(name="Jane Smith", affiliation="MIT CSAIL")
        assert result == []
        mock_create.assert_called_once()


# --- Search failure ---


def test_find_openreview_profile_returns_none_when_search_fails(mock_client: Mock) -> None:
    with patch(f"{MODULE}.search_openreview_profiles", return_value=None):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result is None


# --- No profiles found ---


def test_find_openreview_profile_returns_empty_list_when_no_profiles(mock_client: Mock) -> None:
    with patch(f"{MODULE}.search_openreview_profiles", return_value=[]):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == []


# --- Affiliation filtering ---


def test_find_openreview_profile_returns_empty_list_when_no_affiliation_match(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=[]),
    ):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == []


def test_find_openreview_profile_returns_sorted_ids_when_affiliation_matches(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1"), make_profile("~Jane_Smith2")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
    ):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]


def test_find_openreview_profile_returns_ids_sorted_alphabetically(mock_client: Mock) -> None:
    profiles = [make_profile("~Jane_Smith2"), make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
    ):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", client=mock_client
        )
    assert result == ["~Jane_Smith1", "~Jane_Smith2"]


# --- Email filtering ---


def test_find_openreview_profile_skips_email_filter_when_none(mock_client: Mock) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_email") as mock_email_filter,
    ):
        find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", email=None, client=mock_client
        )
        mock_email_filter.assert_not_called()


def test_find_openreview_profile_applies_email_filter_when_provided(mock_client: Mock) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=profiles) as mock_email_filter,
    ):
        find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", email="jane@mit.edu", client=mock_client
        )
        mock_email_filter.assert_called_once_with(profiles, email="jane@mit.edu")


def test_find_openreview_profile_returns_empty_list_when_email_filter_excludes_all(
    mock_client: Mock,
) -> None:
    profiles = [make_profile("~Jane_Smith1")]
    with (
        patch(f"{MODULE}.search_openreview_profiles", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_affiliation", return_value=profiles),
        patch(f"{MODULE}.filter_profiles_by_email", return_value=[]),
    ):
        result = find_openreview_profile(
            name="Jane Smith", affiliation="MIT CSAIL", email="jane@google.com", client=mock_client
        )
    assert result == []
