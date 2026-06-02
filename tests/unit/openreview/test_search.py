from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from openreview import OpenReviewException
from openreview.api import OpenReviewClient

from candidex.openreview import search_profiles_by_name

MODULE = "candidex.openreview.search"

#############################################
#     Tests for search_profiles_by_name     #
#############################################

# --- Fixtures ---


@pytest.fixture
def mock_client() -> Mock:
    return Mock(spec=OpenReviewClient)


@pytest.fixture
def mock_profiles() -> list[Mock]:
    return [Mock(id="~Thibaut_Durand1"), Mock(id="~Thibaut_Durand2")]


# --- No client available ---


def test_search_profiles_by_name_returns_none_when_no_client() -> None:
    with patch(f"{MODULE}.create_openreview_client", return_value=None):
        result = search_profiles_by_name("Thibaut Durand")
    assert result is None


# --- Client provided vs created ---


def test_search_profiles_by_name_uses_provided_client(
    mock_client: Mock,
    mock_profiles: list[Mock],
) -> None:
    mock_client.search_profiles.return_value = mock_profiles
    with patch(f"{MODULE}.create_openreview_client") as mock_create:
        search_profiles_by_name("Thibaut Durand", client=mock_client)
        mock_create.assert_not_called()


def test_search_profiles_by_name_creates_client_when_none_provided(
    mock_client: Mock,
    mock_profiles: list[Mock],
) -> None:
    mock_client.search_profiles.return_value = mock_profiles
    with patch(f"{MODULE}.create_openreview_client", return_value=mock_client) as mock_create:
        search_profiles_by_name("Thibaut Durand")
        mock_create.assert_called_once()


# --- Name stripping ---


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("Thibaut Durand", "Thibaut Durand", id="no_whitespace"),
        pytest.param("  Thibaut Durand  ", "Thibaut Durand", id="leading_trailing_whitespace"),
        pytest.param("\tThibaut Durand\n", "Thibaut Durand", id="tabs_and_newlines"),
    ],
)
def test_search_profiles_by_name_strips_name(
    name: str,
    expected: str,
    mock_client: Mock,
) -> None:
    mock_client.search_profiles.return_value = []
    search_profiles_by_name(name, client=mock_client)
    mock_client.search_profiles.assert_called_once_with(term=expected)


# --- Successful search ---


def test_search_profiles_by_name_returns_profiles(
    mock_client: Mock,
    mock_profiles: list[Mock],
) -> None:
    mock_client.search_profiles.return_value = mock_profiles
    result = search_profiles_by_name("Thibaut Durand", client=mock_client)
    assert result == mock_profiles


def test_search_profiles_by_name_returns_empty_list_when_no_results(
    mock_client: Mock,
) -> None:
    mock_client.search_profiles.return_value = []
    result = search_profiles_by_name("Unknown Person", client=mock_client)
    assert result == []


# --- API failure ---


def test_search_profiles_by_name_returns_none_on_api_error(
    mock_client: Mock,
) -> None:
    mock_client.search_profiles.side_effect = OpenReviewException("API error")
    result = search_profiles_by_name("Thibaut Durand", client=mock_client)
    assert result is None
