from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from openreview import Profile

from candidex.author import Author
from candidex.openreview import (
    extract_profile_ids_by_author,
    find_author_profile_ids,
    load_or_fetch_profile_ids,
    log_profile_ids_stats,
)
from candidex.openreview.profile_id import FilterMode

if TYPE_CHECKING:
    from pathlib import Path

    from openreview.api import OpenReviewClient

MODULE = "candidex.openreview.profile_id"

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


#############################################
#   Tests for load_or_fetch_profile_ids     #
#############################################


# --- Cache hit ---


def test_load_or_fetch_profile_ids_returns_cached_result(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    path = tmp_path / f"{author.hash()}.json"
    cached_ids = ["~Jane_Smith1"]

    with patch(f"{MODULE}.load_json", return_value=cached_ids) as mock_load:
        path.touch()
        result_author, result_ids = load_or_fetch_profile_ids(author, tmp_path, client=mock_client)

    assert result_author == author
    assert result_ids == cached_ids
    mock_load.assert_called_once_with(path)


def test_load_or_fetch_profile_ids_does_not_call_api_when_cached(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    path = tmp_path / f"{author.hash()}.json"
    path.touch()

    with (
        patch(f"{MODULE}.load_json", return_value=[]),
        patch(f"{MODULE}.find_author_profile_ids") as mock_find,
    ):
        load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
        mock_find.assert_not_called()


# --- Cache miss: client unavailable ---


def test_load_or_fetch_profile_ids_returns_none_when_no_client(
    author: Author, tmp_path: Path
) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result_author, result_ids = load_or_fetch_profile_ids(author, tmp_path)
    assert result_author == author
    assert result_ids is None


def test_load_or_fetch_profile_ids_uses_provided_client(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.find_author_profile_ids", return_value=[]),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
        mock_create.assert_not_called()


def test_load_or_fetch_profile_ids_creates_client_when_none_provided(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.find_author_profile_ids", return_value=[]),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_ids(author, tmp_path)
        mock_create.assert_called_once()


# --- Cache miss: successful fetch ---


def test_load_or_fetch_profile_ids_returns_fetched_profile_ids(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with (
        patch(f"{MODULE}.find_author_profile_ids", return_value=["~Jane_Smith1"]),
        patch(f"{MODULE}.save_json"),
    ):
        result_author, result_ids = load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
    assert result_author == author
    assert result_ids == ["~Jane_Smith1"]


def test_load_or_fetch_profile_ids_saves_result_to_disk(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    expected_path = tmp_path / f"{author.hash()}.json"
    with (
        patch(f"{MODULE}.find_author_profile_ids", return_value=["~Jane_Smith1"]),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
        mock_save.assert_called_once_with(["~Jane_Smith1"], expected_path)


def test_load_or_fetch_profile_ids_calls_api_with_correct_args(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with (
        patch(f"{MODULE}.find_author_profile_ids", return_value=[]) as mock_find,
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
        mock_find.assert_called_once_with(
            name=author.name,
            affiliation=author.format_affiliations(),
            email=author.email,
            mode=FilterMode.ANY,
            client=mock_client,
        )


# --- Cache miss: failed fetch ---


def test_load_or_fetch_profile_ids_does_not_save_when_lookup_fails(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with (
        patch(f"{MODULE}.find_author_profile_ids", return_value=None),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
        mock_save.assert_not_called()


def test_load_or_fetch_profile_ids_returns_none_when_lookup_fails(
    author: Author, tmp_path: Path, mock_client: Mock
) -> None:
    with patch(f"{MODULE}.find_author_profile_ids", return_value=None):
        _, result_ids = load_or_fetch_profile_ids(author, tmp_path, client=mock_client)
    assert result_ids is None


#############################################
#      Tests for extract_profile_ids_by_author        #
#############################################


# --- Client unavailable ---


def test_extract_profile_ids_by_author_returns_empty_dict_when_no_client(
    tmp_path: Path,
) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result = extract_profile_ids_by_author([], tmp_path)
    assert result == {}


def test_extract_profile_ids_by_author_creates_client_when_none_provided(
    tmp_path: Path, mock_client: Mock, author: Author
) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.load_or_fetch_profile_ids", return_value=(author, [])),
    ):
        extract_profile_ids_by_author([author], tmp_path)
        mock_create.assert_called_once()


def test_extract_profile_ids_by_author_uses_provided_client(
    tmp_path: Path, mock_client: Mock, author: Author
) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.load_or_fetch_profile_ids", return_value=(author, [])),
    ):
        extract_profile_ids_by_author([author], tmp_path, client=mock_client)
        mock_create.assert_not_called()


# --- Directory creation ---


def test_extract_profile_ids_by_author_creates_directory_if_not_exists(
    tmp_path: Path, mock_client: Mock
) -> None:
    path = tmp_path / "profile_ids"
    assert not path.exists()
    with patch(f"{MODULE}.load_or_fetch_profile_ids", return_value=(Mock(), [])):
        extract_profile_ids_by_author([], path, client=mock_client)
    assert path.is_dir()


# --- Results ---


def test_extract_profile_ids_by_author_returns_results_for_all_authors(
    tmp_path: Path, mock_client: Mock
) -> None:
    author_a = Author.from_raw("Jane Smith", ["MIT"])
    author_b = Author.from_raw("John Doe", ["Stanford"])

    def side_effect(
        author: Author,
        path: Path,  # noqa: ARG001
        client: OpenReviewClient,  # noqa: ARG001
    ) -> tuple[str, list[str]]:
        return (author, ["~Jane_Smith1"] if author == author_a else ["~John_Doe1"])

    with patch(f"{MODULE}.load_or_fetch_profile_ids", side_effect=side_effect):
        result = extract_profile_ids_by_author([author_a, author_b], tmp_path, client=mock_client)

    assert result == {author_a: ["~Jane_Smith1"], author_b: ["~John_Doe1"]}


def test_extract_profile_ids_by_author_returns_empty_dict_for_empty_authors(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    result = extract_profile_ids_by_author([], tmp_path, client=mock_client)
    assert result == {}


def test_extract_profile_ids_by_author_includes_failed_lookups_as_none(
    tmp_path: Path,
    mock_client: Mock,
    author: Author,
) -> None:
    with patch(f"{MODULE}.load_or_fetch_profile_ids", return_value=(author, None)):
        result = extract_profile_ids_by_author([author], tmp_path, client=mock_client)
    assert result[author] is None


# --- Summary logging ---


def test_extract_profile_ids_by_author_logs_resolved_count(
    tmp_path: Path,
    mock_client: Mock,
    author: Author,
) -> None:
    with (
        patch(f"{MODULE}.load_or_fetch_profile_ids", return_value=(author, ["~Jane_Smith1"])),
        patch(f"{MODULE}.logger") as mock_logger,
    ):
        extract_profile_ids_by_author([author], tmp_path, client=mock_client)
        mock_logger.info.assert_called_with(
            "Profile ID extraction complete. %d/%d authors resolved.",
            1,
            1,
        )


###########################################
#     Tests for log_profile_ids_stats     #
###########################################


@pytest.mark.parametrize(
    (
        "profile_ids_by_author",
        "expected_total",
        "expected_single",
        "expected_multiple",
        "expected_empty",
        "expected_failed",
    ),
    [
        pytest.param({}, 0, 0, 0, 0, 0, id="empty"),
        pytest.param(
            {make_author("Jane Smith"): ["~Jane_Smith1"]},
            1,
            1,
            0,
            0,
            0,
            id="single_match",
        ),
        pytest.param(
            {make_author("Jane Smith"): ["~Jane_Smith1", "~Jane_Smith2"]},
            1,
            0,
            1,
            0,
            0,
            id="multiple_matches",
        ),
        pytest.param(
            {make_author("Jane Smith"): []},
            1,
            0,
            0,
            1,
            0,
            id="no_match",
        ),
        pytest.param(
            {make_author("Jane Smith"): None},
            1,
            0,
            0,
            0,
            1,
            id="failed",
        ),
        pytest.param(
            {
                make_author("Jane Smith"): ["~Jane_Smith1"],
                make_author("John Doe"): ["~John_Doe1", "~John_Doe2"],
                make_author("Alice Brown"): [],
                make_author("Bob Jones"): None,
            },
            4,
            1,
            1,
            1,
            1,
            id="mixed",
        ),
    ],
)
def test_log_profile_ids_stats(
    profile_ids_by_author: dict[Author, list[str] | None],
    expected_total: int,
    expected_single: int,
    expected_multiple: int,
    expected_empty: int,
    expected_failed: int,
) -> None:
    with patch(f"{MODULE}.logger") as mock_logger:
        log_profile_ids_stats(profile_ids_by_author)
        mock_logger.info.assert_called_once_with(
            "Profile ID lookup complete for %d authors — "
            "%d single match, %d multiple matches, %d no match, %d failed.",
            expected_total,
            expected_single,
            expected_multiple,
            expected_empty,
            expected_failed,
        )
