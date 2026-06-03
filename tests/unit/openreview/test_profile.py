from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from iden.io import save_json
from openreview import OpenReviewException, Profile
from openreview.api import OpenReviewClient

from candidex.author import Author
from candidex.openreview import (
    extract_profiles_by_author,
    extract_profiles_by_id,
    fetch_profile_by_id,
    get_unique_profiles,
    load_or_fetch_profile,
    load_or_fetch_profile_by_id,
)

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "candidex.openreview.profile"

# --- Fixtures ---


@pytest.fixture
def mock_client() -> OpenReviewClient:
    return Mock(spec=OpenReviewClient)


@pytest.fixture
def author() -> Author:
    return Author.from_raw("Jane Smith", ["MIT"], "jane@mit.edu")


@pytest.fixture
def mock_profile() -> Profile:
    profile = Mock(spec=Profile, id="~Jane_Smith1")
    profile.to_json.return_value = {"id": "~Jane_Smith1"}
    return profile


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


def test_fetch_profile_by_id_uses_provided_client(mock_client: OpenReviewClient) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    with patch(f"{MODULE}.create_client") as mock_create:
        fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
        mock_create.assert_not_called()


def test_fetch_profile_by_id_creates_client_when_none_provided(
    mock_client: OpenReviewClient,
) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    with patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create:
        fetch_profile_by_id("~Thibaut_Durand1")
        mock_create.assert_called_once()


# --- Successful fetch ---


def test_fetch_profile_by_id_returns_profile_on_success(mock_client: OpenReviewClient) -> None:
    mock_profile = make_profile("~Thibaut_Durand1")
    mock_client.get_profile.return_value = mock_profile
    result = fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
    assert result is mock_profile


def test_fetch_profile_by_id_calls_client_with_correct_id(mock_client: OpenReviewClient) -> None:
    mock_client.get_profile.return_value = make_profile("~Thibaut_Durand1")
    fetch_profile_by_id("~Thibaut_Durand1", client=mock_client)
    mock_client.get_profile.assert_called_once_with("~Thibaut_Durand1")


# --- Profile not found ---


def test_fetch_profile_by_id_returns_none_when_profile_not_found(
    mock_client: OpenReviewClient,
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
def test_fetch_profile_by_id_accepts_various_id_formats(
    profile_id: str, mock_client: OpenReviewClient
) -> None:
    mock_profile = make_profile("~Thibaut_Durand1")
    mock_client.get_profile.return_value = mock_profile
    result = fetch_profile_by_id(profile_id, client=mock_client)
    assert result is mock_profile
    mock_client.get_profile.assert_called_once_with(profile_id)


#############################################
#      Tests for load_or_fetch_profile      #
#############################################


# --- Cache hit ---


def test_load_or_fetch_profile_returns_cached_result(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:
    save_json({}, tmp_path / "~Jane_Smith1.json")

    with (
        patch(f"{MODULE}.load_json", return_value={"id": "~Jane_Smith1"}),
        patch(f"{MODULE}.Profile.from_json", return_value=mock_profile),
    ):
        result_author, result_id, result_profile = load_or_fetch_profile(
            author, "~Jane_Smith1", tmp_path, client=mock_client
        )

    assert result_author == author
    assert result_id == "~Jane_Smith1"
    assert result_profile is mock_profile


def test_load_or_fetch_profile_does_not_call_api_when_cached(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:
    save_json({}, tmp_path / "~Jane_Smith1.json")

    with (
        patch(f"{MODULE}.load_json", return_value={}),
        patch(f"{MODULE}.Profile.from_json", return_value=mock_profile),
        patch(f"{MODULE}.fetch_profile_by_id") as mock_fetch,
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path, client=mock_client)
        mock_fetch.assert_not_called()


# --- Cache miss: client unavailable ---


def test_load_or_fetch_profile_returns_none_when_no_client(author: Author, tmp_path: Path) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result_author, result_id, result_profile = load_or_fetch_profile(
            author, "~Jane_Smith1", tmp_path
        )
    assert result_author == author
    assert result_id == "~Jane_Smith1"
    assert result_profile is None


def test_load_or_fetch_profile_uses_provided_client(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:

    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path, client=mock_client)
        mock_create.assert_not_called()


def test_load_or_fetch_profile_creates_client_when_none_provided(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:

    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path)
        mock_create.assert_called_once()


# --- Cache miss: successful fetch ---


def test_load_or_fetch_profile_returns_fetched_profile(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:

    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        result_author, result_id, result_profile = load_or_fetch_profile(
            author, "~Jane_Smith1", tmp_path, client=mock_client
        )
    assert result_author == author
    assert result_id == "~Jane_Smith1"
    assert result_profile is mock_profile


def test_load_or_fetch_profile_saves_profile_to_disk(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:

    expected_path = tmp_path / "~Jane_Smith1.json"
    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path, client=mock_client)
        mock_save.assert_called_once_with(mock_profile.to_json(), expected_path)


def test_load_or_fetch_profile_calls_fetch_with_correct_id(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
    mock_profile: Mock,
) -> None:

    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile) as mock_fetch,
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path, client=mock_client)
        mock_fetch.assert_called_once_with("~Jane_Smith1", client=mock_client)


# --- Cache miss: failed fetch ---


def test_load_or_fetch_profile_returns_none_when_fetch_fails(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
) -> None:

    with patch(f"{MODULE}.fetch_profile_by_id", return_value=None):
        _, _, result_profile = load_or_fetch_profile(
            author, "~Jane_Smith1", tmp_path, client=mock_client
        )
    assert result_profile is None


def test_load_or_fetch_profile_does_not_save_when_fetch_fails(
    author: Author,
    tmp_path: Path,
    mock_client: OpenReviewClient,
) -> None:

    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=None),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile(author, "~Jane_Smith1", tmp_path, client=mock_client)
        mock_save.assert_not_called()


#############################################
#    Tests for extract_profiles_by_author   #
#############################################


# --- Client unavailable ---


def test_extract_profiles_by_author_returns_empty_lists_when_no_client(
    tmp_path: Path,
    author: Author,
) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result = extract_profiles_by_author({author: ["~Jane_Smith1"]}, tmp_path)
    assert result == {author: None}


def test_extract_profiles_by_author_uses_provided_client(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(
            f"{MODULE}.load_or_fetch_profile", return_value=(author, "~Jane_Smith1", mock_profile)
        ),
    ):
        extract_profiles_by_author({author: ["~Jane_Smith1"]}, tmp_path, client=mock_client)
        mock_create.assert_not_called()


def test_extract_profiles_by_author_creates_client_when_none_provided(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(
            f"{MODULE}.load_or_fetch_profile", return_value=(author, "~Jane_Smith1", mock_profile)
        ),
    ):
        extract_profiles_by_author({author: ["~Jane_Smith1"]}, tmp_path)
        mock_create.assert_called_once()


# --- Directory creation ---


def test_extract_profiles_by_author_creates_directory_if_not_exists(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
    mock_profile: Mock,
) -> None:
    path = tmp_path / "profiles"
    assert not path.exists()
    with patch(
        f"{MODULE}.load_or_fetch_profile", return_value=(author, "~Jane_Smith1", mock_profile)
    ):
        extract_profiles_by_author({author: ["~Jane_Smith1"]}, path, client=mock_client)
    assert path.exists()


# --- Skipping authors ---


def test_extract_profiles_by_author_skips_none_profile_ids(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
) -> None:
    with patch(f"{MODULE}.load_or_fetch_profile") as mock_fetch:
        result = extract_profiles_by_author({author: None}, tmp_path, client=mock_client)
        mock_fetch.assert_not_called()
    assert result[author] == []


def test_extract_profiles_by_author_skips_empty_profile_ids(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
) -> None:
    with patch(f"{MODULE}.load_or_fetch_profile") as mock_fetch:
        result = extract_profiles_by_author({author: []}, tmp_path, client=mock_client)
        mock_fetch.assert_not_called()
    assert result[author] == []


# --- Results ---


def test_extract_profiles_by_author_returns_profiles(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
    mock_profile: Mock,
) -> None:
    with patch(
        f"{MODULE}.load_or_fetch_profile", return_value=(author, "~Jane_Smith1", mock_profile)
    ):
        result = extract_profiles_by_author(
            {author: ["~Jane_Smith1"]}, tmp_path, client=mock_client
        )
    assert result[author] == [mock_profile]


def test_extract_profiles_by_author_returns_multiple_profiles_per_author(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
) -> None:
    profile_1 = Mock(spec=Profile, id="~Jane_Smith1")
    profile_2 = Mock(spec=Profile, id="~Jane_Smith2")

    def side_effect(
        author: Author,
        profile_id: str,
        path: Path,  # noqa: ARG001
        client: OpenReviewClient,  # noqa: ARG001
    ) -> tuple[Author, str, Profile | None]:
        return (author, profile_id, profile_1 if profile_id == "~Jane_Smith1" else profile_2)

    with patch(f"{MODULE}.load_or_fetch_profile", side_effect=side_effect):
        result = extract_profiles_by_author(
            {author: ["~Jane_Smith1", "~Jane_Smith2"]}, tmp_path, client=mock_client
        )
    assert len(result[author]) == 2


def test_extract_profiles_by_author_excludes_failed_fetches(
    tmp_path: Path,
    mock_client: OpenReviewClient,
    author: Author,
) -> None:
    with patch(f"{MODULE}.load_or_fetch_profile", return_value=(author, "~Jane_Smith1", None)):
        result = extract_profiles_by_author(
            {author: ["~Jane_Smith1"]}, tmp_path, client=mock_client
        )
    assert result[author] == []


def test_extract_profiles_by_author_handles_multiple_authors(
    tmp_path: Path,
    mock_client: OpenReviewClient,
) -> None:
    author_a = Author.from_raw("Jane Smith", ["MIT"])
    author_b = Author.from_raw("John Doe", ["Stanford"])
    profile_a = Mock(spec=Profile)
    profile_b = Mock(spec=Profile)

    def side_effect(
        author: Author,
        profile_id: str,
        path: Path,  # noqa: ARG001
        client: OpenReviewClient,  # noqa: ARG001
    ) -> tuple[Author, str, Profile | None]:
        return (author, profile_id, profile_a if author == author_a else profile_b)

    with patch(f"{MODULE}.load_or_fetch_profile", side_effect=side_effect):
        result = extract_profiles_by_author(
            {author_a: ["~Jane_Smith1"], author_b: ["~John_Doe1"]},
            tmp_path,
            client=mock_client,
        )
    assert result[author_a] == [profile_a]
    assert result[author_b] == [profile_b]


def test_extract_profiles_by_author_empty_input(
    tmp_path: Path, mock_client: OpenReviewClient
) -> None:
    result = extract_profiles_by_author({}, tmp_path, client=mock_client)
    assert result == {}


#############################################
#   Tests for load_or_fetch_profile_by_id   #
#############################################


# --- Cache hit ---


def test_load_or_fetch_profile_by_id_returns_cached_result(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    save_json({}, tmp_path / "~Jane_Smith1.json")

    with (
        patch(f"{MODULE}.load_json", return_value={"id": "~Jane_Smith1"}),
        patch(f"{MODULE}.Profile.from_json", return_value=mock_profile),
    ):
        result_id, result_profile = load_or_fetch_profile_by_id(
            "~Jane_Smith1", tmp_path, client=mock_client
        )

    assert result_id == "~Jane_Smith1"
    assert result_profile is mock_profile


def test_load_or_fetch_profile_by_id_does_not_call_api_when_cached(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    save_json({}, tmp_path / "~Jane_Smith1.json")

    with (
        patch(f"{MODULE}.load_json", return_value={}),
        patch(f"{MODULE}.Profile.from_json", return_value=mock_profile),
        patch(f"{MODULE}.fetch_profile_by_id") as mock_fetch,
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path, client=mock_client)
        mock_fetch.assert_not_called()


# --- Cache miss: client unavailable ---


def test_load_or_fetch_profile_by_id_returns_none_when_no_client(
    tmp_path: Path,
) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result_id, result_profile = load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path)
    assert result_id == "~Jane_Smith1"
    assert result_profile is None


def test_load_or_fetch_profile_by_id_uses_provided_client(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path, client=mock_client)
        mock_create.assert_not_called()


def test_load_or_fetch_profile_by_id_creates_client_when_none_provided(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path)
        mock_create.assert_called_once()


# --- Cache miss: successful fetch ---


def test_load_or_fetch_profile_by_id_returns_fetched_profile(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json"),
    ):
        result_id, result_profile = load_or_fetch_profile_by_id(
            "~Jane_Smith1", tmp_path, client=mock_client
        )
    assert result_id == "~Jane_Smith1"
    assert result_profile is mock_profile


def test_load_or_fetch_profile_by_id_saves_profile_to_disk(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    expected_path = tmp_path / "~Jane_Smith1.json"
    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path, client=mock_client)
        mock_save.assert_called_once_with(mock_profile.to_json(), expected_path)


def test_load_or_fetch_profile_by_id_calls_fetch_with_correct_args(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=mock_profile) as mock_fetch,
        patch(f"{MODULE}.save_json"),
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path, client=mock_client)
        mock_fetch.assert_called_once_with("~Jane_Smith1", client=mock_client)


# --- Cache miss: failed fetch ---


def test_load_or_fetch_profile_by_id_returns_none_when_fetch_fails(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    with patch(f"{MODULE}.fetch_profile_by_id", return_value=None):
        _, result_profile = load_or_fetch_profile_by_id(
            "~Jane_Smith1", tmp_path, client=mock_client
        )
    assert result_profile is None


def test_load_or_fetch_profile_by_id_does_not_save_when_fetch_fails(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    with (
        patch(f"{MODULE}.fetch_profile_by_id", return_value=None),
        patch(f"{MODULE}.save_json") as mock_save,
    ):
        load_or_fetch_profile_by_id("~Jane_Smith1", tmp_path, client=mock_client)
        mock_save.assert_not_called()


#############################################
#         Tests for extract_profiles_by_id        #
#############################################


# --- Client unavailable ---


def test_extract_profiles_by_id_returns_none_values_when_no_client(
    tmp_path: Path,
) -> None:
    with patch(f"{MODULE}.create_client", return_value=None):
        result = extract_profiles_by_id(["~Jane_Smith1"], tmp_path)
    assert result == {"~Jane_Smith1": None}


def test_extract_profiles_by_id_uses_provided_client(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client") as mock_create,
        patch(f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", mock_profile)),
    ):
        extract_profiles_by_id(["~Jane_Smith1"], tmp_path, client=mock_client)
        mock_create.assert_not_called()


def test_extract_profiles_by_id_creates_client_when_none_provided(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with (
        patch(f"{MODULE}.create_client", return_value=mock_client) as mock_create,
        patch(f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", mock_profile)),
    ):
        extract_profiles_by_id(["~Jane_Smith1"], tmp_path)
        mock_create.assert_called_once()


# --- Directory creation ---


def test_extract_profiles_by_id_creates_directory_if_not_exists(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    path = tmp_path / "~Jane_Smith1"
    assert not path.exists()
    with patch(
        f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", mock_profile)
    ):
        extract_profiles_by_id(["~Jane_Smith1"], path, client=mock_client)
    assert path.exists()


# --- Deduplication ---


def test_extract_profiles_by_id_deduplicates_profile_ids(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with patch(
        f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", mock_profile)
    ) as mock_fetch:
        extract_profiles_by_id(["~Jane_Smith1", "~Jane_Smith1"], tmp_path, client=mock_client)
        assert mock_fetch.call_count == 1


# --- Results ---


def test_extract_profiles_by_id_returns_profiles(
    tmp_path: Path,
    mock_client: Mock,
    mock_profile: Mock,
) -> None:
    with patch(
        f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", mock_profile)
    ):
        result = extract_profiles_by_id(["~Jane_Smith1"], tmp_path, client=mock_client)
    assert result["~Jane_Smith1"] is mock_profile


def test_extract_profiles_by_id_returns_none_for_failed_fetch(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    with patch(f"{MODULE}.load_or_fetch_profile_by_id", return_value=("~Jane_Smith1", None)):
        result = extract_profiles_by_id(["~Jane_Smith1"], tmp_path, client=mock_client)
    assert result["~Jane_Smith1"] is None


def test_extract_profiles_by_id_handles_multiple_ids(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    profile_a = Mock()
    profile_b = Mock()

    def side_effect(
        profile_id: str,
        path: Path,  # noqa: ARG001
        client: OpenReviewClient,  # noqa: ARG001
    ) -> tuple[str, Profile]:
        return (profile_id, profile_a if profile_id == "~Jane_Smith1" else profile_b)

    with patch(f"{MODULE}.load_or_fetch_profile_by_id", side_effect=side_effect):
        result = extract_profiles_by_id(
            ["~Jane_Smith1", "~John_Doe1"], tmp_path, client=mock_client
        )
    assert result["~Jane_Smith1"] is profile_a
    assert result["~John_Doe1"] is profile_b


def test_extract_profiles_by_id_empty_input(
    tmp_path: Path,
    mock_client: Mock,
) -> None:
    result = extract_profiles_by_id([], tmp_path, client=mock_client)
    assert result == {}
