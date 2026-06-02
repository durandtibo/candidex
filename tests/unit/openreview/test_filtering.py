from __future__ import annotations

from unittest.mock import Mock

import pytest

from candidex.openreview import filter_profiles_by_affiliation, filter_profiles_by_email


def make_profile(
    institutions: list[tuple[str, str]],
    emails: list[str] | None = None,
) -> Mock:
    """Create a mock profile with a list of (name, domain) institution
    tuples and optional confirmed email addresses."""
    profile = Mock()
    profile.content = {
        "history": [
            {"institution": {"name": name, "domain": domain}} for name, domain in institutions
        ],
        "emails": emails or [],
    }
    return profile


@pytest.fixture
def mit_profile() -> Mock:
    return make_profile(
        institutions=[("MIT CSAIL", "mit.edu")],
        emails=["jane@csail.mit.edu"],
    )


@pytest.fixture
def stanford_profile() -> Mock:
    return make_profile(
        institutions=[("Stanford University", "stanford.edu")],
        emails=["john@stanford.edu"],
    )


@pytest.fixture
def empty_profile() -> Mock:
    return make_profile(institutions=[], emails=[])


####################################################
#     Tests for filter_profiles_by_affiliation     #
####################################################


def test_filter_profiles_by_affiliation_returns_empty_for_empty_profiles() -> None:
    assert filter_profiles_by_affiliation([], affiliation="MIT") == []


def test_filter_profiles_by_affiliation_returns_empty_when_no_match(
    stanford_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([stanford_profile], affiliation="MIT") == []


def test_filter_profiles_by_affiliation_handles_empty_history(
    empty_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([empty_profile], affiliation="MIT") == []


@pytest.mark.parametrize(
    "affiliation",
    [
        pytest.param("MIT CSAIL", id="exact_match"),
        pytest.param("MIT", id="partial_match"),
        pytest.param("mit csail", id="case_insensitive"),
    ],
)
def test_filter_profiles_by_affiliation_matches_affiliation(
    affiliation: str,
    mit_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([mit_profile], affiliation=affiliation) == [mit_profile]


def test_filter_profiles_by_affiliation_filters_from_multiple_profiles(
    mit_profile: Mock,
    stanford_profile: Mock,
) -> None:
    result = filter_profiles_by_affiliation([mit_profile, stanford_profile], affiliation="MIT")
    assert result == [mit_profile]


##############################################
#     Tests for filter_profiles_by_email     #
##############################################


def test_filter_profiles_by_email_returns_empty_for_empty_profiles() -> None:
    assert filter_profiles_by_email([], email="jane@mit.edu") == []


def test_filter_profiles_by_email_returns_empty_when_no_match(
    stanford_profile: Mock,
) -> None:
    assert filter_profiles_by_email([stanford_profile], email="jane@mit.edu") == []


def test_filter_profiles_by_email_handles_empty_history_and_emails(
    empty_profile: Mock,
) -> None:
    assert filter_profiles_by_email([empty_profile], email="jane@mit.edu") == []


@pytest.mark.parametrize(
    "email",
    [
        pytest.param("jane@csail.mit.edu", id="matches_confirmed_email"),
        pytest.param("other@mit.edu", id="matches_confirmed_email_domain"),
    ],
)
def test_filter_profiles_by_email_matches_via_profile_email(
    email: str,
    mit_profile: Mock,
) -> None:
    assert filter_profiles_by_email([mit_profile], email=email) == [mit_profile]


@pytest.mark.parametrize(
    "email",
    [
        pytest.param("other@mit.edu", id="matches_institution_domain"),
        pytest.param("other@csail.mit.edu", id="matches_institution_subdomain"),
    ],
)
def test_filter_profiles_by_email_matches_via_institution_domain(email: str) -> None:
    profile = make_profile(institutions=[("MIT CSAIL", "mit.edu")], emails=[])
    assert filter_profiles_by_email([profile], email=email) == [profile]


def test_filter_profiles_by_email_excludes_wrong_domain(mit_profile: Mock) -> None:
    assert filter_profiles_by_email([mit_profile], email="jane@google.com") == []


def test_filter_profiles_by_email_filters_from_multiple_profiles(
    mit_profile: Mock,
    stanford_profile: Mock,
) -> None:
    result = filter_profiles_by_email([mit_profile, stanford_profile], email="jane@mit.edu")
    assert result == [mit_profile]
