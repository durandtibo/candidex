from __future__ import annotations

from unittest.mock import Mock

import pytest

from candidex.openreview import filter_profiles_by_affiliation


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

# --- Empty inputs ---


def test_filter_profiles_by_affiliation_returns_empty_for_empty_profiles() -> None:
    assert filter_profiles_by_affiliation([], affiliation="MIT") == []


def test_filter_profiles_by_affiliation_returns_empty_when_no_match(
    stanford_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([stanford_profile], affiliation="MIT") == []


# --- Affiliation matching ---


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


def test_filter_profiles_by_affiliation_filters_correctly_from_multiple_profiles(
    mit_profile: Mock,
    stanford_profile: Mock,
) -> None:
    result = filter_profiles_by_affiliation([mit_profile, stanford_profile], affiliation="MIT")
    assert result == [mit_profile]


# --- Email matching via confirmed profile emails ---


@pytest.mark.parametrize(
    "email",
    [
        pytest.param("jane@csail.mit.edu", id="matches_confirmed_email_exact"),
        pytest.param("other@mit.edu", id="matches_confirmed_email_domain"),
    ],
)
def test_filter_profiles_by_affiliation_matches_via_profile_email(
    email: str,
    mit_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([mit_profile], affiliation="MIT", email=email) == [
        mit_profile
    ]


# --- Email matching via institution domain ---


@pytest.mark.parametrize(
    "email",
    [
        pytest.param("other@mit.edu", id="matches_institution_domain_exact"),
        pytest.param("other@csail.mit.edu", id="matches_institution_domain_subdomain"),
    ],
)
def test_filter_profiles_by_affiliation_matches_via_institution_domain(
    email: str,
) -> None:
    profile = make_profile(
        institutions=[("MIT CSAIL", "mit.edu")],
        emails=[],
    )
    assert filter_profiles_by_affiliation([profile], affiliation="MIT", email=email) == [profile]


# --- Email exclusion ---


def test_filter_profiles_by_affiliation_excludes_when_email_does_not_match(
    mit_profile: Mock,
) -> None:
    result = filter_profiles_by_affiliation(
        [mit_profile], affiliation="MIT", email="jane@google.com"
    )
    assert result == []


# --- No email check ---


def test_filter_profiles_by_affiliation_skips_email_check_when_none(
    mit_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([mit_profile], affiliation="MIT", email=None) == [
        mit_profile
    ]


# --- Empty profile history ---


def test_filter_profiles_by_affiliation_handles_empty_history(
    empty_profile: Mock,
) -> None:
    assert filter_profiles_by_affiliation([empty_profile], affiliation="MIT") == []
