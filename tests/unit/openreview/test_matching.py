from __future__ import annotations

import pytest

from candidex.openreview import do_affiliations_match, does_email_match_domain

###########################################
#     Tests for do_affiliations_match     #
###########################################


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # --- Substring matching ---
        pytest.param("MIT", "MIT CSAIL, Cambridge, MA, USA", True, id="partial_match"),
        pytest.param("MIT CSAIL, Cambridge, MA, USA", "MIT", True, id="partial_match_reversed"),
        pytest.param("MIT", "MIT", True, id="identical"),
        pytest.param("Stanford University", "MIT CSAIL", False, id="no_match"),
        pytest.param("Google", "DeepMind", False, id="no_match_companies"),
        pytest.param("", "MIT", True, id="empty_string_matches_anything"),
        # --- Case insensitive ---
        pytest.param("mit", "MIT CSAIL", True, id="case_insensitive_lower"),
        pytest.param("MIT CSAIL", "mit csail", True, id="case_insensitive_both"),
        # --- Spacing differences ---
        pytest.param(
            "EastChinaNormalUniversity",
            "East China Normal University",
            True,
            id="spacing_difference",
        ),
        pytest.param(
            "East China Normal University",
            "EastChinaNormalUniversity",
            True,
            id="spacing_difference_reversed",
        ),
        pytest.param(
            "University of Science and Technology of China",
            "UniversityofScienceandTechnologyofChina",
            True,
            id="spacing_difference_long",
        ),
        # --- Unicode normalization ---
        pytest.param(
            "Université de Montréal",
            "Universite de Montreal",
            True,
            id="unicode_french_accents",
        ),
        pytest.param(
            "Universite de Montreal",
            "Université de Montréal",
            True,
            id="unicode_french_accents_reversed",
        ),
        pytest.param(
            "Technische Universität München",
            "Technische Universitat Munchen",
            True,
            id="unicode_german_umlaut",
        ),
        pytest.param(
            "École Polytechnique Fédérale de Lausanne",
            "Ecole Polytechnique Federale de Lausanne",
            True,
            id="unicode_epfl_accents",
        ),
        pytest.param(
            "Università di Bologna",
            "Universita di Bologna",
            True,
            id="unicode_italian_accents",
        ),
        # --- Whitespace stripping ---
        pytest.param("  MIT  ", "MIT CSAIL", True, id="leading_trailing_spaces"),
        pytest.param("MIT", "  MIT CSAIL  ", True, id="leading_trailing_spaces_b"),
    ],
)
def test_do_affiliations_match(a: str, b: str, expected: bool) -> None:
    assert do_affiliations_match(a, b) == expected


#############################################
#     Tests for does_email_match_domain     #
#############################################


@pytest.mark.parametrize(
    ("email", "domain", "expected"),
    [
        pytest.param("jane.smith@mit.edu", "mit.edu", True, id="exact_match"),
        pytest.param("jane.smith@mit.edu", "@mit.edu", True, id="domain_with_at_prefix"),
        pytest.param("jane.smith@csail.mit.edu", "mit.edu", True, id="subdomain_match"),
        pytest.param(
            "jane.smith@csail.mit.edu", "@mit.edu", True, id="subdomain_match_with_at_prefix"
        ),
        pytest.param("jane.smith@MIT.EDU", "mit.edu", True, id="case_insensitive_email"),
        pytest.param("jane.smith@mit.edu", "MIT.EDU", True, id="case_insensitive_domain"),
        pytest.param(
            "jane.smith@mit.edu", "@MIT.EDU", True, id="case_insensitive_domain_with_at_prefix"
        ),
        pytest.param("jane.smith@stanford.edu", "mit.edu", False, id="no_match"),
        pytest.param("jane.smith@mit.edu", "stanford.edu", False, id="no_match_reversed"),
        pytest.param("notanemail", "mit.edu", False, id="missing_at_symbol"),
        pytest.param("jane.smith@mit.edu", "edu", True, id="partial_domain_match"),
        pytest.param(None, "mit.edu", False, id="none_email"),
        pytest.param("jane.smith@mit.edu", None, False, id="none_domain"),
        pytest.param(None, None, False, id="both_none"),
        pytest.param("jane.smith@mit.edu", "", True, id="empty_domain_matches_anything"),
    ],
)
def test_does_email_match_domain(email: str | None, domain: str | None, expected: bool) -> None:
    assert does_email_match_domain(email, domain) == expected
