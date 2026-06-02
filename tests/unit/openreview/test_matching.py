from __future__ import annotations

import pytest

from candidex.openreview import do_affiliations_match

###########################################
#     Tests for do_affiliations_match     #
###########################################


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param("MIT", "MIT CSAIL, Cambridge, MA, USA", True, id="partial_match"),
        pytest.param("MIT CSAIL, Cambridge, MA, USA", "MIT", True, id="partial_match_reversed"),
        pytest.param("mit", "MIT CSAIL", True, id="case_insensitive"),
        pytest.param("MIT CSAIL", "mit csail", True, id="case_insensitive_both"),
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
        pytest.param("Stanford University", "MIT CSAIL", False, id="no_match"),
        pytest.param("Google", "DeepMind", False, id="no_match_companies"),
        pytest.param("MIT", "MIT", True, id="identical"),
        pytest.param("", "MIT", True, id="empty_string_matches_anything"),
        pytest.param(" MIT", "MIT ", True, id="strip_spaces"),
    ],
)
def test_do_affiliations_match(a: str, b: str, expected: bool) -> None:
    assert do_affiliations_match(a, b) == expected
