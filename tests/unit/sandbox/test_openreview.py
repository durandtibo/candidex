from __future__ import annotations

import pytest

from candidex.sandbox.openreview import is_openreview_profile_url

###############################################
#     Tests for is_openreview_profile_url     #
###############################################


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param(
            "https://openreview.net/profile?id=~Thibaut_Durand1",
            True,
            id="valid_tilde_id",
        ),
        pytest.param(
            "https://openreview.net/profile?id=%7EThibaut_Durand1",
            True,
            id="valid_encoded_tilde_id",
        ),
        pytest.param(
            "https://openreview.net/forum?id=abc123",
            False,
            id="forum_url",
        ),
        pytest.param(
            "https://openreview.net/profile",
            False,
            id="profile_without_id",
        ),
        pytest.param(
            "https://openreview.net/",
            False,
            id="root_url",
        ),
        pytest.param(
            "https://scholar.google.com/profile?id=~Thibaut_Durand1",
            False,
            id="wrong_domain",
        ),
        pytest.param(
            "",
            False,
            id="empty_string",
        ),
    ],
)
def test_is_openreview_profile_url(url: str, expected: bool) -> None:
    assert is_openreview_profile_url(url) == expected
