from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest

from candidex.openreview.client import create_openreview_client

MODULE = "candidex.openreview.client"


##############################################
#     Tests for create_openreview_client     #
##############################################

# --- Missing credentials ---


@pytest.mark.parametrize(
    ("env_vars", "username", "password"),
    [
        pytest.param(
            {},
            None,
            None,
            id="no_credentials_at_all",
        ),
        pytest.param(
            {"OPENREVIEW_USERNAME": "user@example.com"},
            None,
            None,
            id="only_username_env_set",
        ),
        pytest.param(
            {"OPENREVIEW_PASSWORD": "secret"},
            None,
            None,
            id="only_password_env_set",
        ),
    ],
)
def test_create_openreview_client_returns_none_when_credentials_missing(
    env_vars: dict,
    username: str | None,
    password: str | None,
) -> None:
    with patch.dict(os.environ, env_vars, clear=True):
        client = create_openreview_client(username=username, password=password)
    assert client is None


# --- Credential priority ---


@pytest.mark.parametrize(
    ("arg_username", "arg_password", "expected_username", "expected_password"),
    [
        pytest.param(
            None,
            None,
            "env@example.com",
            "env_secret",
            id="uses_env_vars_when_no_args",
        ),
        pytest.param(
            "arg@example.com",
            "arg_secret",
            "arg@example.com",
            "arg_secret",
            id="args_take_priority_over_env",
        ),
        pytest.param(
            "arg@example.com",
            None,
            "arg@example.com",
            "env_secret",
            id="arg_username_overrides_env_only",
        ),
        pytest.param(
            None,
            "arg_secret",
            "env@example.com",
            "arg_secret",
            id="arg_password_overrides_env_only",
        ),
    ],
)
def test_create_openreview_client_credential_priority(
    arg_username: str | None,
    arg_password: str | None,
    expected_username: str,
    expected_password: str,
) -> None:
    mock_client = Mock()
    with (
        patch.dict(
            os.environ,
            {"OPENREVIEW_USERNAME": "env@example.com", "OPENREVIEW_PASSWORD": "env_secret"},
            clear=True,
        ),
        patch(f"{MODULE}.OpenReviewClient", return_value=mock_client) as mock_ctor,
    ):
        create_openreview_client(username=arg_username, password=arg_password)
        mock_ctor.assert_called_once_with(
            baseurl="https://api2.openreview.net",
            username=expected_username,
            password=expected_password,
        )


# --- Successful authentication ---


def test_create_openreview_client_returns_client_on_success() -> None:
    mock_client = Mock()
    with (
        patch.dict(
            os.environ,
            {"OPENREVIEW_USERNAME": "user@example.com", "OPENREVIEW_PASSWORD": "secret"},
            clear=True,
        ),
        patch(f"{MODULE}.OpenReviewClient", return_value=mock_client),
    ):
        client = create_openreview_client()
    assert client is mock_client


# --- Failed authentication ---


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(Exception("Connection error"), id="generic_exception"),
        pytest.param(ConnectionError("Network unreachable"), id="connection_error"),
        pytest.param(ValueError("Invalid credentials"), id="value_error"),
    ],
)
def test_create_openreview_client_returns_none_on_authentication_failure(
    exception: Exception,
) -> None:
    with (
        patch.dict(
            os.environ,
            {"OPENREVIEW_USERNAME": "user@example.com", "OPENREVIEW_PASSWORD": "secret"},
            clear=True,
        ),
        patch(f"{MODULE}.OpenReviewClient", side_effect=exception),
    ):
        client = create_openreview_client()
    assert client is None
