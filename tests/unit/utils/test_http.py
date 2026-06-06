from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from candidex.utils.http import HEADERS, fetch_html

MODULE = "candidex.utils.http"


# --- Fixtures ---


@pytest.fixture
def mock_response() -> Mock:
    response = Mock()
    response.status_code = 200
    response.content = b"<html><body>Hello</body></html>"
    response.text = "<html><body>Hello</body></html>"
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def mock_session(mock_response: Mock) -> Mock:
    session = Mock()
    session.get.return_value = mock_response
    return session


################################
#     Tests for fetch_html     #
################################

# --- Successful fetch ---


def test_fetch_html_returns_html_string(mock_session: Mock) -> None:
    result = fetch_html("https://example.com", session=mock_session)
    assert result == "<html><body>Hello</body></html>"


def test_fetch_html_calls_session_get_with_correct_url(mock_session: Mock) -> None:
    fetch_html("https://example.com", session=mock_session)
    mock_session.get.assert_called_once()
    call_kwargs = mock_session.get.call_args
    assert call_kwargs[0][0] == "https://example.com"


def test_fetch_html_calls_raise_for_status(mock_session: Mock, mock_response: Mock) -> None:
    fetch_html("https://example.com", session=mock_session)
    mock_response.raise_for_status.assert_called_once()


# --- Headers ---


def test_fetch_html_uses_default_headers_when_none_provided(
    mock_session: Mock,
) -> None:
    fetch_html("https://example.com", session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["headers"] == HEADERS


def test_fetch_html_uses_provided_headers(mock_session: Mock) -> None:
    custom_headers = {"User-Agent": "MyBot/1.0"}
    fetch_html("https://example.com", headers=custom_headers, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["headers"] == custom_headers


def test_fetch_html_uses_empty_headers_when_empty_dict_provided(
    mock_session: Mock,
) -> None:
    fetch_html("https://example.com", headers={}, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["headers"] == {}


# --- Timeout ---


def test_fetch_html_passes_timeout_to_session(mock_session: Mock) -> None:
    fetch_html("https://example.com", timeout=60, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 60


def test_fetch_html_default_timeout_is_30(mock_session: Mock) -> None:
    fetch_html("https://example.com", session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 30


# --- Session management ---


def test_fetch_html_uses_provided_session(mock_session: Mock) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        fetch_html("https://example.com", session=mock_session)
        mock_session_cls.assert_not_called()


def test_fetch_html_creates_session_when_none_provided(mock_response: Mock) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        fetch_html("https://example.com")
        mock_session_cls.assert_called_once()


def test_fetch_html_closes_session_when_created_internally(mock_response: Mock) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        fetch_html("https://example.com")
        mock_session_instance.close.assert_called_once()


def test_fetch_html_does_not_close_provided_session(mock_session: Mock) -> None:
    fetch_html("https://example.com", session=mock_session)
    mock_session.close.assert_not_called()


def test_fetch_html_closes_session_even_on_error() -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = requests.exceptions.ConnectionError("unreachable")
        mock_session_cls.return_value = mock_session_instance
        with pytest.raises(requests.exceptions.ConnectionError, match="unreachable"):
            fetch_html("https://example.com")
        mock_session_instance.close.assert_called_once()


# --- Retry configuration ---


def test_fetch_html_mounts_retry_adapter(mock_response: Mock) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        fetch_html("https://example.com", max_retries=5)
        assert mock_session_instance.mount.call_count == 2
        calls = [call[0][0] for call in mock_session_instance.mount.call_args_list]
        assert "https://" in calls
        assert "http://" in calls


# --- HTTP errors ---


def test_fetch_html_raises_on_http_error(mock_session: Mock, mock_response: Mock) -> None:
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    with pytest.raises(requests.exceptions.HTTPError):
        fetch_html("https://example.com", session=mock_session)


def test_fetch_html_raises_on_connection_error(mock_session: Mock) -> None:
    mock_session.get.side_effect = requests.exceptions.ConnectionError("unreachable")
    with pytest.raises(requests.exceptions.ConnectionError):
        fetch_html("https://example.com", session=mock_session)


def test_fetch_html_raises_on_timeout(mock_session: Mock) -> None:
    mock_session.get.side_effect = requests.exceptions.ConnectTimeout("timed out")
    with pytest.raises(requests.exceptions.ConnectTimeout):
        fetch_html("https://example.com", session=mock_session)
