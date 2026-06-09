from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import requests

from candidex.pdf.download import download_pdf

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "candidex.pdf.download"


# --- Fixtures ---


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    return tmp_path / "papers" / "paper.pdf"


@pytest.fixture
def mock_response() -> Mock:
    response = Mock()
    response.status_code = 200
    response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
    response.raise_for_status = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    return response


@pytest.fixture
def mock_session(mock_response: Mock) -> Mock:
    session = Mock()
    session.get.return_value = mock_response
    return session


##################################
#     Tests for download_pdf     #
##################################

# --- Skip if already exists ---


def test_download_pdf_returns_true_when_file_exists(pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"existing content")
    assert download_pdf("https://example.com/paper.pdf", pdf_path) is True


def test_download_pdf_does_not_call_session_when_file_exists(
    pdf_path: Path, mock_session: Mock
) -> None:
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"existing content")
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    mock_session.get.assert_not_called()


# --- Directory creation ---


def test_download_pdf_creates_parent_directories(pdf_path: Path, mock_session: Mock) -> None:
    assert not pdf_path.parent.exists()
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert pdf_path.parent.exists()


# --- Successful download ---


def test_download_pdf_returns_true_on_success(pdf_path: Path, mock_session: Mock) -> None:
    assert download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session) is True


def test_download_pdf_writes_content_to_disk(pdf_path: Path, mock_session: Mock) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert pdf_path.read_bytes() == b"chunk1chunk2chunk3"


def test_download_pdf_calls_session_get_with_correct_url(
    pdf_path: Path, mock_session: Mock
) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    args, _kwargs = mock_session.get.call_args
    assert args[0] == "https://example.com/paper.pdf"


def test_download_pdf_calls_session_get_with_stream_true(
    pdf_path: Path, mock_session: Mock
) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["stream"] is True


def test_download_pdf_passes_timeout_to_session(pdf_path: Path, mock_session: Mock) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, timeout=60, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 60


def test_download_pdf_uses_default_timeout(pdf_path: Path, mock_session: Mock) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 30


def test_download_pdf_passes_chunk_size_to_iter_content(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, chunk_size=4096, session=mock_session)
    mock_response.iter_content.assert_called_once_with(chunk_size=4096)


def test_download_pdf_uses_default_chunk_size(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    mock_response.iter_content.assert_called_once_with(chunk_size=8192)


# --- Atomic write ---


def test_download_pdf_does_not_leave_tmp_file_on_success(
    pdf_path: Path, mock_session: Mock
) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert not pdf_path.with_suffix(".tmp").exists()


def test_download_pdf_does_not_leave_tmp_file_on_failure(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert not pdf_path.with_suffix(".tmp").exists()


# --- Session management ---


def test_download_pdf_uses_provided_session(pdf_path: Path, mock_session: Mock) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
        mock_session_cls.assert_not_called()


def test_download_pdf_creates_session_when_none_provided(
    pdf_path: Path, mock_response: Mock
) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        download_pdf("https://example.com/paper.pdf", pdf_path)
        mock_session_cls.assert_called_once()


def test_download_pdf_closes_session_when_created_internally(
    pdf_path: Path, mock_response: Mock
) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        download_pdf("https://example.com/paper.pdf", pdf_path)
        mock_session_instance.close.assert_called_once()


def test_download_pdf_does_not_close_provided_session(pdf_path: Path, mock_session: Mock) -> None:
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    mock_session.close.assert_not_called()


def test_download_pdf_closes_session_on_network_error(pdf_path: Path) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = requests.exceptions.ConnectionError("unreachable")
        mock_session_cls.return_value = mock_session_instance
        download_pdf("https://example.com/paper.pdf", pdf_path)
        mock_session_instance.close.assert_called_once()


# --- HTTP errors ---


def test_download_pdf_returns_false_on_http_error(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    assert download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session) is False


def test_download_pdf_returns_false_on_connection_error(pdf_path: Path, mock_session: Mock) -> None:
    mock_session.get.side_effect = requests.exceptions.ConnectionError("unreachable")
    assert download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session) is False


def test_download_pdf_returns_false_on_timeout(pdf_path: Path, mock_session: Mock) -> None:
    mock_session.get.side_effect = requests.exceptions.ConnectTimeout("timed out")
    assert download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session) is False


def test_download_pdf_does_not_write_file_on_http_error(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
    download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert not pdf_path.exists()


def test_download_pdf_removes_tmp_file_on_unexpected_error(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    mock_response.iter_content.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)
    assert not pdf_path.with_suffix(".tmp").exists()
    assert not pdf_path.exists()


def test_download_pdf_reraises_unexpected_error(
    pdf_path: Path, mock_session: Mock, mock_response: Mock
) -> None:
    mock_response.iter_content.side_effect = RuntimeError("unexpected")
    with pytest.raises(RuntimeError, match="unexpected"):
        download_pdf("https://example.com/paper.pdf", pdf_path, session=mock_session)


# --- Retry configuration ---


def test_download_pdf_mounts_retry_adapter_when_no_session(
    pdf_path: Path, mock_response: Mock
) -> None:
    with patch(f"{MODULE}.requests.Session") as mock_session_cls:
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_cls.return_value = mock_session_instance
        download_pdf("https://example.com/paper.pdf", pdf_path, max_retries=5)
        assert mock_session_instance.mount.call_count == 2
        mounted_prefixes = [c[0][0] for c in mock_session_instance.mount.call_args_list]
        assert "https://" in mounted_prefixes
        assert "http://" in mounted_prefixes
