from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest
from requests import Session

from candidex.paper import Paper, download_pdfs

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "candidex.paper.download"


# --- Fixtures ---


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    return tmp_path / "pdfs"


@pytest.fixture
def paper_a() -> Paper:
    return Paper.from_raw(
        title="Attention Is All You Need",
        authors=["Jane Smith"],
        venue="NeurIPS",
        year=2017,
        pdf_url="https://arxiv.org/pdf/1706.03762",
    )


@pytest.fixture
def paper_b() -> Paper:
    return Paper.from_raw(
        title="BERT",
        authors=["John Doe"],
        venue="NAACL",
        year=2019,
        pdf_url="https://arxiv.org/pdf/1810.04805",
    )


@pytest.fixture
def paper_no_url() -> Paper:
    return Paper.from_raw(
        title="No URL Paper",
        authors=["Alice Brown"],
        venue="CVPR",
        year=2024,
        pdf_url=None,
    )


###################################
#     Tests for download_pdfs     #
###################################

# --- Directory creation ---


def test_download_pdfs_creates_pdf_dir(pdf_dir: Path, paper_a: Paper) -> None:
    assert not pdf_dir.exists()
    with patch(f"{MODULE}.download_pdf", return_value=True):
        download_pdfs([paper_a], pdf_dir=pdf_dir)
    assert pdf_dir.exists()


# --- Empty input ---


def test_download_pdfs_empty_papers_returns_empty_dict(pdf_dir: Path) -> None:
    result = download_pdfs([], pdf_dir=pdf_dir)
    assert result == {}


# --- Papers with no URL ---


def test_download_pdfs_skips_paper_with_no_url(
    pdf_dir: Path,
    paper_no_url: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf") as mock_download:
        result = download_pdfs([paper_no_url], pdf_dir=pdf_dir)
        mock_download.assert_not_called()
    assert result[paper_no_url] is False


# --- Results ---


def test_download_pdfs_returns_true_for_successful_download(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True):
        result = download_pdfs([paper_a], pdf_dir=pdf_dir)
    assert result[paper_a] is True


def test_download_pdfs_returns_false_for_failed_download(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=False):
        result = download_pdfs([paper_a], pdf_dir=pdf_dir)
    assert result[paper_a] is False


def test_download_pdfs_returns_result_for_all_papers(
    pdf_dir: Path,
    paper_a: Paper,
    paper_b: Paper,
    paper_no_url: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True):
        result = download_pdfs([paper_a, paper_b, paper_no_url], pdf_dir=pdf_dir)
    assert set(result.keys()) == {paper_a, paper_b, paper_no_url}


def test_download_pdfs_mixed_results(
    pdf_dir: Path,
    paper_a: Paper,
    paper_b: Paper,
    paper_no_url: Paper,
) -> None:
    def side_effect(
        url: str,
        pdf_path: Path,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> bool:
        return url == paper_a.pdf_url

    with patch(f"{MODULE}.download_pdf", side_effect=side_effect):
        result = download_pdfs([paper_a, paper_b, paper_no_url], pdf_dir=pdf_dir)

    assert result[paper_a] is True
    assert result[paper_b] is False
    assert result[paper_no_url] is False


# --- PDF path construction ---


def test_download_pdfs_uses_paper_hash_as_filename(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True) as mock_download:
        download_pdfs([paper_a], pdf_dir=pdf_dir)
        _, kwargs = mock_download.call_args
        assert kwargs["pdf_path"] == pdf_dir / f"{paper_a.hash()}.pdf"


def test_download_pdfs_passes_correct_url(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True) as mock_download:
        download_pdfs([paper_a], pdf_dir=pdf_dir)
        _, kwargs = mock_download.call_args
        assert kwargs["url"] == paper_a.pdf_url


# --- Parameters passed through ---


def test_download_pdfs_passes_timeout(pdf_dir: Path, paper_a: Paper) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True) as mock_download:
        download_pdfs([paper_a], pdf_dir=pdf_dir, timeout=60)
        _, kwargs = mock_download.call_args
        assert kwargs["timeout"] == 60


def test_download_pdfs_passes_chunk_size(pdf_dir: Path, paper_a: Paper) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True) as mock_download:
        download_pdfs([paper_a], pdf_dir=pdf_dir, chunk_size=4096)
        _, kwargs = mock_download.call_args
        assert kwargs["chunk_size"] == 4096


# --- Shared session ---


def test_download_pdfs_passes_same_session_to_all_downloads(
    pdf_dir: Path,
    paper_a: Paper,
    paper_b: Paper,
) -> None:
    with patch(f"{MODULE}.download_pdf", return_value=True) as mock_download:
        download_pdfs([paper_a, paper_b], pdf_dir=pdf_dir)
        sessions = [call.kwargs["session"] for call in mock_download.call_args_list]
        assert len({id(s) for s in sessions}) == 1


def test_download_pdfs_creates_session_via_create_session(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    with (
        patch(f"{MODULE}.create_session", return_value=Mock(spec=Session)) as mock_create,
        patch(f"{MODULE}.download_pdf", return_value=True),
    ):
        download_pdfs([paper_a], pdf_dir=pdf_dir, max_retries=5)
        mock_create.assert_called_once_with(max_retries=5)


def test_download_pdfs_closes_session_after_completion(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    mock_session = Mock(spec=Session)
    with (
        patch(f"{MODULE}.create_session", return_value=mock_session),
        patch(f"{MODULE}.download_pdf", return_value=True),
    ):
        download_pdfs([paper_a], pdf_dir=pdf_dir)
        mock_session.close.assert_called_once()


def test_download_pdfs_closes_session_on_unexpected_error(
    pdf_dir: Path,
    paper_a: Paper,
) -> None:
    mock_session = Mock(spec=Session)
    with (
        patch(f"{MODULE}.create_session", return_value=mock_session),
        patch(f"{MODULE}.download_pdf", side_effect=RuntimeError("unexpected")),
    ):
        with pytest.raises(RuntimeError):
            download_pdfs([paper_a], pdf_dir=pdf_dir)
        mock_session.close.assert_called_once()
