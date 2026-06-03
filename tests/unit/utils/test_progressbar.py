from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from candidex.utils.progressbar import make_progressbar

######################################
#     Tests for make_progressbar     #
######################################


def test_make_progressbar_returns_progress_instance() -> None:
    assert isinstance(make_progressbar(), Progress)


def test_make_progressbar_default_transient_is_false() -> None:
    assert make_progressbar().live.transient is False


def test_make_progressbar_transient_true() -> None:
    assert make_progressbar(transient=True).live.transient is True


def test_make_progressbar_column_types() -> None:
    progress = make_progressbar()
    column_types = [type(c) for c in progress.columns]
    assert column_types == [
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    ]


def test_make_progressbar_each_call_returns_new_instance() -> None:
    assert make_progressbar() is not make_progressbar()
