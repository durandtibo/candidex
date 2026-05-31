r"""Contain utilities for working with progress bars."""

from __future__ import annotations

__all__ = ["make_progressbar"]

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


def make_progressbar() -> Progress:
    """Create a standardised Rich progress bar for use across the
    codebase.

    Builds a `Progress` instance with a consistent column layout:
    - Description text on the left.
    - A bar showing percentage complete.
    - The percentage as a number (e.g. '42%').
    - An M-of-N counter (e.g. '42/100').
    - Elapsed time since the task started.

    Using this factory ensures all progress bars in the codebase share the
    same appearance. Use as a context manager to start and stop rendering:

    Returns:
        A configured `Progress` instance, ready to use as a context manager.

    Example:
        >>> with make_progressbar() as progress:
        ...     task = progress.add_task("Processing papers", total=len(rows))
        ...     for row in rows:
        ...         process(row)
        ...         progress.advance(task)
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
