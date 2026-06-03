r"""Contain utility for working with progress bars."""

from __future__ import annotations

__all__ = ["make_progressbar"]

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


def make_progressbar(*, transient: bool = False) -> Progress:
    """Create a standardised Rich progress bar for use across the
    codebase.

    Builds a `Progress` instance with a consistent column layout:
    - A spinner indicating the task is active.
    - Description text.
    - A bar showing percentage complete.
    - The percentage as a number (e.g. '42%').
    - An M-of-N counter (e.g. '42/100').
    - Elapsed time since the task started.
    - Estimated time remaining.

    Using this factory ensures all progress bars in the codebase share the
    same appearance. Use as a context manager to start and stop rendering.

    Args:
        transient: If True, the progress bar is cleared from the terminal
                   when the context manager exits. Useful for short-lived
                   tasks where the bar would clutter the output. Defaults
                   to False.

    Returns:
        A configured `Progress` instance, ready to use as a context manager.

    Example:
        ```pycon
        >>> from candidex.utils.progressbar import make_progressbar
        >>> rows = list(range(10))
        >>> with make_progressbar() as progress:
        ...     task = progress.add_task("Processing papers", total=len(rows))
        ...     for row in rows:
        ...         print(row)
        ...         progress.advance(task)
        ...

        ```
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=transient,
    )
