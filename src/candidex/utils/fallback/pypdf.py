r"""Contain fallback implementations used when ``pypdf`` dependency is
not available."""

from __future__ import annotations

__all__ = ["pypdf"]

from types import ModuleType
from typing import Any

from candidex.utils.imports import raise_pypdf_missing_error


class FakeClass:
    r"""Fake class that raises an error because pypdf is not installed.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Raises:
        RuntimeError: pypdf is required for this functionality.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        raise_pypdf_missing_error()


# Create a fake pypdf package
pypdf: ModuleType = ModuleType("pypdf")
pypdf.PdfReader = FakeClass
