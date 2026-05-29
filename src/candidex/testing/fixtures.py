r"""Define pytest mark decorators for conditional test skipping.

``pytest`` is required to use these decorators.
"""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "colorlog_not_available",
]

import pytest

from candidex.utils.imports import is_colorlog_available

colorlog_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_colorlog_available(), reason="Requires colorlog"
)
"""Skip the test if the ``colorlog`` package is not installed."""

colorlog_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_colorlog_available(), reason="Skip if colorlog is available"
)
"""Skip the test if the ``colorlog`` package is installed."""
