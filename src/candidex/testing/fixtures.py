r"""Define pytest mark decorators for conditional test skipping.

``pytest`` is required to use these decorators.
"""

from __future__ import annotations

__all__ = [
    "colorlog_available",
    "colorlog_not_available",
    "pdfminer_available",
    "pdfminer_not_available",
    "pdfplumber_available",
    "pdfplumber_not_available",
    "pypdf_available",
    "pypdf_not_available",
    "pypdfium2_available",
    "pypdfium2_not_available",
]

import pytest

from candidex.utils.imports import (
    is_colorlog_available,
    is_pdfminer_available,
    is_pdfplumber_available,
    is_pypdf_available,
    is_pypdfium2_available,
)

colorlog_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_colorlog_available(), reason="Requires colorlog"
)
"""Skip the test if the ``colorlog`` package is not installed."""

colorlog_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_colorlog_available(), reason="Skip if colorlog is available"
)
"""Skip the test if the ``colorlog`` package is installed."""

pdfminer_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pdfminer_available(), reason="Requires pdfminer"
)
"""Skip the test if the ``pdfminer`` package is not installed."""

pdfminer_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pdfminer_available(), reason="Skip if pdfminer is available"
)
"""Skip the test if the ``pdfminer`` package is installed."""

pdfplumber_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pdfplumber_available(), reason="Requires pdfplumber"
)
"""Skip the test if the ``pdfplumber`` package is not installed."""

pdfplumber_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pdfplumber_available(), reason="Skip if pdfplumber is available"
)
"""Skip the test if the ``pdfplumber`` package is installed."""

pypdf_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pypdf_available(), reason="Requires pypdf"
)
"""Skip the test if the ``pypdf`` package is not installed."""

pypdf_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pypdf_available(), reason="Skip if pypdf is available"
)
"""Skip the test if the ``pypdf`` package is installed."""

pypdfium2_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pypdfium2_available(), reason="Requires pypdfium2"
)
"""Skip the test if the ``pypdfium2`` package is not installed."""

pypdfium2_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pypdfium2_available(), reason="Skip if pypdfium2 is available"
)
"""Skip the test if the ``pypdfium2`` package is installed."""
