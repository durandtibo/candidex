r"""Contain utilities for papers."""

from __future__ import annotations

__all__ = ["Paper", "dataframe_to_papers", "papers_to_dataframe"]

from candidex.paper.conversion import dataframe_to_papers, papers_to_dataframe
from candidex.paper.paper import Paper
