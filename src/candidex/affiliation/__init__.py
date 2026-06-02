r"""Contain utilities for affiliation."""

from __future__ import annotations

__all__ = ["deduplicate_authors", "flatten_authors"]

from candidex.affiliation.deduplication import deduplicate_authors
from candidex.affiliation.flatten import flatten_authors
