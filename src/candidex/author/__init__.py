r"""Contain utilities for authors."""

from __future__ import annotations

__all__ = ["Author", "deduplicate_authors", "sort_authors"]

from candidex.author.author import Author
from candidex.author.deduplication import deduplicate_authors
from candidex.author.sorting import sort_authors
