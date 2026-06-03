r"""Contain utilities for authors."""

from __future__ import annotations

__all__ = [
    "Author",
    "add_profile_ids_to_dataframe",
    "authors_to_dataframe",
    "deduplicate_authors",
    "sort_authors",
]

from candidex.author.author import Author
from candidex.author.conversion import authors_to_dataframe
from candidex.author.deduplication import deduplicate_authors
from candidex.author.frame import add_profile_ids_to_dataframe
from candidex.author.sorting import sort_authors
