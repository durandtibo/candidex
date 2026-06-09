r"""Contain schemas."""

from __future__ import annotations

__all__ = ["AuthorAffiliation", "AuthorExtraction", "PaperAffiliations"]

from candidex.schemas.affiliation import AuthorAffiliation, PaperAffiliations
from candidex.schemas.author import AuthorExtraction
