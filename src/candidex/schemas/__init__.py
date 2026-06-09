r"""Contain schemas."""

from __future__ import annotations

__all__ = ["AuthorAffiliation", "AuthorExtraction", "PaperAffiliations", "PaperAuthorExtraction"]

from candidex.schemas.affiliation import AuthorAffiliation, PaperAffiliations
from candidex.schemas.author import AuthorExtraction, PaperAuthorExtraction
