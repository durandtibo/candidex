r"""Contain openreview functionalities."""

from __future__ import annotations

__all__ = [
    "create_client",
    "do_affiliations_match",
    "does_email_match_domain",
    "extract_profile_ids_by_author",
    "extract_profiles_by_author",
    "extract_profiles_by_id",
    "fetch_profile_by_id",
    "filter_profiles_by_affiliation",
    "filter_profiles_by_email",
    "find_author_profile_ids",
    "get_unique_profiles",
    "load_or_fetch_profile_by_author",
    "load_or_fetch_profile_by_id",
    "load_or_fetch_profile_ids",
    "log_profile_ids_stats",
    "search_profiles_by_name",
]

from candidex.openreview.client import create_client
from candidex.openreview.filtering import (
    filter_profiles_by_affiliation,
    filter_profiles_by_email,
)
from candidex.openreview.matching import do_affiliations_match, does_email_match_domain
from candidex.openreview.profile import (
    extract_profiles_by_author,
    extract_profiles_by_id,
    fetch_profile_by_id,
    get_unique_profiles,
    load_or_fetch_profile_by_author,
    load_or_fetch_profile_by_id,
)
from candidex.openreview.profile_id import (
    extract_profile_ids_by_author,
    find_author_profile_ids,
    load_or_fetch_profile_ids,
    log_profile_ids_stats,
)
from candidex.openreview.search import search_profiles_by_name
