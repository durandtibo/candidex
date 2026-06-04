r"""Contain serialization and deserialization functionalities."""

from __future__ import annotations

import json

__all__ = ["serialize_profiles"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openreview import Profile


def serialize_profiles(profiles: list[Profile]) -> list[str]:
    """Serialize a list of OpenReview profiles to a list of JSON
    strings.

    Converts each `Profile` object to its JSON representation using
    `Profile.to_json()`, then serializes it to a string with `json.dumps`.
    Useful for storing profiles in a Polars DataFrame column of type
    `List[String]`, which supports equality comparison and persistence
    unlike `pl.Object`.

    Args:
        profiles: A list of `openreview.Profile` objects to serialize.

    Returns:
        A list of JSON strings, one per profile, in the same order as
        the input. Returns an empty list if the input is empty.

    Example:
        ```pycon
        >>> from openreview import Profile
        >>> from candidex.openreview import serialize_profiles
        >>> profile = Profile(
        ...     id="~Jane_Smith1",
        ...     content={
        ...         "names": [{"fullname": "Jane Smith"}],
        ...         "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
        ...     },
        ... )
        >>> serialized = serialize_profiles([profile])
        >>> serialized
        ['..."id": "~Jane_Smith1"}']

        ```
    """
    return [json.dumps(profile.to_json()) for profile in profiles]
