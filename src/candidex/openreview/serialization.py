r"""Contain serialization and deserialization functionalities."""

from __future__ import annotations

import json

__all__ = ["deserialize_profiles", "serialize_profiles"]

from openreview import Profile


def serialize_profiles(profiles: list[Profile | None]) -> list[str | None]:
    """Serialize a list of OpenReview profiles to a list of JSON
    strings.

    Converts each `Profile` object to its JSON representation using
    `Profile.to_json()`, then serializes it to a string with `json.dumps`.
    `None` values are preserved as `None` in the output.

    Useful for storing profiles in a Polars DataFrame column of type
    `List[String]`, which supports equality comparison and persistence
    unlike `pl.Object`.

    Args:
        profiles: A list of `openreview.Profile` objects or `None` values
                  to serialize.

    Returns:
        A list of JSON strings or `None` values, one per profile, in the
            same order as the input. Returns an empty list if the input is empty.

    Example:
        ```pycon
        >>> import openreview
        >>> profile = openreview.Profile(
        ...     id="~Jane_Smith1",
        ...     content={
        ...         "names": [{"fullname": "Jane Smith"}],
        ...         "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
        ...     },
        ... )
        >>> from candidex.openreview import serialize_profiles
        >>> serialized = serialize_profiles([profile, None])
        >>> import json
        >>> json.loads(serialized[0])["id"]
        '~Jane_Smith1'
        >>> serialized[1] is None
        True

        ```
    """
    return [json.dumps(profile.to_json()) if profile is not None else None for profile in profiles]


def deserialize_profiles(serialized: list[str | None]) -> list[Profile | None]:
    """Deserialize a list of JSON strings to a list of OpenReview
    profiles.

    Converts each JSON string back to a `Profile` object using `json.loads`
    and `Profile.from_json`. `None` values are preserved as `None` in the
    output. This is the inverse of `serialize_profiles`.

    Args:
        serialized: A list of JSON strings or `None` values, each representing
                    a serialized `openreview.Profile` or absence of a profile,
                    as returned by `serialize_profiles`.

    Returns:
        A list of `openreview.Profile` objects or `None` values in the same
            order as the input. Returns an empty list if the input is empty.

    Example:
        ```
        >>> import openreview
        >>> profile = openreview.Profile(
        ...     id="~Jane_Smith1",
        ...     content={
        ...         "names": [{"fullname": "Jane Smith"}],
        ...         "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
        ...     },
        ... )
        >>> from candidex.openreview import serialize_profiles, deserialize_profiles
        >>> serialized = serialize_profiles([profile, None])
        >>> restored = deserialize_profiles(serialized)
        >>> restored[0].id
        '~Jane_Smith1'
        >>> restored[1] is None
        True

        ```
    """
    return [Profile.from_json(json.loads(s)) if s is not None else None for s in serialized]
