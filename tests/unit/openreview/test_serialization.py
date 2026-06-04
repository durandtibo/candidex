from __future__ import annotations

import json

from openreview import Profile

from candidex.openreview import deserialize_profiles, serialize_profiles

# --- Helpers ---


def make_profile(profile_id: str, fullname: str, position: str, institution: str) -> Profile:
    return Profile(
        id=profile_id,
        content={
            "names": [{"fullname": fullname}],
            "history": [{"position": position, "institution": {"name": institution}}],
        },
    )


##########################################
#     Tests for deserialize_profiles     #
##########################################


def test_deserialize_profiles_empty_list() -> None:
    assert deserialize_profiles([]) == []


def test_deserialize_profiles_returns_list_of_profiles() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = deserialize_profiles(serialize_profiles([profile]))
    assert isinstance(result, list)
    assert all(isinstance(p, Profile) for p in result)


def test_deserialize_profiles_single_profile_id() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = deserialize_profiles(serialize_profiles([profile]))
    assert result[0].id == "~Jane_Smith1"


def test_deserialize_profiles_single_profile_content() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = deserialize_profiles(serialize_profiles([profile]))
    assert result[0].content == {
        "names": [{"fullname": "Jane Smith"}],
        "history": [{"position": "PhD Student", "institution": {"name": "MIT"}}],
    }


def test_deserialize_profiles_multiple_profiles() -> None:
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~John_Doe1", "John Doe", "Professor", "Stanford")
    result = deserialize_profiles(serialize_profiles([profile_a, profile_b]))
    assert len(result) == 2
    assert result[0].id == "~Jane_Smith1"
    assert result[1].id == "~John_Doe1"


def test_deserialize_profiles_preserves_order() -> None:
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~John_Doe1", "John Doe", "Professor", "Stanford")
    profile_c = make_profile("~Alice_Brown1", "Alice Brown", "Postdoc", "CMU")
    result = deserialize_profiles(serialize_profiles([profile_a, profile_b, profile_c]))
    assert result[0].id == "~Jane_Smith1"
    assert result[1].id == "~John_Doe1"
    assert result[2].id == "~Alice_Brown1"


def test_deserialize_profiles_round_trip() -> None:
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~John_Doe1", "John Doe", "Professor", "Stanford")
    profiles = [profile_a, profile_b]
    assert serialize_profiles(
        deserialize_profiles(serialize_profiles(profiles))
    ) == serialize_profiles(profiles)


########################################
#     Tests for serialize_profiles     #
########################################


def test_serialize_profiles_empty_list() -> None:
    assert serialize_profiles([]) == []


def test_serialize_profiles_returns_list_of_strings() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = serialize_profiles([profile])
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


def test_serialize_profiles_single_profile_length() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    assert len(serialize_profiles([profile])) == 1


def test_serialize_profiles_multiple_profiles_length() -> None:
    profiles = [
        make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT"),
        make_profile("~John_Doe1", "John Doe", "Professor", "Stanford"),
    ]
    assert len(serialize_profiles(profiles)) == 2


# --- Content ---


def test_serialize_profiles_preserves_id() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = json.loads(serialize_profiles([profile])[0])
    assert result["id"] == "~Jane_Smith1"


def test_serialize_profiles_preserves_names() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = json.loads(serialize_profiles([profile])[0])
    assert result["content"]["names"] == [{"fullname": "Jane Smith"}]


def test_serialize_profiles_preserves_history() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    result = json.loads(serialize_profiles([profile])[0])
    assert result["content"]["history"] == [
        {"position": "PhD Student", "institution": {"name": "MIT"}}
    ]


# --- Order preservation ---


def test_serialize_profiles_preserves_order() -> None:
    profile_a = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    profile_b = make_profile("~John_Doe1", "John Doe", "Professor", "Stanford")
    profile_c = make_profile("~Alice_Brown1", "Alice Brown", "Postdoc", "CMU")
    result = serialize_profiles([profile_a, profile_b, profile_c])
    assert json.loads(result[0])["id"] == "~Jane_Smith1"
    assert json.loads(result[1])["id"] == "~John_Doe1"
    assert json.loads(result[2])["id"] == "~Alice_Brown1"


# --- Round-trip ---


def test_serialize_profiles_round_trip() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    serialized = serialize_profiles([profile])
    restored = Profile.from_json(json.loads(serialized[0]))
    assert restored.id == profile.id
    assert restored.content == profile.content


def test_serialize_profiles_produces_valid_json() -> None:
    profile = make_profile("~Jane_Smith1", "Jane Smith", "PhD Student", "MIT")
    serialized = serialize_profiles([profile])
    for s in serialized:
        json.loads(s)
