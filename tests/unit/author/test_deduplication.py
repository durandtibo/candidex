from __future__ import annotations

from candidex.author import Author, deduplicate_authors

# --- Helpers ---


def make_author(
    name: str,
    affiliations: list[str] | None = None,
    email: str | None = None,
) -> Author:
    return Author.from_raw(name, affiliations, email)


#########################################
#     Tests for deduplicate_authors     #
#########################################

# --- Basic cases ---


def test_deduplicate_authors_empty_list() -> None:
    assert deduplicate_authors([]) == []


def test_deduplicate_authors_single_author() -> None:
    author = make_author("Jane Smith", ["MIT"])
    assert deduplicate_authors([author]) == [author]


def test_deduplicate_authors_no_duplicates() -> None:
    authors = [make_author("Jane Smith", ["MIT"]), make_author("John Doe", ["Stanford"])]
    assert deduplicate_authors(authors) == authors


# --- Deduplication ---


def test_deduplicate_authors_removes_exact_duplicate() -> None:
    author = make_author("Jane Smith", ["MIT"])
    result = deduplicate_authors([author, author])
    assert result == [author]


def test_deduplicate_authors_removes_duplicate_constructed_separately() -> None:
    a = make_author("Jane Smith", ["MIT"])
    b = make_author("Jane Smith", ["MIT"])
    assert deduplicate_authors([a, b]) == [a]


def test_deduplicate_authors_keeps_same_name_different_affiliations() -> None:
    a = make_author("Jane Smith", ["MIT"])
    b = make_author("Jane Smith", ["Stanford"])
    assert deduplicate_authors([a, b]) == [a, b]


def test_deduplicate_authors_keeps_same_name_different_email() -> None:
    a = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    b = make_author("Jane Smith", ["MIT"], "jane@stanford.edu")
    assert deduplicate_authors([a, b]) == [a, b]


def test_deduplicate_authors_keeps_same_name_none_vs_email() -> None:
    a = make_author("Jane Smith", ["MIT"], None)
    b = make_author("Jane Smith", ["MIT"], "jane@mit.edu")
    assert deduplicate_authors([a, b]) == [a, b]


def test_deduplicate_authors_keeps_same_name_none_vs_affiliations() -> None:
    a = make_author("Jane Smith", None)
    b = make_author("Jane Smith", ["MIT"])
    assert deduplicate_authors([a, b]) == [a, b]


# --- Order preservation ---


def test_deduplicate_authors_preserves_first_occurrence() -> None:
    first = make_author("Jane Smith", ["MIT"])
    second = make_author("Jane Smith", ["MIT"])
    result = deduplicate_authors([first, second])
    assert result[0] is first


def test_deduplicate_authors_preserves_insertion_order() -> None:
    a = make_author("Charlie", ["MIT"])
    b = make_author("Alice", ["Stanford"])
    c = make_author("Bob", ["CMU"])
    assert deduplicate_authors([a, b, c]) == [a, b, c]


def test_deduplicate_authors_preserves_order_with_duplicates() -> None:
    a = make_author("Jane Smith", ["MIT"])
    b = make_author("John Doe", ["Stanford"])
    c = make_author("Jane Smith", ["MIT"])
    result = deduplicate_authors([a, b, c])
    assert result == [a, b]
