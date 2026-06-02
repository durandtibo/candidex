from __future__ import annotations

from candidex.author import Author, sort_authors

# --- Helpers ---


def make_author(
    name: str,
    affiliations: list[str] | None = None,
    email: str | None = None,
) -> Author:
    return Author.from_raw(name, affiliations, email)


##################################
#     Tests for sort_authors     #
##################################


# --- Basic cases ---


def test_sort_authors_empty_list() -> None:
    assert sort_authors([]) == []


def test_sort_authors_single_author() -> None:
    author = make_author("Jane Smith", ["MIT"])
    assert sort_authors([author]) == [author]


def test_sort_authors_already_sorted() -> None:
    authors = [make_author("Alice"), make_author("Bob"), make_author("Charlie")]
    assert sort_authors(authors) == authors


# --- Sorting by name ---


def test_sort_authors_sorts_by_name() -> None:
    a = make_author("John Doe")
    b = make_author("Jane Smith")
    c = make_author("Alice Brown")
    result = sort_authors([a, b, c])
    assert [r.name for r in result] == ["Alice Brown", "Jane Smith", "John Doe"]


def test_sort_authors_is_case_insensitive() -> None:
    a = make_author("alice Brown")
    b = make_author("Bob Smith")
    result = sort_authors([b, a])
    assert result[0].name == "alice Brown"


# --- Sorting by affiliation when names are equal ---


def test_sort_authors_sorts_by_affiliation_when_same_name() -> None:
    a = make_author("Jane Smith", ["Stanford"])
    b = make_author("Jane Smith", ["CMU"])
    c = make_author("Jane Smith", ["MIT"])
    result = sort_authors([a, b, c])
    assert [r.format_affiliations() for r in result] == ["CMU", "MIT", "Stanford"]


def test_sort_authors_affiliation_sort_is_case_insensitive() -> None:
    a = make_author("Jane Smith", ["stanford"])
    b = make_author("Jane Smith", ["MIT"])
    result = sort_authors([a, b])
    assert result[0].format_affiliations() == "MIT"


def test_sort_authors_none_affiliation_sorts_before_non_none() -> None:
    a = make_author("Jane Smith", ["MIT"])
    b = make_author("Jane Smith", None)
    result = sort_authors([a, b])
    assert result[0].affiliations is None


# --- Does not modify original ---


def test_sort_authors_does_not_modify_input() -> None:
    authors = [make_author("John Doe"), make_author("Alice Brown")]
    original_order = list(authors)
    sort_authors(authors)
    assert authors == original_order


# --- Mixed names and affiliations ---


def test_sort_authors_mixed() -> None:
    a = make_author("John Doe", ["Stanford"])
    b = make_author("Jane Smith", ["MIT"])
    c = make_author("Jane Smith", ["CMU"])
    d = make_author("Alice Brown")
    result = sort_authors([a, b, c, d])
    assert [r.name for r in result] == ["Alice Brown", "Jane Smith", "Jane Smith", "John Doe"]
    assert result[1].format_affiliations() == "CMU"
    assert result[2].format_affiliations() == "MIT"


def test_sort_authors_reverse_sorts_descending_by_name() -> None:
    a = make_author("Alice Brown")
    b = make_author("John Doe")
    c = make_author("Jane Smith")
    result = sort_authors([a, b, c], reverse=True)
    assert [r.name for r in result] == ["John Doe", "Jane Smith", "Alice Brown"]


def test_sort_authors_reverse_sorts_descending_by_affiliation_when_same_name() -> None:
    a = make_author("Jane Smith", ["CMU"])
    b = make_author("Jane Smith", ["MIT"])
    c = make_author("Jane Smith", ["Stanford"])
    result = sort_authors([a, b, c], reverse=True)
    assert [r.format_affiliations() for r in result] == ["Stanford", "MIT", "CMU"]


def test_sort_authors_reverse_false_is_same_as_default() -> None:
    authors = [make_author("John Doe"), make_author("Alice Brown")]
    assert sort_authors(authors, reverse=False) == sort_authors(authors)


def test_sort_authors_reverse_single_author_unchanged() -> None:
    author = make_author("Jane Smith", ["MIT"])
    assert sort_authors([author], reverse=True) == [author]


def test_sort_authors_reverse_empty_list() -> None:
    assert sort_authors([], reverse=True) == []
