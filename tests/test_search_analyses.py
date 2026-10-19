import inspect
import itertools
import json

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import agent_eda as ae
from agent_eda.errors import InvalidArgumentError

SUMMARIES = {info.name: info.summary for info in ae.list_analyses().analyses}
TAGS = {info.name: info.tags for info in ae.list_analyses().analyses}

# The analyses that can show raw values, so a query on privacy leads to any of them.
MASKING = tuple(
    info.name for info in ae.list_analyses().analyses if info.supports_masking
)

# The search acceptance cases: a query in an agent's words, not the package's, and
# the analysis it must find first: one name, any of a tuple of names, or None for a
# query no analysis answers, which finds nothing. Each analysis ticket adds its rows
# when the analysis ships, and every earlier row keeps passing.
SEARCH_ACCEPTANCE = [
    ("missing values", "profile_missing_values"),
    ("nulls per column", "profile_missing_values"),
    ("-999 placeholder codes", "detect_missing_sentinels"),
    ("primary key", "find_candidate_keys"),
    ("dataset overview", "describe_dataset"),
    ("mean and standard deviation", "summarize_numeric"),
    ("what kind of column is this", "infer_column_semantics"),
    ("value counts", "summarize_categorical"),
    ("duplicate rows", "find_duplicate_rows"),
    ("correlation", "compute_association_matrix"),
    ("class imbalance", "profile_target"),
    ("check data types against a schema", "validate_column_types"),
    # AX-10: generic words used to dominate, and one-letter words matched.
    ("how many rows", "describe_dataset"),
    ("zero variance", "describe_dataset"),
    ("typos in category labels", "summarize_categorical"),
    ("multicollinearity", "compute_association_matrix"),
    ("percentiles", "summarize_numeric"),
    ("capitalization", "summarize_categorical"),
    ("first look", None),
    ("eda", None),
    ("n", None),
    ("leakage", None),
    # CORE-16 and STD-34: privacy words lead to the analyses that mask.
    ("PII", MASKING),
    ("privacy", MASKING),
    ("anonymize", MASKING),
    ("hide sensitive values", MASKING),
    ("redact email addresses", MASKING),
    # SPEC-11: missing values lead to the profile, not to the sentinels.
    ("nulls", "profile_missing_values"),
    ("NaN", "profile_missing_values"),
    ("missing", "profile_missing_values"),
    ("null counts", "profile_missing_values"),
    ("na counts", "profile_missing_values"),
    ("missingness", "profile_missing_values"),
    # The spec's table of synonyms and tags.
    ("rows", "describe_dataset"),
    ("row count", "describe_dataset"),
    ("constant", "describe_dataset"),
    ("skewed", "summarize_numeric"),
    ("outliers", "summarize_numeric"),
    ("whitespace", "summarize_categorical"),
]


def accepted(expected):
    """The names a query may lead with: none, one, or any of several."""
    return (
        ()
        if expected is None
        else (expected,)
        if isinstance(expected, str)
        else expected
    )


def first_match(query):
    matches = ae.search_analyses(query).matches
    return matches[0].name if matches else None


@pytest.mark.parametrize(
    ("query", "expected"),
    SEARCH_ACCEPTANCE,
    ids=[query for query, _ in SEARCH_ACCEPTANCE],
)
def test_an_agents_wording_finds_the_right_analysis_first(query, expected):
    found = ae.search_analyses(query)

    if expected is None:
        assert found.matches == []
    else:
        assert found.matches[0].name in accepted(expected)


@given(query=st.text() | st.text(min_size=100), limit=st.integers(1, 10))
# The first line once filled up to 120 characters right before a long word, and
# ended in a space.
@example(
    query="0" * 26
    + "\x1f" * 3
    + "\U000107bb" * 2
    + "\U0001239a\U00012544"
    + "\U00018cd6" * 2
    + " "
    + "0" * 23
    + "\x1f" * 7
    + "\x08" * 2
    + "\U0001239a" * 3
    + "\U00014647\U00018cd6\U0002ebe1",
    limit=1,
)
def test_any_query_gives_the_same_strict_listing_within_the_text_caps(query, limit):
    found = ae.search_analyses(query, limit=limit)

    record = found.to_json()
    assert ae.search_analyses(query, limit=limit).to_json() == record
    json.loads(record, parse_constant=reject_constant)
    lines = str(found).splitlines()
    assert max(len(line) for line in lines) <= 120
    assert lines == [line.rstrip() for line in lines]
    assert len(lines) <= 40
    assert len(found.matches) <= min(limit, found.match_count)
    searched = [term.term for term in found.terms]
    for match in found.matches:
        assert match.matched_terms == [t for t in searched if t in match.matched_terms]
    # An analysis named by the query may lead; the rest go by score, then name.
    keys = [(-match.score, match.name) for match in found.matches[1:]]
    assert keys == sorted(keys)


def reject_constant(name):
    raise AssertionError(f"to_json() isn't strict JSON: it holds {name}")


@pytest.mark.parametrize(
    ("query", "synonym"),
    [
        ("nulls", "missing"),
        ("NaN", "missing"),
        ("dupes", "duplicate"),
        ("correlation", "association"),
        ("label", "target"),
    ],
)
def test_an_agents_word_also_searches_for_the_package_word(query, synonym):
    found = ae.search_analyses(query)

    [term] = found.terms
    assert term.term == query.lower()
    assert term.synonyms == [synonym]


def test_one_letter_words_are_dropped_from_the_query():
    found = ae.search_analyses("n x a matrix")

    assert [term.term for term in found.terms] == ["matrix"]


def test_generic_words_alone_give_only_weak_matches():
    found = ae.search_analyses("values per column", limit=100)

    assert found.matches
    assert all(match.weak for match in found.matches)
    lines = str(found).splitlines()
    assert lines[1].startswith("  ")
    assert "(weak match)" in lines[1]
    assert any("weak" in line and not line.startswith("  ") for line in lines)


def test_a_match_on_a_specific_word_is_not_weak():
    found = ae.search_analyses("mean and standard deviation")

    best = found.matches[0]
    assert best.name == "summarize_numeric"
    assert not best.weak
    assert "weak" not in str(found)


def test_a_match_only_in_the_when_to_use_text_is_weak():
    found = ae.search_analyses("trimmed", limit=100)

    [match] = [m for m in found.matches if m.name == "summarize_numeric"]
    assert match.weak


def test_generic_words_weigh_less_than_specific_ones():
    # 'rows' is in find_duplicate_rows' name, yet describe_dataset answers the
    # question; and 'values' and 'columns' never decide a ranking.
    assert first_match("how many rows") == "describe_dataset"
    assert first_match("duplicate values in columns") == "find_duplicate_rows"
    scores = [m.score for m in ae.search_analyses("column", limit=100).matches]
    assert scores == [1] * len(scores)


@pytest.mark.parametrize("query", ["first look", "eda", "First look at my data"])
def test_a_first_look_query_finds_no_analysis_and_points_to_the_first_look_order(query):
    found = ae.search_analyses(query)

    assert found.matches == []
    assert found.first_look is True
    assert "ae.guide()" in str(found)
    assert "first look" in str(found)


def test_other_queries_are_no_first_look():
    assert ae.search_analyses("missing values").first_look is False
    assert ae.search_analyses("xylophone").first_look is False


def test_a_long_query_is_cut_to_forty_characters_in_the_text():
    query = "missing " * 1000

    found = ae.search_analyses(query)

    lines = str(found).splitlines()
    assert len(lines) <= 40
    assert found.query == query
    assert lines[0].startswith(
        "Analyses matching 'missing missing missing missing missing"
    )
    assert lines[0].count("missing") <= 6
    assert "…" in lines[0]


def test_the_words_listed_are_capped_with_their_count():
    query = " ".join(f"word{number}" for number in range(8000 // 8))

    found = ae.search_analyses(query)

    assert len(found.terms) == 20
    assert found.term_count == 1000
    text = str(found)
    assert len(text.splitlines()) <= 40
    assert "first 20 of 1,000 words" in text


def test_an_eight_thousand_character_query_prints_within_the_line_cap():
    for query in ["missing " * 1000, "x" * 8000, "a1 " * 2700, "nulls dupes " * 700]:
        for limit in (1, 5, 100):
            text = str(ae.search_analyses(query, limit=limit))
            lines = text.splitlines()
            assert len(lines) <= 40
            assert max(len(line) for line in lines) <= 120


def test_the_text_shows_the_matches_that_fit_and_counts_the_rest():
    found = ae.search_analyses("column", limit=100)

    text = str(found)
    assert len(text.splitlines()) <= 40
    shown = text.splitlines()[0]
    assert f"({len(found.matches)} of {found.match_count} shown" in shown


@pytest.mark.parametrize("word", ["missing", "duplicate", "association", "target"])
def test_the_package_word_searches_for_itself_only(word):
    # So 'target column' doesn't find the column labels describe_dataset reports.
    [term] = ae.search_analyses(word).terms

    assert term.synonyms == []


def test_a_synonym_finds_analyses_that_use_the_package_word():
    # Both analyses speak of missing values, never of nulls.
    found = ae.search_analyses("nulls", limit=100)

    assert {m.name for m in found.matches} >= {"describe_dataset", "summarize_numeric"}
    assert all(m.matched_terms == ["nulls"] for m in found.matches)


def test_plurals_match_singulars():
    singular = ae.search_analyses("quantile", limit=100)
    plural = ae.search_analyses("quantiles", limit=100)

    assert plural.matches
    assert [(m.name, m.score) for m in plural.matches] == [
        (m.name, m.score) for m in singular.matches
    ]


def test_column_labels_find_describe_dataset_although_label_means_target():
    # 'label' also finds the target, as a synonym. describe_dataset reports
    # duplicated and colliding column labels, so its tags name them and keep it
    # first once a target analysis exists.
    assert "duplicated column labels" in TAGS["describe_dataset"]

    found = ae.search_analyses("duplicated column labels")

    assert found.matches[0].name == "describe_dataset"


@pytest.mark.parametrize(
    "query", ["missing", "column", "counts", "values per column", "nulls or dupes"]
)
def test_ranks_by_score_then_alphabetically_by_name(query):
    found = ae.search_analyses(query, limit=100)

    keys = [(-match.score, match.name) for match in found.matches]
    assert keys
    assert keys == sorted(keys)


def test_a_tie_goes_to_the_alphabetically_first_name():
    matches = ae.search_analyses("missing", limit=100).matches

    # Several analyses mention missing values only in their When to use: text.
    ties = [(a, b) for a, b in itertools.pairwise(matches) if a.score == b.score]
    assert ties
    assert all(first.name < second.name for first, second in ties)


def test_the_same_query_gives_the_same_listing():
    first = ae.search_analyses("Nulls per column, and dupes?")
    second = ae.search_analyses("Nulls per column, and dupes?")

    assert first.to_json() == second.to_json()
    assert [t.term for t in first.terms] == ["nulls", "column", "dupes"]


def test_limit_keeps_the_best_matches_and_the_listing_counts_them_all():
    everything = ae.search_analyses("missing", limit=100)

    found = ae.search_analyses("missing", limit=1)

    assert found.limit == 1
    assert [m.name for m in found.matches] == [everything.matches[0].name]
    assert found.match_count == len(everything.matches) > 1
    assert ae.search_analyses("missing").limit == 5


@pytest.mark.parametrize("limit", [0, -1, 2.5, True, None])
def test_a_limit_that_is_not_a_positive_int_raises(limit):
    with pytest.raises(InvalidArgumentError) as caught:
        ae.search_analyses("missing", limit=limit)

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        f"search_analyses got {limit!r} for limit, but limit must be an int of at "
        "least 1, such as ae.search_analyses('missing values', limit=3)."
    )


def test_a_query_that_is_not_a_string_raises_with_an_example():
    with pytest.raises(InvalidArgumentError) as caught:
        ae.search_analyses(["missing", "values"])  # ty: ignore[invalid-argument-type]  # the wrong type is the test

    assert caught.value.code == "invalid_argument"
    assert str(caught.value) == (
        "search_analyses takes the query as a string, such as "
        "ae.search_analyses('missing values'), but got a value of type list."
    )


def test_prints_the_matches_best_first_with_what_matched():
    found = ae.search_analyses("nulls and dupes")

    lines = str(found).splitlines()

    shown, total = len(found.matches), found.match_count
    assert lines[0] == (
        f"Analyses matching 'nulls and dupes' ({shown} of {total} shown, best first):"
    )
    best = found.matches[0]
    assert lines[1].startswith(f"  {best.name}: {best.summary} Matched: ")
    assert ("Searched for: nulls (also missing), dupes (also duplicate).") in lines
    assert lines[-1] == (
        f"Next: print(ae.describe_analysis({best.name!r})) shows how to call it."
    )
    assert repr(found) == str(found)
    assert max(len(line) for line in lines) <= 120


def test_a_query_without_matches_says_so_and_points_to_the_guide():
    found = ae.search_analyses("xylophone")

    assert found.matches == []
    assert found.match_count == 0
    assert str(found) == (
        "No analysis matches 'xylophone'. Searched for: xylophone.\n"
        "Try other words, or call ae.guide() for every analysis and where to start."
    )


@pytest.mark.parametrize("query", ["what is this?", "", "   "])
def test_a_query_without_searchable_words_searches_for_nothing(query):
    found = ae.search_analyses(query)

    assert found.terms == []
    assert found.matches == []
    # The sentence wraps at 120 characters, its continuation indented.
    expected = (
        f"No analysis matches {query!r}. Searched for: no words, since common ones "
        "such as 'the' and 'of' and single letters don't count."
    )
    assert " ".join(str(found).split()).startswith(" ".join(expected.split()))


def test_words_with_letters_beyond_ascii_stay_whole():
    found = ae.search_analyses("naïve Größe")

    # 'naïve' must not split into 'na', a synonym of missing, and 've'.
    assert [term.term for term in found.terms] == ["naïve", "größe"]
    assert found.matches == []


def test_says_how_to_see_the_matches_beyond_the_limit():
    found = ae.search_analyses("missing", limit=1)

    total = found.match_count
    assert (
        f"More: ae.search_analyses('missing', limit={total}) shows all {total} matches."
    ) in str(found).splitlines()


@pytest.mark.parametrize(
    ("word", "name"),
    [("trimmed", "summarize_numeric"), ("structural", "describe_dataset")],
)
def test_finds_words_from_the_when_to_use_text_of_the_docstring(word, name):
    # The word occurs in the docstring's When to use: text and nowhere else that
    # search reads: not in the name, the tags or the summary.
    when_to_use = docstring(name).split("When to use:")[1].split("Args:")[0]
    assert word in when_to_use
    assert word not in " ".join([name, SUMMARIES[name], *TAGS[name]]).lower()

    found = ae.search_analyses(word, limit=100)

    [match] = [m for m in found.matches if m.name == name]
    assert match.matched_terms == [word]


def test_match_summaries_are_the_first_lines_of_the_docstrings():
    found = ae.search_analyses("column", limit=100)

    assert found.matches
    for match in found.matches:
        assert match.summary == docstring(match.name).splitlines()[0]


def docstring(name):
    doc = inspect.getdoc(getattr(ae, name))
    assert doc
    return doc


def test_each_match_has_its_summary_and_the_terms_that_matched():
    found = ae.search_analyses("mean and standard deviation")

    best = found.matches[0]
    assert best.name == "summarize_numeric"
    assert best.module == "agent_eda.statistics.descriptive"
    assert best.summary == SUMMARIES["summarize_numeric"]
    assert best.matched_terms == ["mean", "standard", "deviation"]
