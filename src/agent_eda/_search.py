"""Keyword search over the registered analyses, deterministic and offline.

Each analysis is searched by the words of its name, its tags, its docstring's
summary line and its When to use: text, read from the registry. Words match
after lowercasing and dropping a plural ending, and a word agents use for one of
the package's words also matches the package's word, as SYNONYMS lists. A query
word scores the weight of the best field it occurs in, and an analysis's score
is the sum over the query's words. A tag that the query holds whole, such as
'missing values' in 'how many missing values', scores more than any field. Words
that most analyses use, GENERIC, score at most 1, so 'rows' doesn't lead to
find_duplicate_rows, and one-letter words are dropped. A match that rests on
generic words or on the When to use: text alone is weak. Ties go to the
alphabetically first name. There are no embeddings, no network and no
randomness, so a query always gives the same ranking.

To make an analysis easier to find, give it tags in its registration decorator:
they weigh more than its docstring's prose. Add to SYNONYMS only a word agents
use where the package's docs use another.
"""

import re
from dataclasses import dataclass

from agent_eda._registry import AnalysisSpec

# How much a query word scores when it occurs in each field of an analysis, and
# when a tag that the query holds whole covers it.
WHOLE_TAG_WEIGHT = 5
NAME_WEIGHT = 4
TAG_WEIGHT = 3
SUMMARY_WEIGHT = 2
WHEN_TO_USE_WEIGHT = 1
# What a generic word scores at most.
GENERIC_WEIGHT = 1
# The most distinct words of a query that are searched, listed and matched.
MAX_TERMS = 20

# Words, without a plural ending, that most analyses use, so that they say little
# about which one fits.
GENERIC = frozenset({"column", "value", "row", "data", "count"})

# Phrases that ask where to start, which no single analysis answers: the guide and
# the MCP server's instructions give the order for a first look.
FIRST_LOOK_PHRASES: tuple[tuple[str, ...], ...] = (
    ("first", "look"),
    ("eda",),
    ("exploratory", "data", "analysis"),
    ("where", "to", "start"),
    ("getting", "started"),
)

# Words too common to say anything about which analysis fits.
STOPWORDS = frozenset(
    (
        *("a", "an", "the", "this", "that", "these", "those", "there", "their"),
        *("i", "me", "my", "we", "our", "it", "its"),
        *("is", "are", "was", "be", "do", "does", "has", "have", "can"),
        *("what", "which", "when", "where", "how", "and", "or", "any"),
        *("of", "on", "in", "at", "to", "by", "for", "from", "as", "per", "with"),
    )
)

# Each group starts with a word the package's docs use, followed by words agents
# use for it. A query word from the rest of a group also matches the first word,
# but not the other way round: 'label' finds the target, while 'target' doesn't
# find the column labels describe_dataset reports.
SYNONYMS: tuple[tuple[str, ...], ...] = (
    ("missing", "null", "nan", "na", "missingness"),
    ("duplicate", "duplicated", "dupe", "dup"),
    ("association", "associated", "correlation", "correlated", "correlate", "corr"),
    ("target", "label", "outcome"),
    # The package's word for hiding values is masking (ADR 0001).
    (
        "masking",
        *("mask", "pii", "privacy", "anonymize", "anonymise", "redact", "hide"),
        *("sensitive", "personal"),
    ),
)

# A run of letters and digits in any script; underscores split words.
_WORD = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class Term:
    """A word of the query and the other words it matches."""

    # The query word, lowercased.
    word: str
    # The package's word for it when agents use it for one, without a plural
    # ending; empty otherwise.
    synonyms: tuple[str, ...]

    @property
    def stems(self) -> frozenset[str]:
        return frozenset({stem(self.word), *self.synonyms})

    @property
    def generic(self) -> bool:
        return stem(self.word) in GENERIC


@dataclass(frozen=True)
class Match:
    """An analysis that matched a query, with its score and the words that matched."""

    spec: AnalysisSpec
    score: int
    matched_terms: list[str]
    # Whether only generic words or the When to use: text matched.
    weak: bool


def terms(query: str) -> list[Term]:
    """The query's words without stopwords and one-letter words, each once, with
    the package's word."""
    package_words = {
        stem(word): stem(group[0]) for group in SYNONYMS for word in group[1:]
    }
    found: dict[str, Term] = {}
    for word in words(query):
        root = stem(word)
        if len(word) < 2 or word in STOPWORDS or root in found:
            continue
        synonym = package_words.get(root)
        found[root] = Term(word, (synonym,) if synonym and synonym != root else ())
    return list(found.values())


def words(text: str) -> list[str]:
    """The words of a text, lowercased; underscores and punctuation split words."""
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """The word without a plural ending, so that 'nulls' matches 'null'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def asks_for_first_look(query: str) -> bool:
    """Whether the query asks where to start, such as 'first look' or 'EDA'."""
    found = words(query)
    return any(
        found[start : start + len(phrase)] == list(phrase)
        for phrase in FIRST_LOOK_PHRASES
        for start in range(len(found))
    )


def rank(specs: list[AnalysisSpec], searched: list[Term], query: str) -> list[Match]:
    """The analyses that match the terms, best first, ties by name.

    An analysis whose name is the query itself comes first.
    """
    exact = "_".join(words(query))
    query_stems = frozenset().union(*(term.stems for term in searched))
    matches = []
    for spec in specs:
        fields = (
            (NAME_WEIGHT, _stems(spec.name)),
            (SUMMARY_WEIGHT, _stems(spec.summary)),
            (WHEN_TO_USE_WEIGHT, _stems(spec.docstring.when_to_use)),
        )
        tags = [_stems(tag) for tag in spec.tags]
        whole = set().union(*(tag for tag in tags if _held_whole(tag, query_stems)))
        score = 0
        matched = []
        strong = False
        for term in searched:
            if term.stems & whole:
                weight = WHOLE_TAG_WEIGHT
                strong = True
            else:
                weight = max(
                    (
                        *(weight for weight, found in fields if term.stems & found),
                        *(TAG_WEIGHT for tag in tags if term.stems & tag),
                    ),
                    default=0,
                )
                if term.generic:
                    weight = min(weight, GENERIC_WEIGHT)
                elif weight >= SUMMARY_WEIGHT:
                    strong = True
            if weight:
                score += weight
                matched.append(term.word)
        if matched:
            matches.append(
                Match(spec=spec, score=score, matched_terms=matched, weak=not strong)
            )
    return sorted(
        matches,
        key=lambda match: (match.spec.name != exact, -match.score, match.spec.name),
    )


def _held_whole(tag: set[str], query_stems: frozenset[str]) -> bool:
    """Whether the query holds the whole tag: every word of it but the generic
    ones, or every word when all of them are generic."""
    needed = {word for word in tag if word not in GENERIC} or tag
    return bool(needed) and needed <= query_stems


def _stems(text: str) -> set[str]:
    return {stem(word) for word in words(text)}
