"""Keyword search over the registered analyses, deterministic and offline.

Each analysis is searched by the words of its name, its tags, its docstring's
summary line and its When to use: text, read from the registry. Words match
after lowercasing and dropping a plural ending, and a word agents use for one of
the package's words also matches the package's word, as SYNONYMS lists. A query
word scores the weight of the best field it occurs in, and an analysis's score
is the sum over the query's words. Ties go to the alphabetically first name.
There are no embeddings, no network and no randomness, so a query always gives
the same ranking.

To make an analysis easier to find, give it tags in its registration decorator:
they weigh more than its docstring's prose. Add to SYNONYMS only a word agents
use where the package's docs use another.
"""

import re
from dataclasses import dataclass

from agent_eda._registry import AnalysisSpec

# How much a query word scores when it occurs in each field of an analysis.
NAME_WEIGHT = 4
TAG_WEIGHT = 3
SUMMARY_WEIGHT = 2
WHEN_TO_USE_WEIGHT = 1

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
    ("missing", "null", "nan", "na"),
    ("duplicate", "duplicated", "dupe", "dup"),
    ("association", "associated", "correlation", "correlated", "correlate", "corr"),
    ("target", "label", "outcome"),
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


@dataclass(frozen=True)
class Match:
    """An analysis that matched a query, with its score and the words that matched."""

    spec: AnalysisSpec
    score: int
    matched_terms: list[str]


def terms(query: str) -> list[Term]:
    """The query's words without stopwords, each once, with the package's word."""
    package_words = {
        stem(word): stem(group[0]) for group in SYNONYMS for word in group[1:]
    }
    found: dict[str, Term] = {}
    for word in words(query):
        root = stem(word)
        if word in STOPWORDS or root in found:
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


def rank(specs: list[AnalysisSpec], searched: list[Term], query: str) -> list[Match]:
    """The analyses that match the terms, best first, ties by name.

    An analysis whose name is the query itself comes first.
    """
    exact = "_".join(words(query))
    matches = []
    for spec in specs:
        fields = (
            (NAME_WEIGHT, _stems(spec.name)),
            (TAG_WEIGHT, _stems(" ".join(spec.tags))),
            (SUMMARY_WEIGHT, _stems(spec.summary)),
            (WHEN_TO_USE_WEIGHT, _stems(spec.docstring.when_to_use)),
        )
        score = 0
        matched = []
        for term in searched:
            weight = max(
                (weight for weight, found in fields if term.stems & found), default=0
            )
            if weight:
                score += weight
                matched.append(term.word)
        if matched:
            matches.append(Match(spec=spec, score=score, matched_terms=matched))
    return sorted(
        matches,
        key=lambda match: (match.spec.name != exact, -match.score, match.spec.name),
    )


def _stems(text: str) -> set[str]:
    return {stem(word) for word in words(text)}
