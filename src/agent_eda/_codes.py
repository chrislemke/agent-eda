"""What each code of the package-wide lists means.

The lists are `Literal`s in `agent_eda.results`, with one comment above each
code. The comment is the only copy of the code's meaning, so that the list and
its meanings can't disagree: describe_analysis and the README's Reference read
the comments from here.
"""

import functools
import inspect
import io
import tokenize
import typing

from agent_eda import results
from agent_eda.results import LimitationCode, StatusReasonCode


def status_reason_meanings() -> dict[StatusReasonCode, str]:
    """Each status reason code with its meaning, in the order the list has them."""
    return typing.cast(
        "dict[StatusReasonCode, str]", _meanings("StatusReasonCode", StatusReasonCode)
    )


def limitation_meanings() -> dict[LimitationCode, str]:
    """Each limitation code with its meaning, in the order the list has them."""
    return typing.cast(
        "dict[LimitationCode, str]", _meanings("LimitationCode", LimitationCode)
    )


@functools.cache
def _meanings(name: str, literal: object) -> dict[str, str]:
    """The comment above each code of the `Literal` that `results.<name>` assigns.

    A code without a comment, or every code when the source isn't available,
    has an empty meaning.
    """
    codes = typing.get_args(literal)
    found: dict[str, str] = {}
    try:
        source = inspect.getsource(results)
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (OSError, TypeError, tokenize.TokenError):
        return dict.fromkeys(codes, "")
    # The comments above the code that follows them, inside the brackets that
    # follow `<name> =` at the start of a line.
    depth = 0
    inside = False
    pending: list[str] = []
    for position, token in enumerate(tokens):
        if not inside:
            if (
                token.type == tokenize.NAME
                and token.string == name
                and token.start[1] == 0
                and tokens[position + 1].string == "="
            ):
                inside = True
            continue
        if token.type == tokenize.OP and token.string == "[":
            depth += 1
        elif token.type == tokenize.OP and token.string == "]":
            depth -= 1
            if depth == 0:
                break
        elif token.type == tokenize.COMMENT:
            pending.append(token.string.lstrip("# ").rstrip())
        elif token.type == tokenize.STRING and depth == 1:
            found[token.string.strip("\"'")] = " ".join(pending)
            pending = []
    return {code: found.get(code, "") for code in codes}
