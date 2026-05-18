import re

CANCEL_VERBS: list[str] = [
    r"cancel\w*",
    r"terminat\w*",
    r"discontinu\w*",
    r"unsubscribe",
    r"end",
    r"stop",
    r"quit",
    r"close",
    r"remov\w*",
]

SUBJECT_NOUNS: list[str] = [
    "subscription",
    "contract",
    "account",
    "service",
    "membership",
    "plan",
    "agreement",
    "policy",
    "invoice",
    "billing",
    "cancellation",
]

STRONG_KEYWORDS: list[str] = [
    "cancel",
    "cancellation",
    "unsubscribe",
    "terminate",
    "discontinue",
    r"opt.?out",
    "remove me",
    r"delete.?account",
    r"close.?account",
]


def build_simple_pattern(keywords: list[str] = STRONG_KEYWORDS) -> re.Pattern:
    alternation = "|".join(keywords)
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def build_context_pattern(
    verbs: list[str] = CANCEL_VERBS,
    nouns: list[str] = SUBJECT_NOUNS,
    proximity: int = 60,
) -> re.Pattern:
    v = "|".join(verbs)
    n = "|".join(nouns)
    return re.compile(
        rf"\b({v})\b.{{0,{proximity}}}\b({n})\b"
        rf"|\b({n})\b.{{0,{proximity}}}\b({v})\b",
        re.IGNORECASE | re.DOTALL,
    )


CANCELLATION_PATTERN = build_simple_pattern()
CANCELLATION_CONTEXT_PATTERN = build_context_pattern()
