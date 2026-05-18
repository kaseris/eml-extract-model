import re

ISSUANCE_VERBS: list[str] = [
    r"issu\w*",
    r"bind\w*",
    r"secure\w*",
    r"request\w*",
    r"ask\w*",
    r"submit\w*",
    r"generat\w*",
    r"creat\w*",
    r"activat\w*",
    r"confirm\w*",
    r"approv\w*",
    r"enroll\w*",
    r"initiat\w*",
    r"process\w*",
    r"prepar\w*",
    r"dispatch\w*",
    r"deliver\w*",
    r"send\w*",
    r"sent",
]

SUBJECT_NOUNS: list[str] = [
    "policy",
    "coverage",
    "certificate",
    "document",
    "declaration",
    "endorsement",
    "binder",
    "rider",
    "contract",
    "agreement",
    "insurance",
    "premium",
    "proposal",
]

STRONG_KEYWORDS: list[str] = [
    r"policy.?issuance",
    r"policy.?issued",
    r"new\b.{0,30}\bpolicy",
    r"policy.?number",
    r"certificate.?of.?insurance",
    r"declaration.?page",
    r"coverage.?confirm\w*",
    r"policy.?activat\w*",
    r"policy.?effective",
    "underwriting",
]


def build_simple_pattern(keywords: list[str] = STRONG_KEYWORDS) -> re.Pattern:
    alternation = "|".join(keywords)
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def build_context_pattern(
    verbs: list[str] = ISSUANCE_VERBS,
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


POLICY_ISSUANCE_PATTERN = build_simple_pattern()
POLICY_ISSUANCE_CONTEXT_PATTERN = build_context_pattern()
