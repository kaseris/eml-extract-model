import re
from dataclasses import dataclass

from ....schemas.categories import EMailCategories
from .patterns.cancellation import (
    CANCELLATION_PATTERN,
    CANCELLATION_CONTEXT_PATTERN,
)
from .patterns.policy_issuance import (
    POLICY_ISSUANCE_PATTERN,
    POLICY_ISSUANCE_CONTEXT_PATTERN,
)


@dataclass(frozen=True)
class CategoryRule:
    label: EMailCategories
    patterns: tuple[re.Pattern, ...]


RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        label=EMailCategories.CANCELLATION,
        patterns=(CANCELLATION_PATTERN, CANCELLATION_CONTEXT_PATTERN),
    ),
    CategoryRule(
        label=EMailCategories.POLICY_ISSUANCE,
        patterns=(POLICY_ISSUANCE_PATTERN, POLICY_ISSUANCE_CONTEXT_PATTERN),
    ),
)
