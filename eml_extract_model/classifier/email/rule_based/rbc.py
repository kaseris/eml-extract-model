import logging

from .registry import RULES
from ....config import settings
from ....errors import EmptyInputError
from ....schemas.definitions import ClassificationResult

logger = logging.getLogger(__name__)


class RuleBasedClassifier:
    def __call__(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            raise EmptyInputError()
        for rule in RULES:
            for pattern in rule.patterns:
                if pattern.search(text):
                    logger.info(
                        'rule match: label=%s pattern=%s',
                        rule.label.value,
                        pattern.pattern,
                    )
                    return ClassificationResult(
                        label=rule.label.value,
                        confidence=settings.MATCH_CONFIDENCE,
                    )
        logger.info('rule match: no pattern matched')
        return ClassificationResult(label="", confidence=settings.NO_MATCH_CONFIDENCE)
