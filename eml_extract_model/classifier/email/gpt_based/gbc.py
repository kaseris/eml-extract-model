import logging

from .prompts import EMAIL_CLASSIFICATION_PROMPT
from ....config import settings
from ....core.gpt_chain import GPTChain
from ....errors import EmptyInputError
from ....schemas.definitions import ClassificationResult

logger = logging.getLogger(__name__)


class GPTBasedClassifier:
    def __init__(self, model: str = settings.CAPABLE_MODEL) -> None:
        self._chain = GPTChain(EMAIL_CLASSIFICATION_PROMPT, 'email_body', model)
        logger.info('GPTBasedClassifier initialised: model=%s', model)

    async def __call__(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            raise EmptyInputError()
        logger.info('GPTBasedClassifier called: %d chars', len(text))
        return await self._chain.run(text)
