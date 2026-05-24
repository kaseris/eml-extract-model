import logging

from .prompts import ATTACHMENT_CLASSIFICATION_PROMPT
from ....config import settings
from ....extraction import AttachmentChain
from ....errors import EmptyInputError
from ....schemas.definitions import ClassificationResult

logger = logging.getLogger(__name__)


class GPTBasedAttachmentClassifier:
    def __init__(self, model: str = settings.CAPABLE_MODEL) -> None:
        self._chain = AttachmentChain(ATTACHMENT_CLASSIFICATION_PROMPT, 'attachment_content', model)
        logger.info('GPTBasedAttachmentClassifier initialised: model=%s', model)

    async def __call__(self, content: str) -> ClassificationResult:
        if not content or not content.strip():
            raise EmptyInputError()
        logger.info('GPTBasedAttachmentClassifier called: %d chars', len(content))
        return await self._chain.run(content)
