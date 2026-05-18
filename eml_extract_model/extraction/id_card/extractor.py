import logging

from .prompts import ID_CARD_EXTRACTION_PROMPT
from ...config import settings
from ...errors import EmptyInputError
from ...schemas.definitions import IDCardExtractionResult
from ..chain import ExtractionChain

logger = logging.getLogger(__name__)


class IDCardExtractor:
    def __init__(self, model: str = settings.CAPABLE_MODEL) -> None:
        self._chain = ExtractionChain(
            ID_CARD_EXTRACTION_PROMPT,
            'id_card_content',
            IDCardExtractionResult,
            model,
        )
        logger.info('IDCardExtractor initialised: model=%s', model)

    async def __call__(self, content: str) -> IDCardExtractionResult:
        if not content or not content.strip():
            raise EmptyInputError()
        logger.info('IDCardExtractor called')
        result = await self._chain.run(content)
        logger.info(
            'IDCardExtractor result: first_name=%r last_name=%r',
            result.first_name.value,
            result.last_name.value,
        )
        return result
