import logging

from .prompts import APPLICATION_DOC_EXTRACTION_PROMPT
from ...config import settings
from ...errors import EmptyInputError
from ...schemas.definitions import ApplicationDocumentExtractionResult
from ..chain import ExtractionChain

logger = logging.getLogger(__name__)


class ApplicationDocumentExtractor:
    def __init__(self, model: str = settings.CAPABLE_MODEL) -> None:
        self._chain = ExtractionChain(
            APPLICATION_DOC_EXTRACTION_PROMPT,
            'application_doc_content',
            ApplicationDocumentExtractionResult,
            model,
        )
        logger.info('ApplicationDocumentExtractor initialised: model=%s', model)

    async def __call__(self, content: str) -> ApplicationDocumentExtractionResult:
        if not content or not content.strip():
            raise EmptyInputError()
        logger.info('ApplicationDocumentExtractor called')
        result = await self._chain.run(content)
        logger.info(
            'ApplicationDocumentExtractor result: applicant_name=%r policy_number=%r',
            result.applicant_name.value,
            result.policy_number.value,
        )
        return result
