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
            'ApplicationDocumentExtractor result: '
            'policy_number=%r(%.2f) applicant_name=%r(%.2f) '
            'application_date=%r(%.2f) coverage_type=%r(%.2f) '
            'premium_amount=%r(%.2f) agent_name=%r(%.2f)',
            result.policy_number.value, result.policy_number.confidence,
            result.applicant_name.value, result.applicant_name.confidence,
            result.application_date.value, result.application_date.confidence,
            result.coverage_type.value, result.coverage_type.confidence,
            result.premium_amount.value, result.premium_amount.confidence,
            result.agent_name.value, result.agent_name.confidence,
        )
        return result
