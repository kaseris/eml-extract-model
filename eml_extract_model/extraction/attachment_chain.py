import logging
import time

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings
from ..errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    UnrecognisedLabelError,
)
from ..schemas.categories import AttachmentCategories, EMailCategories
from ..schemas.definitions import ClassificationResult, GPTClassificationResponse

logger = logging.getLogger(__name__)

# Union of email and attachment-specific labels — all valid outcomes for an attachment.
_VALID_LABELS = {e.value for e in EMailCategories} | {e.value for e in AttachmentCategories}


class AttachmentChain:
    """Classification chain scoped to attachment labels.

    Validates against the union of EMailCategories and AttachmentCategories so
    that an attachment can be classified as cancellation, policy_issuance, or
    id_card. GPTChain is left untouched.
    """

    def __init__(self, prompt: ChatPromptTemplate, invoke_key: str, model: str) -> None:
        llm = ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)
        self._chain = prompt | llm.with_structured_output(GPTClassificationResponse)
        self._invoke_key = invoke_key
        self._model = model

    async def run(self, text: str) -> ClassificationResult:
        logger.info('attachment_chain invoke: model=%s invoke_key=%s', self._model, self._invoke_key)
        try:
            t0 = time.perf_counter()
            result = GPTClassificationResponse.model_validate(
                await self._chain.ainvoke({self._invoke_key: text})
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        except openai.AuthenticationError as exc:
            logger.error('attachment_chain: authentication error', exc_info=True)
            raise LLMAuthError() from exc
        except openai.RateLimitError as exc:
            logger.error('attachment_chain: rate limit exceeded', exc_info=True)
            raise LLMRateLimitError() from exc
        except openai.APITimeoutError as exc:
            logger.error('attachment_chain: request timed out', exc_info=True)
            raise LLMTimeoutError() from exc
        except openai.APIConnectionError as exc:
            logger.error('attachment_chain: connection error', exc_info=True)
            raise LLMConnectionError() from exc
        except openai.APIError as exc:
            logger.error('attachment_chain: api error', exc_info=True)
            raise LLMError() from exc

        if result.label not in _VALID_LABELS:
            raise UnrecognisedLabelError(
                f'LLM returned {result.label!r}; valid labels: {", ".join(sorted(_VALID_LABELS))}'
            )
        logger.info(
            'attachment_chain result: label=%r confidence=%.2f elapsed_ms=%d',
            result.label, result.confidence, elapsed_ms,
        )
        if result.confidence < settings.LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                'attachment_chain: low confidence result label=%r confidence=%.2f',
                result.label, result.confidence,
            )
        return ClassificationResult(label=result.label, confidence=result.confidence)
