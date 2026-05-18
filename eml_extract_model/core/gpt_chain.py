import logging

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..config import settings
from ..errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMError,
    UnrecognisedLabelError,
)
from ..schemas.categories import EMailCategories
from ..schemas.definitions import ClassificationResult, GPTClassificationResponse

logger = logging.getLogger(__name__)

_VALID_LABELS = {e.value for e in EMailCategories}


class GPTChain:
    def __init__(self, prompt: ChatPromptTemplate, invoke_key: str, model: str) -> None:
        llm = ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)
        self._chain = prompt | llm.with_structured_output(GPTClassificationResponse)
        self._invoke_key = invoke_key
        self._model = model

    async def run(self, text: str) -> ClassificationResult:
        logger.info('gpt_chain invoke: model=%s invoke_key=%s', self._model, self._invoke_key)
        try:
            result = GPTClassificationResponse.model_validate(
                await self._chain.ainvoke({self._invoke_key: text})
            )
        except openai.AuthenticationError as exc:
            raise LLMAuthError() from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError() from exc
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError() from exc
        except openai.APIConnectionError as exc:
            raise LLMConnectionError() from exc
        except openai.APIError as exc:
            raise LLMError() from exc

        if result.label not in _VALID_LABELS:
            raise UnrecognisedLabelError(
                f'LLM returned {result.label!r}; valid labels: {", ".join(sorted(_VALID_LABELS))}'
            )
        logger.info('gpt_chain result: label=%r confidence=%.2f', result.label, result.confidence)
        return ClassificationResult(label=result.label, confidence=result.confidence)
