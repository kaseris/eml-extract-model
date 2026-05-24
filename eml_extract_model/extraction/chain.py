import logging
import time
from typing import Type

import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import settings
from ..errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..retry import make_retry_on_rate_limit

logger = logging.getLogger(__name__)
_retry = make_retry_on_rate_limit('extraction_chain', logger)


class ExtractionChain:
    """Structured field-extraction chain for arbitrary Pydantic output schemas.

    Counterpart to GPTChain: no label validation, returns the caller-supplied
    Pydantic model directly. Use this for extraction tasks where the output is
    a set of named fields rather than a classification label.
    """

    def __init__(
        self,
        prompt: ChatPromptTemplate,
        invoke_key: str,
        output_schema: Type[BaseModel],
        model: str,
    ) -> None:
        llm = ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)
        self._chain = prompt | llm.with_structured_output(output_schema)
        self._invoke_key = invoke_key
        self._model = model
        self._output_schema = output_schema

    @_retry
    async def run(self, text: str) -> BaseModel:
        logger.info(
            'extraction_chain invoke: model=%s invoke_key=%s schema=%s',
            self._model,
            self._invoke_key,
            self._output_schema.__name__,
        )
        try:
            t0 = time.perf_counter()
            result = self._output_schema.model_validate(
                await self._chain.ainvoke({self._invoke_key: text})
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        except openai.AuthenticationError as exc:
            logger.error('extraction_chain: authentication error', exc_info=True)
            raise LLMAuthError() from exc
        except openai.RateLimitError as exc:
            logger.error('extraction_chain: rate limit exceeded', exc_info=True)
            raise LLMRateLimitError() from exc
        except openai.APITimeoutError as exc:
            logger.error('extraction_chain: request timed out', exc_info=True)
            raise LLMTimeoutError() from exc
        except openai.APIConnectionError as exc:
            logger.error('extraction_chain: connection error', exc_info=True)
            raise LLMConnectionError() from exc
        except openai.APIError as exc:
            logger.error('extraction_chain: api error', exc_info=True)
            raise LLMError() from exc

        logger.info(
            'extraction_chain result: schema=%s elapsed_ms=%d',
            self._output_schema.__name__, elapsed_ms,
        )
        return result
