import logging
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

logger = logging.getLogger(__name__)


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

    async def run(self, text: str) -> BaseModel:
        logger.info(
            'extraction_chain invoke: model=%s invoke_key=%s schema=%s',
            self._model,
            self._invoke_key,
            self._output_schema.__name__,
        )
        try:
            result = self._output_schema.model_validate(
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

        logger.info('extraction_chain result: schema=%s', self._output_schema.__name__)
        return result
