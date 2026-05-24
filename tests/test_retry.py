import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from eml_extract_model.core.gpt_chain import GPTChain
from eml_extract_model.extraction.attachment_chain import AttachmentChain
from eml_extract_model.extraction.chain import ExtractionChain
from eml_extract_model.errors import LLMAuthError, LLMRateLimitError
from eml_extract_model.schemas.definitions import (
    ExtractedField,
    GPTClassificationResponse,
    IDCardExtractionResult,
)


class _FakeRateLimitError(openai.RateLimitError):
    def __init__(self): pass

class _FakeAuthError(openai.AuthenticationError):
    def __init__(self): pass


_SUCCESS_CLASSIFICATION = GPTClassificationResponse(label='cancellation', confidence=0.9)
_SUCCESS_EXTRACTION = IDCardExtractionResult(
    first_name=ExtractedField(value='Jane', confidence=0.99),
    last_name=ExtractedField(value='Smith', confidence=0.99),
)


# -- Factories ----------------------------------------------------------------

def _make_gpt_chain() -> GPTChain:
    with patch('eml_extract_model.core.gpt_chain.ChatOpenAI'):
        chain = GPTChain(prompt=MagicMock(), invoke_key='email_body', model='gpt-4o-mini')
    return chain


def _make_attachment_chain() -> AttachmentChain:
    with patch('eml_extract_model.extraction.attachment_chain.ChatOpenAI'):
        chain = AttachmentChain(
            prompt=MagicMock(), invoke_key='attachment_content', model='gpt-4o-mini'
        )
    return chain


def _make_extraction_chain() -> ExtractionChain:
    with patch('eml_extract_model.extraction.chain.ChatOpenAI'):
        chain = ExtractionChain(
            prompt=MagicMock(),
            invoke_key='id_card_content',
            output_schema=IDCardExtractionResult,
            model='gpt-4o-mini',
        )
    return chain


# -- GPTChain -----------------------------------------------------------------

class TestGPTChainRetry:
    async def test_retries_twice_then_succeeds(self):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=[
            _FakeRateLimitError(),
            _FakeRateLimitError(),
            _SUCCESS_CLASSIFICATION,
        ])
        result = await chain.run('cancel my policy')
        assert result.label == 'cancellation'
        assert chain._chain.ainvoke.call_count == 3

    async def test_exhausts_retries_raises_rate_limit_error(self):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 3

    async def test_auth_error_not_retried(self):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with pytest.raises(LLMAuthError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 1

    async def test_rate_limit_error_reraised_not_wrapped(self):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        exc = None
        with pytest.raises(LLMRateLimitError) as exc_info:
            await chain.run('text')
        assert type(exc_info.value) is LLMRateLimitError


# -- AttachmentChain ----------------------------------------------------------

class TestAttachmentChainRetry:
    async def test_retries_twice_then_succeeds(self):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=[
            _FakeRateLimitError(),
            _FakeRateLimitError(),
            _SUCCESS_CLASSIFICATION,
        ])
        result = await chain.run('cancellation notice')
        assert result.label == 'cancellation'
        assert chain._chain.ainvoke.call_count == 3

    async def test_exhausts_retries_raises_rate_limit_error(self):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 3

    async def test_auth_error_not_retried(self):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with pytest.raises(LLMAuthError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 1

    async def test_rate_limit_error_reraised_not_wrapped(self):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError) as exc_info:
            await chain.run('text')
        assert type(exc_info.value) is LLMRateLimitError


# -- ExtractionChain ----------------------------------------------------------

class TestExtractionChainRetry:
    async def test_retries_twice_then_succeeds(self):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=[
            _FakeRateLimitError(),
            _FakeRateLimitError(),
            _SUCCESS_EXTRACTION,
        ])
        result = await chain.run('DRIVER LICENSE JANE SMITH')
        assert result.first_name.value == 'Jane'
        assert chain._chain.ainvoke.call_count == 3

    async def test_exhausts_retries_raises_rate_limit_error(self):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 3

    async def test_auth_error_not_retried(self):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with pytest.raises(LLMAuthError):
            await chain.run('text')
        assert chain._chain.ainvoke.call_count == 1

    async def test_rate_limit_error_reraised_not_wrapped(self):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError) as exc_info:
            await chain.run('text')
        assert type(exc_info.value) is LLMRateLimitError
