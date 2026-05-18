import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from eml_extract_model.core.gpt_chain import GPTChain
from eml_extract_model.errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    UnrecognisedLabelError,
)
from eml_extract_model.schemas.definitions import ClassificationResult, GPTClassificationResponse


# Minimal subclasses that bypass the complex openai exception constructors
# while still satisfying isinstance checks in GPTChain's except clauses.
class _FakeAuthError(openai.AuthenticationError):
    def __init__(self): pass

class _FakeRateLimitError(openai.RateLimitError):
    def __init__(self): pass

class _FakeTimeoutError(openai.APITimeoutError):
    def __init__(self): pass

class _FakeConnectionError(openai.APIConnectionError):
    def __init__(self): pass

class _FakeAPIError(openai.APIError):
    def __init__(self): pass


def _make_chain(label: str = "cancellation", confidence: float = 0.9) -> GPTChain:
    """Construct a GPTChain with its LangChain internals fully mocked out."""
    with patch("eml_extract_model.core.gpt_chain.ChatOpenAI"):
        chain = GPTChain(prompt=MagicMock(), invoke_key="email_body", model="gpt-4o-mini")
    mock_response = GPTClassificationResponse(label=label, confidence=confidence)
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)
    chain._chain = mock_runnable
    return chain


class TestHappyPath:
    async def test_returns_classification_result(self):
        chain = _make_chain(label="cancellation", confidence=0.9)
        result = await chain.run("Please cancel my policy")
        assert isinstance(result, ClassificationResult)

    async def test_label_and_confidence_passed_through(self):
        chain = _make_chain(label="cancellation", confidence=0.75)
        result = await chain.run("cancel this")
        assert result.label == "cancellation"
        assert result.confidence == 0.75

    async def test_policy_issuance_label_accepted(self):
        chain = _make_chain(label="policy_issuance", confidence=0.88)
        result = await chain.run("Your policy has been issued")
        assert result.label == "policy_issuance"

    async def test_invokes_chain_with_correct_key(self):
        chain = _make_chain()
        mock_invoke = AsyncMock(
            return_value=GPTClassificationResponse(label="cancellation", confidence=1.0)
        )
        chain._chain = MagicMock()
        chain._chain.ainvoke = mock_invoke
        await chain.run("some text")
        mock_invoke.assert_awaited_once_with({"email_body": "some text"})


class TestLabelValidation:
    async def test_unrecognised_label_raises(self):
        chain = _make_chain(label="garbage_label", confidence=0.9)
        with pytest.raises(UnrecognisedLabelError):
            await chain.run("some text")

    async def test_unrecognised_label_error_message(self):
        chain = _make_chain(label="unknown", confidence=0.5)
        with pytest.raises(UnrecognisedLabelError, match="unknown"):
            await chain.run("text")


class TestErrorMapping:
    async def test_auth_error_maps_to_llm_auth_error(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with pytest.raises(LLMAuthError):
            await chain.run("text")

    async def test_rate_limit_error_maps(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with pytest.raises(LLMRateLimitError):
            await chain.run("text")

    async def test_timeout_error_maps(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeTimeoutError())
        with pytest.raises(LLMTimeoutError):
            await chain.run("text")

    async def test_connection_error_maps(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeConnectionError())
        with pytest.raises(LLMConnectionError):
            await chain.run("text")

    async def test_generic_api_error_maps_to_llm_error(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAPIError())
        with pytest.raises(LLMError):
            await chain.run("text")

    async def test_llm_auth_error_is_llm_error(self):
        chain = _make_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with pytest.raises(LLMError):
            await chain.run("text")
