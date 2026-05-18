import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eml_extract_model.classifier.email.gpt_based.gbc import GPTBasedClassifier
from eml_extract_model.errors import EmptyInputError
from eml_extract_model.schemas.definitions import ClassificationResult


def _make_classifier(label: str = "cancellation", confidence: float = 0.9) -> tuple:
    """Return (classifier, mock_chain) with GPTChain fully mocked."""
    expected = ClassificationResult(label=label, confidence=confidence)
    with patch("eml_extract_model.classifier.email.gpt_based.gbc.GPTChain") as MockGPTChain:
        mock_chain = MagicMock()
        mock_chain.run = AsyncMock(return_value=expected)
        MockGPTChain.return_value = mock_chain
        clf = GPTBasedClassifier()
    return clf, mock_chain


class TestEmptyInput:
    async def test_empty_string_raises(self):
        clf, _ = _make_classifier()
        with pytest.raises(EmptyInputError):
            await clf("")

    async def test_whitespace_only_raises(self):
        clf, _ = _make_classifier()
        with pytest.raises(EmptyInputError):
            await clf("   ")

    async def test_newline_only_raises(self):
        clf, _ = _make_classifier()
        with pytest.raises(EmptyInputError):
            await clf("\n")


class TestDelegation:
    async def test_delegates_to_chain_run(self):
        clf, mock_chain = _make_classifier(label="cancellation", confidence=0.95)
        result = await clf("I want to cancel my subscription")
        mock_chain.run.assert_awaited_once_with("I want to cancel my subscription")

    async def test_returns_chain_result(self):
        clf, _ = _make_classifier(label="cancellation", confidence=0.95)
        result = await clf("Cancel please")
        assert isinstance(result, ClassificationResult)
        assert result.label == "cancellation"
        assert result.confidence == 0.95

    async def test_policy_issuance_result_passes_through(self):
        clf, _ = _make_classifier(label="policy_issuance", confidence=0.88)
        result = await clf("Your policy has been issued")
        assert result.label == "policy_issuance"

    async def test_chain_constructed_with_email_invoke_key(self):
        with patch("eml_extract_model.classifier.email.gpt_based.gbc.GPTChain") as MockGPTChain:
            MockGPTChain.return_value = MagicMock()
            GPTBasedClassifier()
            _, call_kwargs = MockGPTChain.call_args
            args = MockGPTChain.call_args[0]
        assert args[1] == "email_body"
