import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eml_extract_model.classifier.attachment.gpt_based.gbc import GPTBasedAttachmentClassifier
from eml_extract_model.errors import EmptyInputError
from eml_extract_model.schemas.definitions import ClassificationResult


def _make_classifier(label: str = "policy_issuance", confidence: float = 0.9) -> tuple:
    """Return (classifier, mock_chain) with AttachmentChain fully mocked."""
    expected = ClassificationResult(label=label, confidence=confidence)
    with patch("eml_extract_model.classifier.attachment.gpt_based.gbc.AttachmentChain") as MockChain:
        mock_chain = MagicMock()
        mock_chain.run = AsyncMock(return_value=expected)
        MockChain.return_value = mock_chain
        clf = GPTBasedAttachmentClassifier()
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
        clf, mock_chain = _make_classifier(label="policy_issuance", confidence=0.92)
        await clf("This document is a certificate of insurance")
        mock_chain.run.assert_awaited_once_with("This document is a certificate of insurance")

    async def test_returns_chain_result(self):
        clf, _ = _make_classifier(label="policy_issuance", confidence=0.92)
        result = await clf("This document is a certificate of insurance")
        assert isinstance(result, ClassificationResult)
        assert result.label == "policy_issuance"
        assert result.confidence == 0.92

    async def test_cancellation_result_passes_through(self):
        clf, _ = _make_classifier(label="cancellation", confidence=0.8)
        result = await clf("Cancellation notice attached")
        assert result.label == "cancellation"

    async def test_id_card_result_passes_through(self):
        clf, _ = _make_classifier(label="id_card", confidence=0.95)
        result = await clf("DRIVER LICENSE STATE OF NEW YORK")
        assert result.label == "id_card"

    async def test_chain_constructed_with_attachment_invoke_key(self):
        with patch("eml_extract_model.classifier.attachment.gpt_based.gbc.AttachmentChain") as MockChain:
            MockChain.return_value = MagicMock()
            GPTBasedAttachmentClassifier()
            args = MockChain.call_args[0]
        assert args[1] == "attachment_content"
