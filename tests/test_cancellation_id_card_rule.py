import pytest

from eml_extract_model.errors import MissingIDCardAttachmentError, MultipleIDCardAttachmentsError
from eml_extract_model.rules.cancellation import validate_cancellation_attachments
from eml_extract_model.schemas.definitions import ClassificationResult


class TestValidateCancellationAttachments:
    def test_empty_attachment_list_raises_missing(self):
        with pytest.raises(MissingIDCardAttachmentError):
            validate_cancellation_attachments([])

    def test_no_id_card_among_attachments_raises_missing(self):
        attachments = [
            ClassificationResult(label="policy_issuance", confidence=0.9),
            ClassificationResult(label="cancellation", confidence=0.8),
        ]
        with pytest.raises(MissingIDCardAttachmentError):
            validate_cancellation_attachments(attachments)

    def test_exactly_one_id_card_returns_it(self):
        attachments = [
            ClassificationResult(label="policy_issuance", confidence=0.9),
            ClassificationResult(label="id_card", confidence=0.95),
        ]
        result = validate_cancellation_attachments(attachments)
        assert result.label == "id_card"
        assert result.confidence == 0.95

    def test_single_id_card_attachment_passes(self):
        attachments = [ClassificationResult(label="id_card", confidence=0.97)]
        result = validate_cancellation_attachments(attachments)
        assert result.label == "id_card"

    def test_two_id_cards_raises_multiple(self):
        attachments = [
            ClassificationResult(label="id_card", confidence=0.95),
            ClassificationResult(label="id_card", confidence=0.88),
        ]
        with pytest.raises(MultipleIDCardAttachmentsError):
            validate_cancellation_attachments(attachments)

    def test_three_id_cards_raises_multiple(self):
        attachments = [ClassificationResult(label="id_card", confidence=0.9)] * 3
        with pytest.raises(MultipleIDCardAttachmentsError):
            validate_cancellation_attachments(attachments)

    def test_returns_classification_result_instance(self):
        attachments = [ClassificationResult(label="id_card", confidence=0.91)]
        result = validate_cancellation_attachments(attachments)
        assert isinstance(result, ClassificationResult)
