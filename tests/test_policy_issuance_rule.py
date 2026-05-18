import pytest

from eml_extract_model.errors import (
    ApplicantNameMismatchError,
    MissingApplicationDocumentError,
    MultipleApplicationDocumentsError,
    PolicyIssuanceMissingIDCardError,
    PolicyIssuanceMultipleIDCardsError,
)
from eml_extract_model.rules.policy_issuance import (
    validate_applicant_name_match,
    validate_policy_issuance_attachments,
)
from eml_extract_model.schemas.definitions import (
    ApplicationDocumentExtractionResult,
    ClassificationResult,
    ExtractedField,
    IDCardExtractionResult,
)


def _id_card(label="id_card", confidence=0.95):
    return ClassificationResult(label=label, confidence=confidence)


def _app_doc(confidence=0.92):
    return ClassificationResult(label="application_document", confidence=confidence)


class TestValidatePolicyIssuanceAttachments:
    def test_empty_list_raises_missing_id_card(self):
        with pytest.raises(PolicyIssuanceMissingIDCardError):
            validate_policy_issuance_attachments([])

    def test_no_id_card_raises_missing_id_card(self):
        with pytest.raises(PolicyIssuanceMissingIDCardError):
            validate_policy_issuance_attachments([_app_doc()])

    def test_no_application_document_raises_missing_app_doc(self):
        with pytest.raises(MissingApplicationDocumentError):
            validate_policy_issuance_attachments([_id_card()])

    def test_exactly_one_of_each_returns_none(self):
        result = validate_policy_issuance_attachments([_id_card(), _app_doc()])
        assert result is None

    def test_order_does_not_matter(self):
        result = validate_policy_issuance_attachments([_app_doc(), _id_card()])
        assert result is None

    def test_two_id_cards_raises_multiple_id_cards(self):
        with pytest.raises(PolicyIssuanceMultipleIDCardsError):
            validate_policy_issuance_attachments([_id_card(), _id_card(), _app_doc()])

    def test_two_app_docs_raises_multiple_app_docs(self):
        with pytest.raises(MultipleApplicationDocumentsError):
            validate_policy_issuance_attachments([_id_card(), _app_doc(), _app_doc()])

    def test_extra_unrelated_attachments_do_not_affect_validation(self):
        other = ClassificationResult(label="cancellation", confidence=0.5)
        result = validate_policy_issuance_attachments([_id_card(), _app_doc(), other])
        assert result is None

    def test_id_card_missing_when_only_other_docs_present(self):
        other = ClassificationResult(label="cancellation", confidence=0.5)
        with pytest.raises(PolicyIssuanceMissingIDCardError):
            validate_policy_issuance_attachments([_app_doc(), other])


class TestValidateApplicantNameMatch:
    def _id_card_result(self, first="Mary", last="Davis"):
        return IDCardExtractionResult(
            first_name=ExtractedField(value=first, confidence=0.99),
            last_name=ExtractedField(value=last, confidence=0.99),
        )

    def _app_doc_result(self, name="Mary Davis"):
        return ApplicationDocumentExtractionResult(
            applicant_name=ExtractedField(value=name, confidence=0.97),
        )

    def test_matching_names_returns_none(self):
        result = validate_applicant_name_match(
            self._id_card_result(), self._app_doc_result()
        )
        assert result is None

    def test_case_insensitive_match(self):
        result = validate_applicant_name_match(
            self._id_card_result(first="MARY", last="DAVIS"),
            self._app_doc_result("mary davis"),
        )
        assert result is None

    def test_reversed_order_in_app_doc_matches(self):
        result = validate_applicant_name_match(
            self._id_card_result(),
            self._app_doc_result("Davis, Mary"),
        )
        assert result is None

    def test_mismatched_name_raises_error(self):
        with pytest.raises(ApplicantNameMismatchError):
            validate_applicant_name_match(
                self._id_card_result(first="Mary", last="Davis"),
                self._app_doc_result("John Smith"),
            )

    def test_mismatched_first_name_raises_error(self):
        with pytest.raises(ApplicantNameMismatchError):
            validate_applicant_name_match(
                self._id_card_result(first="Mary", last="Davis"),
                self._app_doc_result("Jane Davis"),
            )

    def test_id_card_first_name_none_raises_error(self):
        id_card = IDCardExtractionResult(
            first_name=ExtractedField(value=None, confidence=0.0),
            last_name=ExtractedField(value="Davis", confidence=0.99),
        )
        with pytest.raises(ApplicantNameMismatchError):
            validate_applicant_name_match(id_card, self._app_doc_result())

    def test_id_card_last_name_none_raises_error(self):
        id_card = IDCardExtractionResult(
            first_name=ExtractedField(value="Mary", confidence=0.99),
            last_name=ExtractedField(value=None, confidence=0.0),
        )
        with pytest.raises(ApplicantNameMismatchError):
            validate_applicant_name_match(id_card, self._app_doc_result())

    def test_app_doc_applicant_name_none_raises_error(self):
        app_doc = ApplicationDocumentExtractionResult(
            applicant_name=ExtractedField(value=None, confidence=0.0),
        )
        with pytest.raises(ApplicantNameMismatchError):
            validate_applicant_name_match(self._id_card_result(), app_doc)
