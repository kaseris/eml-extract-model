import pytest
from pydantic import ValidationError

from eml_extract_model.schemas.categories import EMailCategories
from eml_extract_model.schemas.definitions import ClassificationResult, GPTClassificationResponse


class TestEMailCategories:
    def test_cancellation_value(self):
        assert EMailCategories.CANCELLATION.value == "cancellation"

    def test_policy_issuance_value(self):
        assert EMailCategories.POLICY_ISSUANCE.value == "policy_issuance"

    def test_is_str_subclass(self):
        assert isinstance(EMailCategories.CANCELLATION, str)

    def test_all_members(self):
        assert {e.value for e in EMailCategories} == {"cancellation", "policy_issuance"}


class TestClassificationResult:
    def test_valid_construction(self):
        r = ClassificationResult(label="cancellation", confidence=1.0)
        assert r.label == "cancellation"
        assert r.confidence == 1.0

    def test_empty_label_no_match(self):
        r = ClassificationResult(label="", confidence=0.0)
        assert r.label == ""
        assert r.confidence == 0.0

    def test_confidence_coerced_to_float(self):
        r = ClassificationResult(label="x", confidence=1)
        assert isinstance(r.confidence, float)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResult(label="x", confidence="not-a-number")

    def test_missing_label_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResult(confidence=0.5)

    def test_missing_confidence_raises(self):
        with pytest.raises(ValidationError):
            ClassificationResult(label="x")


class TestGPTClassificationResponse:
    def test_valid_construction(self):
        r = GPTClassificationResponse(label="policy_issuance", confidence=0.85)
        assert r.label == "policy_issuance"
        assert r.confidence == 0.85

    def test_invalid_missing_fields(self):
        with pytest.raises(ValidationError):
            GPTClassificationResponse(label="only-label")
