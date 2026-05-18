import pytest

from eml_extract_model.schemas.definitions import ExtractedField, IDCardExtractionResult


class TestExtractedField:
    def test_defaults(self):
        f = ExtractedField()
        assert f.value is None
        assert f.confidence == 0.0

    def test_with_value_and_confidence(self):
        f = ExtractedField(value="01/01/2030", confidence=0.95)
        assert f.value == "01/01/2030"
        assert f.confidence == 0.95

    def test_value_without_confidence_defaults_to_zero(self):
        f = ExtractedField(value="Jane")
        assert f.confidence == 0.0


class TestIDCardExtractionResult:
    def test_all_fields_default_to_empty_extracted_field(self):
        result = IDCardExtractionResult()
        for field_name in ("first_name", "last_name", "date_of_birth", "expiration_date", "sex", "height"):
            f = getattr(result, field_name)
            assert isinstance(f, ExtractedField)
            assert f.value is None
            assert f.confidence == 0.0

    def test_populated_result(self):
        result = IDCardExtractionResult(
            first_name=ExtractedField(value="Jane", confidence=0.99),
            last_name=ExtractedField(value="Smith", confidence=0.99),
            date_of_birth=ExtractedField(value="03/22/1990", confidence=0.95),
            expiration_date=ExtractedField(value="06/30/2028", confidence=0.9),
            sex=ExtractedField(value="F", confidence=0.99),
            height=ExtractedField(value="165 cm", confidence=0.8),
        )
        assert result.first_name.value == "Jane"
        assert result.last_name.value == "Smith"
        assert result.date_of_birth.value == "03/22/1990"
        assert result.expiration_date.value == "06/30/2028"
        assert result.sex.value == "F"
        assert result.height.value == "165 cm"

    def test_partial_result_missing_fields_are_none(self):
        result = IDCardExtractionResult(
            first_name=ExtractedField(value="Jane", confidence=0.99),
            last_name=ExtractedField(value="Smith", confidence=0.99),
        )
        assert result.date_of_birth.value is None
        assert result.height.value is None
