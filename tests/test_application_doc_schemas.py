from eml_extract_model.schemas.definitions import ApplicationDocumentExtractionResult, ExtractedField


class TestApplicationDocumentExtractionResult:
    def test_all_fields_default_to_empty_extracted_field(self):
        result = ApplicationDocumentExtractionResult()
        for field_name in (
            'policy_number',
            'applicant_name',
            'application_date',
            'coverage_type',
            'premium_amount',
            'agent_name',
        ):
            f = getattr(result, field_name)
            assert isinstance(f, ExtractedField)
            assert f.value is None
            assert f.confidence == 0.0

    def test_populated_result(self):
        result = ApplicationDocumentExtractionResult(
            policy_number=ExtractedField(value='POL-2025-001234', confidence=0.99),
            applicant_name=ExtractedField(value='Mary Davis', confidence=0.97),
            application_date=ExtractedField(value='05/17/2025', confidence=0.95),
            coverage_type=ExtractedField(value='Auto', confidence=0.98),
            premium_amount=ExtractedField(value='1200.00', confidence=0.96),
            agent_name=ExtractedField(value='Robert Wilson', confidence=0.94),
        )
        assert result.policy_number.value == 'POL-2025-001234'
        assert result.applicant_name.value == 'Mary Davis'
        assert result.application_date.value == '05/17/2025'
        assert result.coverage_type.value == 'Auto'
        assert result.premium_amount.value == '1200.00'
        assert result.agent_name.value == 'Robert Wilson'

    def test_partial_result_missing_fields_are_none(self):
        result = ApplicationDocumentExtractionResult(
            applicant_name=ExtractedField(value='Mary Davis', confidence=0.97),
        )
        assert result.policy_number.value is None
        assert result.application_date.value is None
        assert result.agent_name.value is None
