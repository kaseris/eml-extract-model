import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eml_extract_model.extraction.application_doc.extractor import ApplicationDocumentExtractor
from eml_extract_model.errors import EmptyInputError
from eml_extract_model.schemas.definitions import ApplicationDocumentExtractionResult, ExtractedField


def _make_extractor(result: ApplicationDocumentExtractionResult | None = None) -> tuple:
    """Return (extractor, mock_chain) with ExtractionChain fully mocked."""
    if result is None:
        result = ApplicationDocumentExtractionResult(
            policy_number=ExtractedField(value='POL-2025-001234', confidence=0.99),
            applicant_name=ExtractedField(value='Mary Davis', confidence=0.97),
            application_date=ExtractedField(value='05/17/2025', confidence=0.95),
            coverage_type=ExtractedField(value='Auto', confidence=0.98),
            premium_amount=ExtractedField(value='1200.00', confidence=0.96),
            agent_name=ExtractedField(value='Robert Wilson', confidence=0.94),
        )
    with patch('eml_extract_model.extraction.application_doc.extractor.ExtractionChain') as MockChain:
        mock_chain = MagicMock()
        mock_chain.run = AsyncMock(return_value=result)
        MockChain.return_value = mock_chain
        extractor = ApplicationDocumentExtractor()
    return extractor, mock_chain


class TestEmptyInput:
    async def test_empty_string_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor('')

    async def test_whitespace_only_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor('   ')

    async def test_newline_only_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor('\n')


class TestDelegation:
    async def test_delegates_to_chain_run_with_full_content(self):
        extractor, mock_chain = _make_extractor()
        content = 'APPLICATION FOR INSURANCE\nApplicant: Mary Davis\nPolicy: POL-2025-001234'
        await extractor(content)
        mock_chain.run.assert_awaited_once_with(content)

    async def test_returns_application_document_extraction_result(self):
        extractor, _ = _make_extractor()
        result = await extractor('APPLICATION FOR INSURANCE\nApplicant: Mary Davis')
        assert isinstance(result, ApplicationDocumentExtractionResult)

    async def test_all_fields_returned(self):
        extractor, _ = _make_extractor()
        result = await extractor('APPLICATION FOR INSURANCE\nApplicant: Mary Davis')
        assert result.policy_number.value == 'POL-2025-001234'
        assert result.applicant_name.value == 'Mary Davis'
        assert result.application_date.value == '05/17/2025'
        assert result.coverage_type.value == 'Auto'
        assert result.premium_amount.value == '1200.00'
        assert result.agent_name.value == 'Robert Wilson'

    async def test_missing_field_has_none_value_and_zero_confidence(self):
        partial = ApplicationDocumentExtractionResult(
            applicant_name=ExtractedField(value='Mary Davis', confidence=0.97),
        )
        extractor, _ = _make_extractor(result=partial)
        result = await extractor('APPLICATION FOR INSURANCE\nApplicant: Mary Davis')
        assert result.policy_number.value is None
        assert result.policy_number.confidence == 0.0
        assert result.agent_name.value is None
        assert result.agent_name.confidence == 0.0

    async def test_chain_constructed_with_application_doc_invoke_key(self):
        with patch('eml_extract_model.extraction.application_doc.extractor.ExtractionChain') as MockChain:
            MockChain.return_value = MagicMock()
            ApplicationDocumentExtractor()
            args = MockChain.call_args[0]
        assert args[1] == 'application_doc_content'

    async def test_chain_constructed_with_application_document_extraction_result_schema(self):
        with patch('eml_extract_model.extraction.application_doc.extractor.ExtractionChain') as MockChain:
            MockChain.return_value = MagicMock()
            ApplicationDocumentExtractor()
            args = MockChain.call_args[0]
        assert args[2] is ApplicationDocumentExtractionResult
