import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eml_extract_model.extraction.id_card.extractor import IDCardExtractor
from eml_extract_model.errors import EmptyInputError
from eml_extract_model.schemas.definitions import ExtractedField, IDCardExtractionResult


def _make_extractor(result: IDCardExtractionResult | None = None) -> tuple:
    """Return (extractor, mock_chain) with ExtractionChain fully mocked."""
    if result is None:
        result = IDCardExtractionResult(
            first_name=ExtractedField(value="Jane", confidence=0.99),
            last_name=ExtractedField(value="Smith", confidence=0.99),
            date_of_birth=ExtractedField(value="03/22/1990", confidence=0.95),
            expiration_date=ExtractedField(value="06/30/2028", confidence=0.9),
            sex=ExtractedField(value="F", confidence=0.99),
            height=ExtractedField(value="165 cm", confidence=0.8),
        )
    with patch("eml_extract_model.extraction.id_card.extractor.ExtractionChain") as MockChain:
        mock_chain = MagicMock()
        mock_chain.run = AsyncMock(return_value=result)
        MockChain.return_value = mock_chain
        extractor = IDCardExtractor()
    return extractor, mock_chain


class TestEmptyInput:
    async def test_empty_string_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor("")

    async def test_whitespace_only_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor("   ")

    async def test_newline_only_raises(self):
        extractor, _ = _make_extractor()
        with pytest.raises(EmptyInputError):
            await extractor("\n")


class TestDelegation:
    async def test_delegates_to_chain_run_with_full_content(self):
        extractor, mock_chain = _make_extractor()
        content = "DRIVER LICENSE\nJANE SMITH\nDOB: 03/22/1990"
        await extractor(content)
        mock_chain.run.assert_awaited_once_with(content)

    async def test_returns_id_card_extraction_result(self):
        extractor, _ = _make_extractor()
        result = await extractor("DRIVER LICENSE\nJANE SMITH")
        assert isinstance(result, IDCardExtractionResult)

    async def test_all_fields_returned(self):
        extractor, _ = _make_extractor()
        result = await extractor("DRIVER LICENSE\nJANE SMITH\nDOB: 03/22/1990")
        assert result.first_name.value == "Jane"
        assert result.last_name.value == "Smith"
        assert result.expiration_date.value == "06/30/2028"
        assert result.sex.value == "F"
        assert result.height.value == "165 cm"

    async def test_missing_field_has_none_value_and_zero_confidence(self):
        partial = IDCardExtractionResult(
            first_name=ExtractedField(value="Jane", confidence=0.99),
            last_name=ExtractedField(value="Smith", confidence=0.99),
            expiration_date=ExtractedField(value="06/30/2028", confidence=0.9),
            sex=ExtractedField(value="F", confidence=0.99),
        )
        extractor, _ = _make_extractor(result=partial)
        result = await extractor("DRIVER LICENSE\nJANE SMITH")
        assert result.date_of_birth.value is None
        assert result.date_of_birth.confidence == 0.0
        assert result.height.value is None
        assert result.height.confidence == 0.0

    async def test_chain_constructed_with_id_card_invoke_key(self):
        with patch("eml_extract_model.extraction.id_card.extractor.ExtractionChain") as MockChain:
            MockChain.return_value = MagicMock()
            IDCardExtractor()
            args = MockChain.call_args[0]
        assert args[1] == "id_card_content"

    async def test_chain_constructed_with_id_card_extraction_result_schema(self):
        with patch("eml_extract_model.extraction.id_card.extractor.ExtractionChain") as MockChain:
            MockChain.return_value = MagicMock()
            IDCardExtractor()
            args = MockChain.call_args[0]
        assert args[2] is IDCardExtractionResult
