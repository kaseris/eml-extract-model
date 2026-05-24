import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from eml_extract_model.core.gpt_chain import GPTChain
from eml_extract_model.extraction.attachment_chain import AttachmentChain
from eml_extract_model.extraction.chain import ExtractionChain
from eml_extract_model.extraction.id_card.extractor import IDCardExtractor
from eml_extract_model.extraction.application_doc.extractor import ApplicationDocumentExtractor
from eml_extract_model.classifier.email.gpt_based.gbc import GPTBasedClassifier
from eml_extract_model.classifier.attachment.gpt_based.gbc import GPTBasedAttachmentClassifier
from eml_extract_model.errors import (
    LLMAuthError,
    LLMRateLimitError,
)
from eml_extract_model.schemas.definitions import (
    GPTClassificationResponse,
    ExtractedField,
    IDCardExtractionResult,
    ApplicationDocumentExtractionResult,
)


# -- Fake openai exceptions that bypass complex constructors ------------------

class _FakeRateLimitError(openai.RateLimitError):
    def __init__(self): pass

class _FakeAuthError(openai.AuthenticationError):
    def __init__(self): pass


# -- Factory helpers ----------------------------------------------------------

def _make_gpt_chain(label='cancellation', confidence=0.9) -> GPTChain:
    with patch('eml_extract_model.core.gpt_chain.ChatOpenAI'):
        chain = GPTChain(prompt=MagicMock(), invoke_key='email_body', model='gpt-4o-mini')
    mock_response = GPTClassificationResponse(label=label, confidence=confidence)
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)
    chain._chain = mock_runnable
    return chain


def _make_attachment_chain(label='policy_issuance', confidence=0.9) -> AttachmentChain:
    with patch('eml_extract_model.extraction.attachment_chain.ChatOpenAI'):
        chain = AttachmentChain(
            prompt=MagicMock(), invoke_key='attachment_content', model='gpt-4o-mini'
        )
    mock_response = GPTClassificationResponse(label=label, confidence=confidence)
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_response)
    chain._chain = mock_runnable
    return chain


def _make_extraction_chain() -> ExtractionChain:
    with patch('eml_extract_model.extraction.chain.ChatOpenAI'):
        chain = ExtractionChain(
            prompt=MagicMock(),
            invoke_key='id_card_content',
            output_schema=IDCardExtractionResult,
            model='gpt-4o-mini',
        )
    mock_result = IDCardExtractionResult(
        first_name=ExtractedField(value='Jane', confidence=0.99),
        last_name=ExtractedField(value='Smith', confidence=0.99),
    )
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_result)
    chain._chain = mock_runnable
    return chain


# -- 1. Latency logging (elapsed_ms) -----------------------------------------

class TestLatencyLogging:
    async def test_gpt_chain_result_log_includes_elapsed_ms(self, caplog):
        chain = _make_gpt_chain()
        with caplog.at_level(logging.DEBUG):
            await chain.run('some email text')
        assert any('elapsed_ms' in m for m in caplog.messages)

    async def test_attachment_chain_result_log_includes_elapsed_ms(self, caplog):
        chain = _make_attachment_chain()
        with caplog.at_level(logging.DEBUG):
            await chain.run('attachment text')
        assert any('elapsed_ms' in m for m in caplog.messages)

    async def test_extraction_chain_result_log_includes_elapsed_ms(self, caplog):
        chain = _make_extraction_chain()
        with caplog.at_level(logging.DEBUG):
            await chain.run('id card text')
        assert any('elapsed_ms' in m for m in caplog.messages)

    async def test_gpt_chain_elapsed_ms_is_integer(self, caplog):
        chain = _make_gpt_chain()
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        result_msg = next(
            (m for m in caplog.messages if 'result' in m and 'elapsed_ms' in m), None
        )
        assert result_msg is not None
        # Extract the elapsed_ms value — must be a valid integer
        import re
        match = re.search(r'elapsed_ms=(\d+)', result_msg)
        assert match is not None
        assert int(match.group(1)) >= 0


# -- 2. Error logging before re-raise ----------------------------------------

class TestErrorLogging:
    async def test_gpt_chain_logs_error_on_rate_limit(self, caplog):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    async def test_gpt_chain_logs_error_on_auth(self, caplog):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeAuthError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMAuthError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    async def test_gpt_chain_error_record_has_exc_info(self, caplog):
        chain = _make_gpt_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records[0].exc_info is not None

    async def test_attachment_chain_logs_error_on_rate_limit(self, caplog):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    async def test_attachment_chain_error_record_has_exc_info(self, caplog):
        chain = _make_attachment_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records[0].exc_info is not None

    async def test_extraction_chain_logs_error_on_rate_limit(self, caplog):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    async def test_extraction_chain_error_record_has_exc_info(self, caplog):
        chain = _make_extraction_chain()
        chain._chain.ainvoke = AsyncMock(side_effect=_FakeRateLimitError())
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(LLMRateLimitError):
                await chain.run('text')
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records[0].exc_info is not None


# -- 3. Full field confidence logging in extractors ---------------------------

class TestExtractorFullFieldLogging:
    async def test_id_card_extractor_logs_all_six_fields(self, caplog):
        result = IDCardExtractionResult(
            first_name=ExtractedField(value='Jane', confidence=0.99),
            last_name=ExtractedField(value='Smith', confidence=0.99),
            date_of_birth=ExtractedField(value='03/22/1990', confidence=0.95),
            expiration_date=ExtractedField(value='06/30/2028', confidence=0.9),
            sex=ExtractedField(value='F', confidence=0.99),
            height=ExtractedField(value='165 cm', confidence=0.8),
        )
        with patch('eml_extract_model.extraction.id_card.extractor.ExtractionChain') as MockChain:
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(return_value=result)
            MockChain.return_value = mock_chain
            extractor = IDCardExtractor()
        with caplog.at_level(logging.DEBUG):
            await extractor('DRIVER LICENSE\nJANE SMITH')
        result_msg = next(
            (m for m in caplog.messages if 'IDCardExtractor result' in m), None
        )
        assert result_msg is not None
        for field_name in ('first_name', 'last_name', 'date_of_birth', 'expiration_date', 'sex', 'height'):
            assert field_name in result_msg, f'{field_name!r} missing from log: {result_msg!r}'

    async def test_id_card_extractor_logs_confidence_scores(self, caplog):
        result = IDCardExtractionResult(
            first_name=ExtractedField(value='Jane', confidence=0.99),
            last_name=ExtractedField(value='Smith', confidence=0.99),
            date_of_birth=ExtractedField(value=None, confidence=0.0),
            expiration_date=ExtractedField(value='06/30/2028', confidence=0.9),
            sex=ExtractedField(value='F', confidence=0.99),
            height=ExtractedField(value=None, confidence=0.0),
        )
        with patch('eml_extract_model.extraction.id_card.extractor.ExtractionChain') as MockChain:
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(return_value=result)
            MockChain.return_value = mock_chain
            extractor = IDCardExtractor()
        with caplog.at_level(logging.DEBUG):
            await extractor('DRIVER LICENSE\nJANE SMITH')
        result_msg = next(
            (m for m in caplog.messages if 'IDCardExtractor result' in m), None
        )
        assert result_msg is not None
        assert '0.00' in result_msg or '0.0' in result_msg

    async def test_application_doc_extractor_logs_all_six_fields(self, caplog):
        result = ApplicationDocumentExtractionResult(
            policy_number=ExtractedField(value='POL-001', confidence=0.99),
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
        with caplog.at_level(logging.DEBUG):
            await extractor('APPLICATION FOR INSURANCE')
        result_msg = next(
            (m for m in caplog.messages if 'ApplicationDocumentExtractor result' in m), None
        )
        assert result_msg is not None
        for field_name in (
            'policy_number', 'applicant_name', 'application_date',
            'coverage_type', 'premium_amount', 'agent_name',
        ):
            assert field_name in result_msg, f'{field_name!r} missing from log: {result_msg!r}'

    async def test_application_doc_extractor_logs_confidence_scores(self, caplog):
        result = ApplicationDocumentExtractionResult(
            applicant_name=ExtractedField(value='Mary Davis', confidence=0.97),
        )
        with patch('eml_extract_model.extraction.application_doc.extractor.ExtractionChain') as MockChain:
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(return_value=result)
            MockChain.return_value = mock_chain
            extractor = ApplicationDocumentExtractor()
        with caplog.at_level(logging.DEBUG):
            await extractor('APPLICATION FOR INSURANCE')
        result_msg = next(
            (m for m in caplog.messages if 'ApplicationDocumentExtractor result' in m), None
        )
        assert result_msg is not None
        assert '0.00' in result_msg or '0.0' in result_msg


# -- 4. Low-confidence warning ------------------------------------------------

class TestLowConfidenceWarning:
    async def test_gpt_chain_warns_on_low_confidence(self, caplog):
        chain = _make_gpt_chain(label='cancellation', confidence=0.3)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        gpt_warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'eml_extract_model.core.gpt_chain'
        ]
        assert len(gpt_warning_records) >= 1

    async def test_gpt_chain_no_warning_on_high_confidence(self, caplog):
        chain = _make_gpt_chain(label='cancellation', confidence=0.9)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        gpt_warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'eml_extract_model.core.gpt_chain'
        ]
        assert len(gpt_warning_records) == 0

    async def test_gpt_chain_no_warning_at_threshold(self, caplog):
        chain = _make_gpt_chain(label='cancellation', confidence=0.5)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        gpt_warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'eml_extract_model.core.gpt_chain'
        ]
        assert len(gpt_warning_records) == 0

    async def test_attachment_chain_warns_on_low_confidence(self, caplog):
        chain = _make_attachment_chain(label='policy_issuance', confidence=0.2)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'eml_extract_model.extraction.attachment_chain'
        ]
        assert len(warning_records) >= 1

    async def test_attachment_chain_no_warning_on_high_confidence(self, caplog):
        chain = _make_attachment_chain(label='policy_issuance', confidence=0.8)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'eml_extract_model.extraction.attachment_chain'
        ]
        assert len(warning_records) == 0

    async def test_low_confidence_warning_includes_label_and_confidence(self, caplog):
        chain = _make_gpt_chain(label='cancellation', confidence=0.1)
        with caplog.at_level(logging.DEBUG):
            await chain.run('some text')
        warning_msgs = [
            m for r, m in zip(caplog.records, caplog.messages)
            if r.levelno == logging.WARNING and r.name == 'eml_extract_model.core.gpt_chain'
        ]
        assert len(warning_msgs) >= 1
        assert 'cancellation' in warning_msgs[0]
        assert '0.10' in warning_msgs[0]


# -- 5. Input length at classifier entry --------------------------------------

class TestClassifierEntryCharCount:
    async def test_gpt_based_classifier_logs_char_count(self, caplog):
        with patch('eml_extract_model.classifier.email.gpt_based.gbc.GPTChain') as MockGPTChain:
            from eml_extract_model.schemas.definitions import ClassificationResult
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(
                return_value=ClassificationResult(label='cancellation', confidence=0.9)
            )
            MockGPTChain.return_value = mock_chain
            clf = GPTBasedClassifier()
        text = 'Please cancel my policy today'
        with caplog.at_level(logging.DEBUG):
            await clf(text)
        entry_msgs = [
            m for r, m in zip(caplog.records, caplog.messages)
            if r.name == 'eml_extract_model.classifier.email.gpt_based.gbc'
            and 'called' in m
        ]
        assert len(entry_msgs) >= 1
        assert str(len(text)) in entry_msgs[0]

    async def test_gpt_based_classifier_char_count_matches_input(self, caplog):
        with patch('eml_extract_model.classifier.email.gpt_based.gbc.GPTChain') as MockGPTChain:
            from eml_extract_model.schemas.definitions import ClassificationResult
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(
                return_value=ClassificationResult(label='cancellation', confidence=0.9)
            )
            MockGPTChain.return_value = mock_chain
            clf = GPTBasedClassifier()
        text = 'x' * 42
        with caplog.at_level(logging.DEBUG):
            await clf(text)
        entry_msgs = [
            m for r, m in zip(caplog.records, caplog.messages)
            if r.name == 'eml_extract_model.classifier.email.gpt_based.gbc'
            and 'called' in m
        ]
        assert '42' in entry_msgs[0]

    async def test_gpt_based_attachment_classifier_logs_char_count(self, caplog):
        with patch('eml_extract_model.classifier.attachment.gpt_based.gbc.AttachmentChain') as MockChain:
            from eml_extract_model.schemas.definitions import ClassificationResult
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(
                return_value=ClassificationResult(label='policy_issuance', confidence=0.9)
            )
            MockChain.return_value = mock_chain
            clf = GPTBasedAttachmentClassifier()
        text = 'Certificate of insurance document'
        with caplog.at_level(logging.DEBUG):
            await clf(text)
        entry_msgs = [
            m for r, m in zip(caplog.records, caplog.messages)
            if r.name == 'eml_extract_model.classifier.attachment.gpt_based.gbc'
            and 'called' in m
        ]
        assert len(entry_msgs) >= 1
        assert str(len(text)) in entry_msgs[0]

    async def test_gpt_based_attachment_classifier_char_count_matches_input(self, caplog):
        with patch('eml_extract_model.classifier.attachment.gpt_based.gbc.AttachmentChain') as MockChain:
            from eml_extract_model.schemas.definitions import ClassificationResult
            mock_chain = MagicMock()
            mock_chain.run = AsyncMock(
                return_value=ClassificationResult(label='policy_issuance', confidence=0.9)
            )
            MockChain.return_value = mock_chain
            clf = GPTBasedAttachmentClassifier()
        text = 'y' * 100
        with caplog.at_level(logging.DEBUG):
            await clf(text)
        entry_msgs = [
            m for r, m in zip(caplog.records, caplog.messages)
            if r.name == 'eml_extract_model.classifier.attachment.gpt_based.gbc'
            and 'called' in m
        ]
        assert '100' in entry_msgs[0]
