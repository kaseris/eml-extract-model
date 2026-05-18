import pytest

from eml_extract_model.errors import (
    EmlExtractError,
    ConfigurationError,
    InputError,
    EmptyInputError,
    ClassificationError,
    UnrecognisedLabelError,
    LLMError,
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMConnectionError,
    DocumentIntelligenceError,
    DocIntelAuthError,
    DocIntelConnectionError,
    DocIntelUnsupportedFormatError,
)


class TestHierarchy:
    def test_base_is_exception(self):
        assert issubclass(EmlExtractError, Exception)

    def test_configuration_error(self):
        assert issubclass(ConfigurationError, EmlExtractError)

    def test_input_error(self):
        assert issubclass(InputError, EmlExtractError)

    def test_empty_input_error(self):
        assert issubclass(EmptyInputError, InputError)

    def test_classification_error(self):
        assert issubclass(ClassificationError, EmlExtractError)

    def test_unrecognised_label_error(self):
        assert issubclass(UnrecognisedLabelError, ClassificationError)

    def test_llm_error(self):
        assert issubclass(LLMError, EmlExtractError)

    def test_llm_auth_error(self):
        assert issubclass(LLMAuthError, LLMError)

    def test_llm_rate_limit_error(self):
        assert issubclass(LLMRateLimitError, LLMError)

    def test_llm_timeout_error(self):
        assert issubclass(LLMTimeoutError, LLMError)

    def test_llm_connection_error(self):
        assert issubclass(LLMConnectionError, LLMError)

    def test_doc_intel_error(self):
        assert issubclass(DocumentIntelligenceError, EmlExtractError)

    def test_doc_intel_auth_error(self):
        assert issubclass(DocIntelAuthError, DocumentIntelligenceError)

    def test_doc_intel_connection_error(self):
        assert issubclass(DocIntelConnectionError, DocumentIntelligenceError)

    def test_doc_intel_unsupported_format_error(self):
        assert issubclass(DocIntelUnsupportedFormatError, DocumentIntelligenceError)


class TestAttributes:
    def test_empty_input_code(self):
        assert EmptyInputError.code == "empty_input"

    def test_empty_input_http_status(self):
        assert EmptyInputError.http_status_hint == 422

    def test_unrecognised_label_code(self):
        assert UnrecognisedLabelError.code == "unrecognised_label"

    def test_unrecognised_label_http_status(self):
        assert UnrecognisedLabelError.http_status_hint == 502

    def test_llm_rate_limit_http_status(self):
        assert LLMRateLimitError.http_status_hint == 429

    def test_llm_timeout_http_status(self):
        assert LLMTimeoutError.http_status_hint == 504

    def test_configuration_error_http_status(self):
        assert ConfigurationError.http_status_hint == 500


class TestRaisability:
    def test_empty_input_error_is_raisable(self):
        with pytest.raises(EmptyInputError):
            raise EmptyInputError()

    def test_empty_input_caught_as_input_error(self):
        with pytest.raises(InputError):
            raise EmptyInputError()

    def test_unrecognised_label_carries_message(self):
        exc = UnrecognisedLabelError("LLM returned 'garbage'")
        assert "garbage" in str(exc)

    def test_llm_auth_caught_as_llm_error(self):
        with pytest.raises(LLMError):
            raise LLMAuthError()

    def test_llm_errors_caught_as_eml_extract_error(self):
        with pytest.raises(EmlExtractError):
            raise LLMConnectionError()
