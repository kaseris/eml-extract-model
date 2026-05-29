class EmlExtractError(Exception):
    """Base exception for all eml-extract-model errors."""

    code: str = "eml_extract_error"
    description: str = "An unexpected error occurred in eml-extract-model."
    http_status_hint: int = 500


class ConfigurationError(EmlExtractError):
    """Required env var is missing or a settings value is invalid at startup."""

    code = "configuration_error"
    description = "Library is misconfigured. Check required environment variables."
    http_status_hint = 500


class InputError(EmlExtractError):
    """Input passed to a classifier is invalid."""

    code = "input_error"
    description = "Invalid input provided to the classifier."
    http_status_hint = 422


class EmptyInputError(InputError):
    """Text or attachment content is empty or whitespace-only."""

    code = "empty_input"
    description = "Input text must not be empty."
    http_status_hint = 422


class UnsupportedAttachmentError(InputError):
    """File extension is not in the supported attachment extensions."""

    code = "unsupported_attachment"
    description = "Attachment file type is not supported."
    http_status_hint = 422


class ClassificationError(EmlExtractError):
    """A classifier could not produce a valid result."""

    code = "classification_error"
    description = "Classification failed."
    http_status_hint = 500


class UnrecognisedLabelError(ClassificationError):
    """The LLM returned a label that is not in EMailCategories."""

    code = "unrecognised_label"
    description = "LLM returned an unrecognised classification label."
    http_status_hint = 502


class LLMError(EmlExtractError):
    """Base for all OpenAI / LangChain call failures."""

    code = "llm_error"
    description = "LLM call failed."
    http_status_hint = 502


class LLMAuthError(LLMError):
    """OpenAI API key is invalid or rejected."""

    code = "llm_auth_error"
    description = "OpenAI authentication failed. Check OPENAI_API_KEY."
    http_status_hint = 502


class LLMRateLimitError(LLMError):
    """OpenAI rate limit or quota exceeded."""

    code = "llm_rate_limit"
    description = "OpenAI rate limit exceeded. Retry after a short delay."
    http_status_hint = 429


class LLMTimeoutError(LLMError):
    """OpenAI call timed out."""

    code = "llm_timeout"
    description = "LLM call timed out."
    http_status_hint = 504


class LLMConnectionError(LLMError):
    """Network-level failure reaching the OpenAI API."""

    code = "llm_connection_error"
    description = "Could not reach the OpenAI API."
    http_status_hint = 502


class DocumentIntelligenceError(EmlExtractError):
    """Base for all Azure Document Intelligence failures."""

    code = "doc_intel_error"
    description = "Document Intelligence call failed."
    http_status_hint = 502


class DocIntelAuthError(DocumentIntelligenceError):
    """Azure Document Intelligence credentials are invalid or rejected."""

    code = "doc_intel_auth_error"
    description = (
        "Document Intelligence authentication failed. Check DOC_INTEL_API_KEY."
    )
    http_status_hint = 502


class DocIntelConnectionError(DocumentIntelligenceError):
    """Network-level failure reaching the Azure Document Intelligence endpoint."""

    code = "doc_intel_connection_error"
    description = "Could not reach the Document Intelligence endpoint."
    http_status_hint = 502


class DocIntelUnsupportedFormatError(DocumentIntelligenceError):
    """Attachment format is not supported by Document Intelligence."""

    code = "doc_intel_unsupported_format"
    description = "Attachment format is not supported for extraction."
    http_status_hint = 422


class PDFExtractionError(EmlExtractError):
    """Base for all PDF extraction failures."""

    code = "pdf_extraction_error"
    description = "PDF extraction failed."
    http_status_hint = 500


class PDFParsingError(PDFExtractionError):
    """PDF bytes are malformed or cannot be opened by PyMuPDF."""

    code = "pdf_parsing_error"
    description = "Could not parse PDF. The file may be malformed or corrupted."
    http_status_hint = 500


class OCRError(PDFExtractionError):
    """Base for all OCR (Document Intelligence) call failures during PDF reading."""

    code = "ocr_error"
    description = "OCR call failed."
    http_status_hint = 502


class OCRAuthError(OCRError):
    """Azure Document Intelligence credentials are invalid or rejected."""

    code = "ocr_auth_error"
    description = "OCR authentication failed. Check DOC_INTEL_API_KEY."
    http_status_hint = 502


class OCRConnectionError(OCRError):
    """Network-level failure reaching the Azure Document Intelligence endpoint."""

    code = "ocr_connection_error"
    description = "Could not reach the OCR endpoint."
    http_status_hint = 502


class OCRImageTooLargeError(OCRError):
    """Azure Document Intelligence rejected the image because it exceeds the size limit."""

    code = "ocr_image_too_large"
    description = "OCR input image exceeds the provider's maximum allowed size."
    http_status_hint = 413


class OCRUnsupportedFormatError(PDFExtractionError):
    """Document format is not supported by the OCR provider."""

    code = "ocr_unsupported_format"
    description = "Document format is not supported for OCR extraction."
    http_status_hint = 422


class BusinessRuleError(EmlExtractError):
    """A domain business rule was violated."""

    code = "business_rule_error"
    description = "A business rule was violated."
    http_status_hint = 422


class MissingIDCardAttachmentError(BusinessRuleError):
    """A cancellation email has no attachment classified as id_card."""

    code = "missing_id_card_attachment"
    description = "A cancellation email must have exactly one ID card attachment; none found."
    http_status_hint = 422


class MultipleIDCardAttachmentsError(BusinessRuleError):
    """A cancellation email has more than one attachment classified as id_card."""

    code = "multiple_id_card_attachments"
    description = "A cancellation email must have exactly one ID card attachment; multiple found."
    http_status_hint = 422


class PolicyIssuanceMissingIDCardError(BusinessRuleError):
    """A policy issuance email has no attachment classified as id_card."""

    code = "policy_issuance_missing_id_card"
    description = "A policy issuance email must have exactly one ID card attachment; none found."
    http_status_hint = 422


class PolicyIssuanceMultipleIDCardsError(BusinessRuleError):
    """A policy issuance email has more than one attachment classified as id_card."""

    code = "policy_issuance_multiple_id_cards"
    description = "A policy issuance email must have exactly one ID card attachment; multiple found."
    http_status_hint = 422


class MissingApplicationDocumentError(BusinessRuleError):
    """A policy issuance email has no attachment classified as application_document."""

    code = "missing_application_document"
    description = "A policy issuance email must have exactly one application document attachment; none found."
    http_status_hint = 422


class MultipleApplicationDocumentsError(BusinessRuleError):
    """A policy issuance email has more than one attachment classified as application_document."""

    code = "multiple_application_documents"
    description = "A policy issuance email must have exactly one application document attachment; multiple found."
    http_status_hint = 422


class ApplicantNameMismatchError(BusinessRuleError):
    """The applicant name on the application document does not match the ID card."""

    code = "applicant_name_mismatch"
    description = "The applicant name on the application document does not match the name on the ID card."
    http_status_hint = 422
