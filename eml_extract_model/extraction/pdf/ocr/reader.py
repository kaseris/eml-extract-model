import logging

from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)

from ....config import settings
from ....errors import OCRAuthError, OCRConnectionError, OCRError, OCRUnsupportedFormatError
from ....schemas.definitions import OCRResponse
from .client import get_client

logger = logging.getLogger(__name__)

# HTTP status code returned by Document Intelligence for unsupported formats.
_UNSUPPORTED_FORMAT_STATUS = 415


class DocumentIntelligenceOCR:
    """Async wrapper around Azure Document Intelligence for OCR on documents.

    Uses the settings singleton for endpoint, API key, and layout model ID.
    Raises a specific OCR error subclass on any failure — never returns None.
    """

    def __init__(self) -> None:
        self._client = get_client(
            endpoint=settings.DOC_INTEL_ENDPOINT,
            key=settings.DOC_INTEL_API_KEY,
        )
        logger.info('DocumentIntelligenceOCR initialised')

    async def __call__(self, bytes_source: bytes) -> OCRResponse:
        logger.info('DocumentIntelligenceOCR called: %d bytes', len(bytes_source))
        try:
            poller = self._client.begin_analyze_document(
                model_id=settings.DOC_INTEL_LAYOUT,
                body=AnalyzeDocumentRequest(bytes_source=bytes_source),
            )
            result: AnalyzeResult = poller.result()
        except ClientAuthenticationError as exc:
            raise OCRAuthError(
                'Document Intelligence authentication failed. Check DOC_INTEL_API_KEY.'
            ) from exc
        except ServiceRequestError as exc:
            raise OCRConnectionError(
                'Could not reach the Document Intelligence endpoint.'
            ) from exc
        except HttpResponseError as exc:
            if exc.status_code == _UNSUPPORTED_FORMAT_STATUS:
                raise OCRUnsupportedFormatError(
                    'Document format is not supported for OCR extraction.'
                ) from exc
            raise OCRError(
                f'Document Intelligence returned an error: {exc}'
            ) from exc
        except Exception as exc:
            raise OCRError(
                f'Unexpected error during OCR: {exc}'
            ) from exc

        ocr_response = OCRResponse.model_validate(result.as_dict())
        logger.info(
            'DocumentIntelligenceOCR result: %d chars %d pages',
            len(ocr_response.content),
            len(ocr_response.pages),
        )
        return ocr_response
