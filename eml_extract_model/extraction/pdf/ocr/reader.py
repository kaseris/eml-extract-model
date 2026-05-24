import logging
import time

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
            t0 = time.perf_counter()
            poller = self._client.begin_analyze_document(
                model_id=settings.DOC_INTEL_LAYOUT,
                body=AnalyzeDocumentRequest(bytes_source=bytes_source),
            )
            result: AnalyzeResult = poller.result()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        except ClientAuthenticationError as exc:
            logger.error('doc_intelligence_ocr: authentication error', exc_info=True)
            raise OCRAuthError(
                'Document Intelligence authentication failed. Check DOC_INTEL_API_KEY.'
            ) from exc
        except ServiceRequestError as exc:
            logger.error('doc_intelligence_ocr: connection error', exc_info=True)
            raise OCRConnectionError(
                'Could not reach the Document Intelligence endpoint.'
            ) from exc
        except HttpResponseError as exc:
            if exc.status_code == _UNSUPPORTED_FORMAT_STATUS:
                logger.error('doc_intelligence_ocr: unsupported format', exc_info=True)
                raise OCRUnsupportedFormatError(
                    'Document format is not supported for OCR extraction.'
                ) from exc
            logger.error('doc_intelligence_ocr: http response error', exc_info=True)
            raise OCRError(
                f'Document Intelligence returned an error: {exc}'
            ) from exc
        except Exception as exc:
            logger.error('doc_intelligence_ocr: unexpected error', exc_info=True)
            raise OCRError(
                f'Unexpected error during OCR: {exc}'
            ) from exc

        ocr_response = OCRResponse.model_validate(result.as_dict())
        logger.info(
            'DocumentIntelligenceOCR result: %d chars %d pages elapsed_ms=%d',
            len(ocr_response.content),
            len(ocr_response.pages),
            elapsed_ms,
        )
        return ocr_response
