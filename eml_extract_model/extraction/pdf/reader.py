import logging
from pathlib import Path
from typing import Optional

from ...config import settings
from ...errors import UnsupportedAttachmentError
from ...schemas.definitions import PDFReadResult
from .ocr.reader import DocumentIntelligenceOCR
from .pymupdf.reader import PyMuPDFReader

logger = logging.getLogger(__name__)


class PDFReader:
    """Orchestrates attachment text extraction by routing on file extension.

    PDFs: runs PyMuPDF first (sync, no I/O).  Each page is evaluated
    independently using the image-area heuristic in PyMuPDFReader.  If all
    pages are clean text, PyMuPDF output is returned as-is (no OCR call).  If
    any page is detected as image-dominant, the full document is sent to OCR
    once; the final text is then stitched page-by-page — OCR text for scanned
    pages, PyMuPDF text for normal pages.

    Images: skips PyMuPDF entirely and goes straight to OCR.

    Supported extensions are governed by ``settings.SUPPORTED_PDF_EXTENSIONS``
    and ``settings.SUPPORTED_IMAGE_EXTENSIONS``. Anything else raises
    ``UnsupportedAttachmentError``.

    Dependency injection is supported for both sub-readers; the OCR client is
    constructed lazily so it is only created when actually needed.
    """

    def __init__(
        self,
        pymupdf_reader: Optional[PyMuPDFReader] = None,
        ocr: Optional[DocumentIntelligenceOCR] = None,
    ) -> None:
        self._pymupdf_reader = pymupdf_reader or PyMuPDFReader()
        self._ocr = ocr
        logger.info('PDFReader initialised')

    async def __call__(self, bytes_source: bytes, filename: str) -> PDFReadResult:
        logger.info('PDFReader called: filename=%r %d bytes', filename, len(bytes_source))

        ext = Path(filename).suffix.lower()

        if ext in settings.SUPPORTED_IMAGE_EXTENSIONS:
            return await self._read_image(bytes_source)

        if ext in settings.SUPPORTED_PDF_EXTENSIONS:
            return await self._read_pdf(bytes_source)

        supported = (
            settings.SUPPORTED_PDF_EXTENSIONS | settings.SUPPORTED_IMAGE_EXTENSIONS
        )
        raise UnsupportedAttachmentError(
            f'Unsupported file extension {ext!r}. Supported: {sorted(supported)}'
        )

    async def _read_image(self, bytes_source: bytes) -> PDFReadResult:
        logger.info('PDFReader: image attachment — routing directly to OCR')
        ocr_result = await self._get_ocr()(bytes_source)
        logger.info('PDFReader result: image OCR succeeded')
        return PDFReadResult(
            text=ocr_result.content,
            used_ocr=True,
            ocr_result=ocr_result,
        )

    async def _read_pdf(self, bytes_source: bytes) -> PDFReadResult:
        pymupdf_result = self._pymupdf_reader(bytes_source)

        if not pymupdf_result.has_scanned_pages:
            logger.info('PDFReader result: text extraction succeeded (no OCR needed)')
            return PDFReadResult(
                text=pymupdf_result.content,
                used_ocr=False,
                pymupdf_result=pymupdf_result,
            )

        scanned = sum(1 for p in pymupdf_result.pages if p.is_scanned)
        logger.info(
            'PDFReader: %d/%d pages are scanned — running OCR then stitching',
            scanned,
            pymupdf_result.page_count,
        )
        ocr_result = await self._get_ocr()(bytes_source)

        ocr_by_page = {p.pageNumber: p.content for p in ocr_result.pages}
        stitched = [
            ocr_by_page.get(page.page_number, '') if page.is_scanned else page.text
            for page in pymupdf_result.pages
        ]

        logger.info('PDFReader result: hybrid stitch succeeded (used_ocr=True)')
        return PDFReadResult(
            text='\n'.join(stitched),
            used_ocr=True,
            pymupdf_result=pymupdf_result,
            ocr_result=ocr_result,
        )

    def _get_ocr(self) -> DocumentIntelligenceOCR:
        if self._ocr is None:
            self._ocr = DocumentIntelligenceOCR()
        return self._ocr
