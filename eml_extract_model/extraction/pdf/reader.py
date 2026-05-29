import datetime
import logging
from pathlib import Path
from typing import Optional

from ...config import settings
from ...errors import OCRImageTooLargeError, UnsupportedAttachmentError
from ...schemas.definitions import PageImage, PDFReadResult, PyMuPDFPage, PyMuPDFResponse
from .ocr.reader import DocumentIntelligenceOCR
from .pymupdf.reader import PyMuPDFReader
from .synthesis import assemble_image_pdf

logger = logging.getLogger(__name__)


class PDFReader:
    """Orchestrates attachment text extraction by routing on file extension.

    PDFs: runs PyMuPDF first (sync, no I/O).  Each page is evaluated
    independently using the image-area heuristic in PyMuPDFReader.  If all
    pages are clean text, PyMuPDF output is returned as-is (no OCR call).

    If any page is image-dominant, the embedded images from those pages are
    extracted and packed into a single synthetic PDF (one image per page) and
    sent to OCR in one batch.  Each OCR result is mapped back to its source
    image, and per page the native text blocks and image OCR text are merged in
    reading order (by bounding box) — so OCR'd text lands where the image sat
    while the page's own text is preserved.  If a scanned page yields no
    extractable images, the reader falls back to OCR-ing the whole document and
    stitching scanned pages page-by-page.

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

        # Collect embedded images from scanned pages, preserving document order.
        # ``refs`` records which (page, image-index) each collected image came
        # from so OCR results can be mapped back to their exact source position.
        refs: list[tuple[int, int]] = []
        images: list[PageImage] = []
        for page in pymupdf_result.pages:
            if not page.is_scanned:
                continue
            for idx, image in enumerate(page.images):
                if image.image_bytes:
                    refs.append((page.page_number, idx))
                    images.append(image)

        if not images:
            logger.info(
                'PDFReader: scanned pages have no extractable images — '
                'falling back to whole-document OCR'
            )
            return await self._whole_document_ocr(bytes_source, pymupdf_result)

        logger.info(
            'PDFReader: OCR-ing %d embedded image(s) from scanned pages in one batch',
            len(images),
        )
        ocr_text_by_ref, ocr_result = await self._ocr_images(refs, images)

        stitched = [
            self._merge_page(page, ocr_text_by_ref) for page in pymupdf_result.pages
        ]

        logger.info('PDFReader result: per-image positional stitch succeeded (used_ocr=True)')
        return PDFReadResult(
            text='\n'.join(stitched),
            used_ocr=True,
            pymupdf_result=pymupdf_result,
            ocr_result=ocr_result,
        )

    def _merge_page(
        self, page: PyMuPDFPage, ocr_text_by_ref: dict[tuple[int, int], str]
    ) -> str:
        """Merge a page's native text blocks with its image OCR text in reading order.

        Non-scanned pages are returned verbatim.  For scanned pages, each text
        block and each image (carrying its OCR text) is ordered by its bounding
        box — top-to-bottom, then left-to-right — so OCR'd image text lands at
        the position the image occupied on the page.
        """
        if not page.is_scanned:
            return page.text

        items = [(b.bbox[1], b.bbox[0], b.text) for b in page.blocks]
        for idx, image in enumerate(page.images):
            text = ocr_text_by_ref.get((page.page_number, idx), '')
            items.append((image.bbox[1], image.bbox[0], text))

        items.sort(key=lambda item: (item[0], item[1]))
        return '\n'.join(text for _, _, text in items if text)

    async def _ocr_images(
        self,
        refs: list[tuple[int, int]],
        images: list[PageImage],
    ) -> tuple[dict[tuple[int, int], str], object]:
        """Attempt a single batch OCR call; fall back to per-image on size errors."""
        ocr_pdf = assemble_image_pdf(images)
        try:
            ocr_result = await self._get_ocr()(ocr_pdf)
            ocr_text_by_ref = {
                ref: ocr_page.content
                for ref, ocr_page in zip(refs, ocr_result.pages)
            }
            return ocr_text_by_ref, ocr_result
        except OCRImageTooLargeError:
            logger.warning(
                'PDFReader: batch OCR payload too large (%d bytes) — '
                'retrying per-image and skipping oversized ones',
                len(ocr_pdf),
            )
            return await self._ocr_images_individually(refs, images), None

    async def _ocr_images_individually(
        self,
        refs: list[tuple[int, int]],
        images: list[PageImage],
    ) -> dict[tuple[int, int], str]:
        """OCR each image in its own call; save and skip any that are still too large."""
        ocr_text_by_ref: dict[tuple[int, int], str] = {}
        for ref, image in zip(refs, images):
            single_pdf = assemble_image_pdf([image])
            try:
                ocr_result = await self._get_ocr()(single_pdf)
                if ocr_result.pages:
                    ocr_text_by_ref[ref] = ocr_result.pages[0].content
            except OCRImageTooLargeError:
                saved_path = self._save_debug_image(image, ref)
                logger.warning(
                    'PDFReader: image (page=%d, idx=%d) still too large after downscaling '
                    '— skipped; saved to %s',
                    ref[0], ref[1], saved_path,
                )
        return ocr_text_by_ref

    def _save_debug_image(self, image: PageImage, ref: tuple[int, int]) -> Path:
        """Write image bytes to the debug directory and return the saved path."""
        debug_dir = settings.OCR_IMAGE_DEBUG_DIR
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'oversized_page{ref[0]}_img{ref[1]}_{timestamp}.{image.ext}'
        path = debug_dir / filename
        path.write_bytes(image.image_bytes)
        logger.info('PDFReader: oversized image saved to %s', path)
        return path

    async def _whole_document_ocr(
        self, bytes_source: bytes, pymupdf_result: PyMuPDFResponse
    ) -> PDFReadResult:
        """Send the entire document to OCR and stitch page-by-page.

        Used when scanned pages are detected but no embedded images can be
        extracted from them — OCR text replaces scanned pages, PyMuPDF text is
        kept for the rest.
        """
        ocr_result = await self._get_ocr()(bytes_source)
        ocr_by_page = {p.pageNumber: p.content for p in ocr_result.pages}
        stitched = [
            ocr_by_page.get(page.page_number, '') if page.is_scanned else page.text
            for page in pymupdf_result.pages
        ]
        logger.info('PDFReader result: whole-document hybrid stitch succeeded (used_ocr=True)')
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
