import logging
from typing import List

import fitz

from ....errors import PDFParsingError
from ....schemas.definitions import PyMuPDFPage, PyMuPDFResponse

logger = logging.getLogger(__name__)


class PyMuPDFReader:
    """Extracts text from a PDF in-memory with PyMuPDF and flags scanned documents.

    A page is considered image-only when its stripped text contains fewer than
    ``per_page_min_chars`` characters. The document is considered scanned when
    at least ``scanned_page_fraction`` of its pages are image-only.
    """

    def __init__(
        self,
        per_page_min_chars: int = 20,
        scanned_page_fraction: float = 0.8,
    ) -> None:
        self._per_page_min_chars = per_page_min_chars
        self._scanned_page_fraction = scanned_page_fraction
        logger.info(
            'PyMuPDFReader initialised: per_page_min_chars=%d scanned_page_fraction=%.2f',
            per_page_min_chars,
            scanned_page_fraction,
        )

    def __call__(self, bytes_source: bytes) -> PyMuPDFResponse:
        logger.info('PyMuPDFReader called: %d bytes', len(bytes_source))
        try:
            with fitz.open(stream=bytes_source, filetype='pdf') as doc:
                pages = [
                    PyMuPDFPage(page_number=i + 1, text=page.get_text('text') or '')
                    for i, page in enumerate(doc)
                ]
                page_count = doc.page_count
        except Exception as exc:
            raise PDFParsingError(
                f'PyMuPDF could not open or parse the PDF: {exc}'
            ) from exc

        response = PyMuPDFResponse(
            pages=pages,
            page_count=page_count,
            is_scanned=self._is_scanned(pages),
        )

        logger.info(
            'PyMuPDFReader result: %d chars across %d pages (is_scanned=%s)',
            sum(p.char_count for p in pages),
            page_count,
            response.is_scanned,
        )
        return response

    def _is_scanned(self, pages: List[PyMuPDFPage]) -> bool:
        if not pages:
            return False
        image_only = sum(1 for p in pages if p.char_count < self._per_page_min_chars)
        return image_only / len(pages) >= self._scanned_page_fraction
