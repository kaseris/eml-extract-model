import logging
from typing import List

import fitz

from ....errors import PDFParsingError
from ....schemas.definitions import PyMuPDFPage, PyMuPDFResponse

logger = logging.getLogger(__name__)


class PyMuPDFReader:
    """Extracts text from a PDF in-memory with PyMuPDF and flags scanned pages.

    A page is considered scanned when the total area of its raster images
    represents at least ``image_area_threshold`` (default 10%) of the page
    area.  Only that page is flagged; pages below the threshold are treated as
    normal text pages regardless of the rest of the document.
    """

    def __init__(self, image_area_threshold: float = 0.10) -> None:
        self._image_area_threshold = image_area_threshold
        logger.info(
            'PyMuPDFReader initialised: image_area_threshold=%.2f',
            image_area_threshold,
        )

    def __call__(self, bytes_source: bytes) -> PyMuPDFResponse:
        logger.info('PyMuPDFReader called: %d bytes', len(bytes_source))
        try:
            with fitz.open(stream=bytes_source, filetype='pdf') as doc:
                pages = [
                    PyMuPDFPage(
                        page_number=i + 1,
                        text=page.get_text('text') or '',
                        is_scanned=self._page_is_scanned(page),
                    )
                    for i, page in enumerate(doc)
                ]
                page_count = doc.page_count
        except PDFParsingError:
            raise
        except Exception as exc:
            raise PDFParsingError(
                f'PyMuPDF could not open or parse the PDF: {exc}'
            ) from exc

        scanned_count = sum(1 for p in pages if p.is_scanned)
        logger.info(
            'PyMuPDFReader result: %d chars across %d pages (%d scanned)',
            sum(p.char_count for p in pages),
            page_count,
            scanned_count,
        )
        return PyMuPDFResponse(pages=pages, page_count=page_count)

    def _page_is_scanned(self, page: fitz.Page) -> bool:
        """Return True if raster images cover >= threshold of the page area."""
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return False
        image_area = sum(
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            for img in page.get_image_info()
            for bbox in [img['bbox']]
        )
        return (image_area / page_area) >= self._image_area_threshold
