import logging
from typing import List

import fitz

from ....errors import PDFParsingError
from ....schemas.definitions import PageImage, PyMuPDFPage, PyMuPDFResponse, TextBlock

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
                pages = []
                for i, page in enumerate(doc):
                    image_info = page.get_image_info(xrefs=True)
                    ratio = self._image_area_ratio(page, image_info)
                    is_scanned = ratio >= self._image_area_threshold
                    logger.debug(
                        'PyMuPDFReader page %d: image_ratio=%.3f threshold=%.2f is_scanned=%s',
                        i + 1, ratio, self._image_area_threshold, is_scanned,
                    )
                    pages.append(PyMuPDFPage(
                        page_number=i + 1,
                        text=page.get_text('text') or '',
                        is_scanned=is_scanned,
                        blocks=self._extract_blocks(page),
                        images=self._extract_images(doc, image_info),
                    ))
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

    def _image_area_ratio(self, page: fitz.Page, image_info: list) -> float:
        """Return the fraction of the page area covered by raster images, capped at 1.0."""
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return 0.0
        image_area = sum(
            (img['bbox'][2] - img['bbox'][0]) * (img['bbox'][3] - img['bbox'][1])
            for img in image_info
        )
        return min(image_area / page_area, 1.0)

    def _extract_blocks(self, page: fitz.Page) -> List[TextBlock]:
        """Return positioned, non-empty text blocks for reading-order merges."""
        blocks = []
        for block in page.get_text('blocks'):
            x0, y0, x1, y1, text, _block_no, block_type = block
            if block_type != 0:  # 0 == text block; 1 == image block
                continue
            stripped = text.strip()
            if stripped:
                blocks.append(TextBlock(bbox=(x0, y0, x1, y1), text=stripped))
        return blocks

    def _extract_images(self, doc: fitz.Document, image_info: list) -> List[PageImage]:
        """Extract embedded raster images (bbox + raw bytes) for downstream OCR."""
        images = []
        for img in image_info:
            xref = img.get('xref', 0)
            if not xref:
                continue
            extracted = doc.extract_image(xref)
            images.append(PageImage(
                bbox=img['bbox'],
                image_bytes=extracted['image'],
                ext=extracted['ext'],
            ))
        return images
