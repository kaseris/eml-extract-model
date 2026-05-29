from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eml_extract_model.errors import OCRImageTooLargeError, UnsupportedAttachmentError
from eml_extract_model.extraction.pdf.reader import PDFReader
from eml_extract_model.schemas.definitions import (
    OCRPage,
    OCRResponse,
    OCRSpan,
    PageImage,
    PDFReadResult,
    PyMuPDFPage,
    PyMuPDFResponse,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_response(*texts: str) -> PyMuPDFResponse:
    """Build a PyMuPDFResponse where every page is a clean text page."""
    pages = [
        PyMuPDFPage(page_number=i + 1, text=t, is_scanned=False)
        for i, t in enumerate(texts)
    ]
    return PyMuPDFResponse(pages=pages, page_count=len(pages))


def _scanned_response(*texts: str) -> PyMuPDFResponse:
    """Build a PyMuPDFResponse where every page is scanned (no usable text)."""
    pages = [
        PyMuPDFPage(page_number=i + 1, text=t, is_scanned=True)
        for i, t in enumerate(texts)
    ]
    return PyMuPDFResponse(pages=pages, page_count=len(pages))


def _mixed_response(flags: list[tuple[bool, str]]) -> PyMuPDFResponse:
    """Build a PyMuPDFResponse from (is_scanned, text) pairs."""
    pages = [
        PyMuPDFPage(page_number=i + 1, text=text, is_scanned=scanned)
        for i, (scanned, text) in enumerate(flags)
    ]
    return PyMuPDFResponse(pages=pages, page_count=len(pages))


def _ocr_response(*page_contents: str) -> OCRResponse:
    """Build an OCRResponse with per-page content already populated."""
    pages = [
        OCRPage(pageNumber=i + 1, content=c)
        for i, c in enumerate(page_contents)
    ]
    full_content = '\n'.join(page_contents)
    return OCRResponse(content=full_content, pages=pages)


def _make_reader(
    pymupdf_result: PyMuPDFResponse,
    ocr_result: OCRResponse | None = None,
) -> PDFReader:
    mock_pymupdf = MagicMock(return_value=pymupdf_result)
    mock_ocr = AsyncMock(return_value=ocr_result) if ocr_result is not None else None
    return PDFReader(pymupdf_reader=mock_pymupdf, ocr=mock_ocr)


# ---------------------------------------------------------------------------
# All-text PDF — no OCR needed
# ---------------------------------------------------------------------------

class TestAllTextPDF:
    async def test_returns_pymupdf_text(self):
        reader = _make_reader(_text_response('Page 1', 'Page 2'))
        result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'Page 1\nPage 2'

    async def test_used_ocr_is_false(self):
        reader = _make_reader(_text_response('text'))
        result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is False

    async def test_ocr_is_never_called(self):
        mock_pymupdf = MagicMock(return_value=_text_response('text'))
        mock_ocr = AsyncMock()
        reader = PDFReader(pymupdf_reader=mock_pymupdf, ocr=mock_ocr)
        await reader(b'fake', 'doc.pdf')
        mock_ocr.assert_not_awaited()

    async def test_pymupdf_result_stored(self):
        pymupdf_res = _text_response('text')
        reader = _make_reader(pymupdf_res)
        result = await reader(b'fake', 'doc.pdf')
        assert result.pymupdf_result is pymupdf_res

    async def test_ocr_result_is_none(self):
        reader = _make_reader(_text_response('text'))
        result = await reader(b'fake', 'doc.pdf')
        assert result.ocr_result is None


# ---------------------------------------------------------------------------
# All-scanned PDF — every page goes to OCR
# ---------------------------------------------------------------------------

class TestAllScannedPDF:
    async def test_used_ocr_is_true(self):
        reader = _make_reader(
            _scanned_response('', ''),
            _ocr_response('OCR page 1', 'OCR page 2'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is True

    async def test_text_comes_from_ocr(self):
        reader = _make_reader(
            _scanned_response('', ''),
            _ocr_response('OCR page 1', 'OCR page 2'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'OCR page 1\nOCR page 2'

    async def test_ocr_result_stored(self):
        ocr_res = _ocr_response('OCR text')
        reader = _make_reader(_scanned_response(''), ocr_res)
        result = await reader(b'fake', 'doc.pdf')
        assert result.ocr_result is ocr_res

    async def test_pymupdf_result_also_stored(self):
        pymupdf_res = _scanned_response('')
        reader = _make_reader(pymupdf_res, _ocr_response('OCR text'))
        result = await reader(b'fake', 'doc.pdf')
        assert result.pymupdf_result is pymupdf_res


# ---------------------------------------------------------------------------
# Mixed PDF — hybrid stitching
# ---------------------------------------------------------------------------

class TestMixedPDF:
    async def test_text_page_uses_pymupdf_text(self):
        # Page 1 text, page 2 scanned
        reader = _make_reader(
            _mixed_response([(False, 'PyMuPDF text'), (True, '')]),
            _ocr_response('OCR page 1', 'OCR page 2'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert 'PyMuPDF text' in result.text

    async def test_scanned_page_uses_ocr_text(self):
        reader = _make_reader(
            _mixed_response([(False, 'PyMuPDF text'), (True, '')]),
            _ocr_response('OCR page 1', 'OCR page 2'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert 'OCR page 2' in result.text

    async def test_hybrid_stitched_order(self):
        # 3 pages: text, scanned, text
        reader = _make_reader(
            _mixed_response([
                (False, 'First text'),
                (True, ''),
                (False, 'Third text'),
            ]),
            _ocr_response('OCR 1', 'OCR 2', 'OCR 3'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'First text\nOCR 2\nThird text'

    async def test_used_ocr_is_true_for_mixed(self):
        reader = _make_reader(
            _mixed_response([(False, 'text'), (True, '')]),
            _ocr_response('OCR 1', 'OCR 2'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is True

    async def test_ocr_called_exactly_once(self):
        mock_pymupdf = MagicMock(
            return_value=_mixed_response([(False, 'text'), (True, '')])
        )
        mock_ocr = AsyncMock(return_value=_ocr_response('OCR 1', 'OCR 2'))
        reader = PDFReader(pymupdf_reader=mock_pymupdf, ocr=mock_ocr)
        await reader(b'fake', 'doc.pdf')
        mock_ocr.assert_awaited_once()

    async def test_missing_ocr_page_falls_back_to_empty(self):
        # OCR returns fewer pages than expected — scanned page gets empty string
        reader = _make_reader(
            _mixed_response([(True, ''), (True, '')]),
            _ocr_response('OCR page 1'),   # only 1 page in OCR result
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'OCR page 1\n'


# ---------------------------------------------------------------------------
# Per-image positional merge — scanned pages with extractable embedded images
# ---------------------------------------------------------------------------

def _img(bbox, marker: bytes = b'imgbytes') -> PageImage:
    return PageImage(bbox=bbox, image_bytes=marker, ext='png')


def _blk(bbox, text: str) -> TextBlock:
    return TextBlock(bbox=bbox, text=text)


class TestPerImagePositionalMerge:
    async def test_ocr_text_spliced_at_image_position(self):
        # Two text blocks (top y=10, bottom y=100) with an image between (y=50).
        page = PyMuPDFPage(
            page_number=1,
            is_scanned=True,
            blocks=[_blk((0, 10, 50, 20), 'TOP'), _blk((0, 100, 50, 110), 'BOTTOM')],
            images=[_img((0, 50, 50, 60))],
        )
        reader = _make_reader(
            PyMuPDFResponse(pages=[page], page_count=1),
            _ocr_response('MIDDLE'),
        )
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-fake',
        ):
            result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'TOP\nMIDDLE\nBOTTOM'

    async def test_used_ocr_is_true(self):
        page = PyMuPDFPage(
            page_number=1, is_scanned=True,
            blocks=[_blk((0, 0, 10, 10), 'A')], images=[_img((0, 20, 10, 30))],
        )
        reader = _make_reader(PyMuPDFResponse(pages=[page], page_count=1), _ocr_response('X'))
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-fake',
        ):
            result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is True

    async def test_ocr_called_with_assembled_image_pdf(self):
        page = PyMuPDFPage(
            page_number=1, is_scanned=True,
            blocks=[], images=[_img((0, 0, 10, 10))],
        )
        mock_ocr = AsyncMock(return_value=_ocr_response('OCR'))
        reader = PDFReader(
            pymupdf_reader=MagicMock(return_value=PyMuPDFResponse(pages=[page], page_count=1)),
            ocr=mock_ocr,
        )
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-assembled',
        ):
            await reader(b'original', 'doc.pdf')
        mock_ocr.assert_awaited_once_with(b'%PDF-assembled')

    async def test_images_batched_in_single_ocr_call(self):
        # Two scanned pages, one image each — one OCR call, mapped per page.
        pages = [
            PyMuPDFPage(page_number=1, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))]),
            PyMuPDFPage(page_number=2, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))]),
        ]
        mock_ocr = AsyncMock(return_value=_ocr_response('OCR-A', 'OCR-B'))
        reader = PDFReader(
            pymupdf_reader=MagicMock(return_value=PyMuPDFResponse(pages=pages, page_count=2)),
            ocr=mock_ocr,
        )
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-fake',
        ):
            result = await reader(b'fake', 'doc.pdf')
        mock_ocr.assert_awaited_once()
        assert result.text == 'OCR-A\nOCR-B'

    async def test_text_page_kept_verbatim_in_merge(self):
        pages = [
            PyMuPDFPage(page_number=1, is_scanned=False, text='Plain text page'),
            PyMuPDFPage(page_number=2, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))]),
        ]
        reader = _make_reader(
            PyMuPDFResponse(pages=pages, page_count=2),
            _ocr_response('SCANNED OCR'),
        )
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-fake',
        ):
            result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'Plain text page\nSCANNED OCR'

    async def test_image_without_bytes_excluded_from_ocr(self):
        # An image with empty bytes is not OCR'd; page still emits its blocks.
        page = PyMuPDFPage(
            page_number=1, is_scanned=True,
            blocks=[_blk((0, 0, 10, 10), 'KEPT')],
            images=[PageImage(bbox=(0, 20, 10, 30), image_bytes=b'', ext='png')],
        )
        # No extractable images at all -> falls back to whole-document OCR.
        reader = _make_reader(
            PyMuPDFResponse(pages=[page], page_count=1),
            _ocr_response('WHOLE DOC OCR'),
        )
        result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'WHOLE DOC OCR'


# ---------------------------------------------------------------------------
# Image files — routed straight to OCR
# ---------------------------------------------------------------------------

class TestImageFiles:
    async def test_jpg_routes_to_ocr(self):
        mock_pymupdf = MagicMock()
        mock_ocr = AsyncMock(return_value=_ocr_response('image text'))
        reader = PDFReader(pymupdf_reader=mock_pymupdf, ocr=mock_ocr)
        result = await reader(b'fake', 'scan.jpg')
        assert result.used_ocr is True
        assert result.text == 'image text'

    async def test_jpg_pymupdf_never_called(self):
        mock_pymupdf = MagicMock()
        mock_ocr = AsyncMock(return_value=_ocr_response('image text'))
        reader = PDFReader(pymupdf_reader=mock_pymupdf, ocr=mock_ocr)
        await reader(b'fake', 'scan.jpg')
        mock_pymupdf.assert_not_called()

    async def test_png_routes_to_ocr(self):
        mock_ocr = AsyncMock(return_value=_ocr_response('png text'))
        reader = PDFReader(pymupdf_reader=MagicMock(), ocr=mock_ocr)
        result = await reader(b'fake', 'photo.png')
        assert result.used_ocr is True

    async def test_tiff_routes_to_ocr(self):
        mock_ocr = AsyncMock(return_value=_ocr_response('tiff text'))
        reader = PDFReader(pymupdf_reader=MagicMock(), ocr=mock_ocr)
        result = await reader(b'fake', 'scan.tiff')
        assert result.used_ocr is True


# ---------------------------------------------------------------------------
# Unsupported extension
# ---------------------------------------------------------------------------

class TestUnsupportedExtension:
    async def test_docx_raises(self):
        reader = PDFReader(pymupdf_reader=MagicMock(), ocr=AsyncMock())
        with pytest.raises(UnsupportedAttachmentError):
            await reader(b'fake', 'document.docx')

    async def test_txt_raises(self):
        reader = PDFReader(pymupdf_reader=MagicMock(), ocr=AsyncMock())
        with pytest.raises(UnsupportedAttachmentError):
            await reader(b'fake', 'file.txt')

    async def test_error_message_contains_extension(self):
        reader = PDFReader(pymupdf_reader=MagicMock(), ocr=AsyncMock())
        with pytest.raises(UnsupportedAttachmentError, match=r'\.xlsx'):
            await reader(b'fake', 'data.xlsx')


# ---------------------------------------------------------------------------
# OCRImageTooLargeError — batch fallback and per-image skip
# ---------------------------------------------------------------------------

class TestOCRImageTooLarge:
    async def test_falls_back_to_per_image_on_batch_error(self):
        # Batch call raises; individual calls succeed.
        pages = [
            PyMuPDFPage(page_number=1, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))]),
            PyMuPDFPage(page_number=2, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))]),
        ]
        batch_fail = AsyncMock(side_effect=[
            OCRImageTooLargeError('too large'),  # batch call
            _ocr_response('OCR-A'),              # per-image call for page 1
            _ocr_response('OCR-B'),              # per-image call for page 2
        ])
        reader = PDFReader(
            pymupdf_reader=MagicMock(return_value=PyMuPDFResponse(pages=pages, page_count=2)),
            ocr=batch_fail,
        )
        with patch(
            'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
            return_value=b'%PDF-fake',
        ):
            result = await reader(b'fake', 'doc.pdf')
        assert result.text == 'OCR-A\nOCR-B'

    async def test_skips_and_saves_oversized_individual_image(self, tmp_path):
        page = PyMuPDFPage(
            page_number=1, is_scanned=True,
            blocks=[_blk((0, 0, 10, 5), 'KEPT')],
            images=[_img((0, 10, 10, 20), b'bigbytes')],
        )
        # Batch fails, then per-image also fails.
        always_fail = AsyncMock(side_effect=OCRImageTooLargeError('too large'))
        reader = PDFReader(
            pymupdf_reader=MagicMock(
                return_value=PyMuPDFResponse(pages=[page], page_count=1)
            ),
            ocr=always_fail,
        )
        with (
            patch(
                'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
                return_value=b'%PDF-fake',
            ),
            patch(
                'eml_extract_model.extraction.pdf.reader.settings.OCR_IMAGE_DEBUG_DIR',
                tmp_path,
            ),
        ):
            result = await reader(b'fake', 'doc.pdf')

        # Skipped image produces no OCR text; native text block is still kept.
        assert result.text == 'KEPT'
        saved = list(tmp_path.iterdir())
        assert len(saved) == 1
        assert saved[0].read_bytes() == b'bigbytes'

    async def test_used_ocr_true_even_when_all_images_skipped(self, tmp_path):
        page = PyMuPDFPage(
            page_number=1, is_scanned=True, blocks=[], images=[_img((0, 0, 10, 10))],
        )
        always_fail = AsyncMock(side_effect=OCRImageTooLargeError('too large'))
        reader = PDFReader(
            pymupdf_reader=MagicMock(
                return_value=PyMuPDFResponse(pages=[page], page_count=1)
            ),
            ocr=always_fail,
        )
        with (
            patch(
                'eml_extract_model.extraction.pdf.reader.assemble_image_pdf',
                return_value=b'%PDF-fake',
            ),
            patch(
                'eml_extract_model.extraction.pdf.reader.settings.OCR_IMAGE_DEBUG_DIR',
                tmp_path,
            ),
        ):
            result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is True


# ---------------------------------------------------------------------------
# OCR lazy initialisation
# ---------------------------------------------------------------------------

class TestOCRLazyInit:
    async def test_ocr_not_constructed_for_text_pdf(self):
        # If no OCR is injected and the PDF is all text, OCR client is never built
        reader = PDFReader(
            pymupdf_reader=MagicMock(return_value=_text_response('text')),
            ocr=None,
        )
        # Should not raise even though no OCR client is configured
        result = await reader(b'fake', 'doc.pdf')
        assert result.used_ocr is False
