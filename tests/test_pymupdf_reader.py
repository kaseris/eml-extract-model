from unittest.mock import MagicMock, patch

import pytest

from eml_extract_model.errors import PDFParsingError
from eml_extract_model.extraction.pdf.pymupdf.reader import PyMuPDFReader
from eml_extract_model.schemas.definitions import PyMuPDFPage, PyMuPDFResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(
    text: str = '',
    images: list[tuple[float, float, float, float]] | None = None,
    width: float = 100.0,
    height: float = 100.0,
) -> MagicMock:
    """Return a mock fitz.Page with controlled text, images, and dimensions."""
    page = MagicMock()
    page.get_text.return_value = text
    page.get_image_info.return_value = [{'bbox': img} for img in (images or [])]
    page.rect.width = width
    page.rect.height = height
    return page


def _make_doc(*pages: MagicMock) -> MagicMock:
    """Return a mock fitz.Document that iterates over the given pages."""
    doc = MagicMock()
    doc.__iter__ = MagicMock(return_value=iter(pages))
    doc.__enter__ = MagicMock(return_value=doc)
    doc.__exit__ = MagicMock(return_value=False)
    doc.page_count = len(pages)
    return doc


# ---------------------------------------------------------------------------
# Per-page is_scanned flag
# ---------------------------------------------------------------------------

class TestPageScanDetection:
    def test_text_only_page_is_not_scanned(self):
        page = _make_page(text='Hello world', images=[])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is False

    def test_full_page_image_is_scanned(self):
        # Image covers 100% of the 100×100 page
        page = _make_page(text='', images=[(0, 0, 100, 100)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is True

    def test_small_logo_below_threshold_is_not_scanned(self):
        # Logo covers 5% of page; default threshold is 10%
        page = _make_page(text='Some text', images=[(0, 0, 10, 5)])  # 50/10000 = 0.5%... wait let me recalc
        # Page is 100×100 = 10 000 units². Image 10×5 = 50 units² → 0.5%
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is False

    def test_image_at_threshold_is_scanned(self):
        # Image covers exactly 10% of 100×100 page: 100×10 = 1000 units²
        page = _make_page(text='', images=[(0, 0, 100, 10)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is True

    def test_image_just_below_threshold_is_not_scanned(self):
        # Image covers 9.99% of 100×100 page: 99.9 units² → below threshold
        page = _make_page(text='', images=[(0, 0, 100, 9.99)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is False

    def test_multiple_images_summed(self):
        # Two images each covering 6% → 12% total → above threshold
        page = _make_page(text='', images=[(0, 0, 100, 6), (0, 10, 100, 16)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is True

    def test_page_with_zero_area_is_not_scanned(self):
        page = _make_page(text='', images=[(0, 0, 10, 10)], width=0.0, height=0.0)
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is False


# ---------------------------------------------------------------------------
# Configurable threshold
# ---------------------------------------------------------------------------

class TestConfigurableThreshold:
    def test_custom_threshold_50_percent(self):
        # Image covers 30% of page → not scanned with 50% threshold
        page = _make_page(text='', images=[(0, 0, 100, 30)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader(image_area_threshold=0.50)(b'fake')
        assert result.pages[0].is_scanned is False

    def test_custom_threshold_5_percent(self):
        # Image covers 8% of page → scanned with 5% threshold
        page = _make_page(text='', images=[(0, 0, 100, 8)])
        doc = _make_doc(page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader(image_area_threshold=0.05)(b'fake')
        assert result.pages[0].is_scanned is True


# ---------------------------------------------------------------------------
# Multi-page mixed documents
# ---------------------------------------------------------------------------

class TestMultiPageDocuments:
    def test_all_text_pages_none_scanned(self):
        pages = [_make_page(text='Page text', images=[]) for _ in range(3)]
        doc = _make_doc(*pages)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert all(not p.is_scanned for p in result.pages)

    def test_all_image_pages_all_scanned(self):
        pages = [_make_page(text='', images=[(0, 0, 100, 100)]) for _ in range(3)]
        doc = _make_doc(*pages)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert all(p.is_scanned for p in result.pages)

    def test_mixed_pages_flags_correctly(self):
        text_page = _make_page(text='Normal text', images=[])
        scanned_page = _make_page(text='', images=[(0, 0, 100, 100)])
        doc = _make_doc(text_page, scanned_page, text_page)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.pages[0].is_scanned is False
        assert result.pages[1].is_scanned is True
        assert result.pages[2].is_scanned is False

    def test_page_numbers_are_1_indexed(self):
        pages = [_make_page() for _ in range(3)]
        doc = _make_doc(*pages)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert [p.page_number for p in result.pages] == [1, 2, 3]

    def test_page_count_matches(self):
        pages = [_make_page() for _ in range(4)]
        doc = _make_doc(*pages)
        with patch('fitz.open', return_value=doc):
            result = PyMuPDFReader()(b'fake')
        assert result.page_count == 4


# ---------------------------------------------------------------------------
# has_scanned_pages property on PyMuPDFResponse
# ---------------------------------------------------------------------------

class TestHasScannedPages:
    def test_no_scanned_pages(self):
        response = PyMuPDFResponse(
            pages=[
                PyMuPDFPage(page_number=1, text='hello', is_scanned=False),
                PyMuPDFPage(page_number=2, text='world', is_scanned=False),
            ],
            page_count=2,
        )
        assert response.has_scanned_pages is False

    def test_one_scanned_page(self):
        response = PyMuPDFResponse(
            pages=[
                PyMuPDFPage(page_number=1, text='hello', is_scanned=False),
                PyMuPDFPage(page_number=2, text='', is_scanned=True),
            ],
            page_count=2,
        )
        assert response.has_scanned_pages is True

    def test_all_scanned(self):
        response = PyMuPDFResponse(
            pages=[PyMuPDFPage(page_number=i, text='', is_scanned=True) for i in range(1, 4)],
            page_count=3,
        )
        assert response.has_scanned_pages is True

    def test_empty_pages(self):
        response = PyMuPDFResponse(pages=[], page_count=0)
        assert response.has_scanned_pages is False


# ---------------------------------------------------------------------------
# content property with per-page is_scanned
# ---------------------------------------------------------------------------

class TestContentProperty:
    def test_content_joins_all_pages(self):
        response = PyMuPDFResponse(
            pages=[
                PyMuPDFPage(page_number=1, text='Page one', is_scanned=False),
                PyMuPDFPage(page_number=2, text='Page two', is_scanned=False),
            ],
            page_count=2,
        )
        assert response.content == 'Page one\nPage two'

    def test_content_includes_scanned_page_text(self):
        # PyMuPDF may still extract some text even from scanned pages;
        # the content property concatenates whatever text exists.
        response = PyMuPDFResponse(
            pages=[
                PyMuPDFPage(page_number=1, text='Real text', is_scanned=False),
                PyMuPDFPage(page_number=2, text='', is_scanned=True),
            ],
            page_count=2,
        )
        assert response.content == 'Real text\n'


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_pdf_parsing_error_on_bad_bytes(self):
        with patch('fitz.open', side_effect=Exception('corrupted PDF')):
            with pytest.raises(PDFParsingError, match='corrupted PDF'):
                PyMuPDFReader()(b'not a pdf')
