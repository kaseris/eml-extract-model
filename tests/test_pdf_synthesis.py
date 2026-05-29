import io

import fitz
from PIL import Image

from eml_extract_model.config import settings
from eml_extract_model.extraction.pdf.synthesis import (
    _to_jpeg_bytes,
    assemble_image_pdf,
    prepare_image_for_ocr,
)
from eml_extract_model.schemas.definitions import PageImage

_MAX_DIMENSION_PX = settings.OCR_MAX_IMAGE_DIMENSION_PX


def _png_bytes(width: int = 20, height: int = 10) -> bytes:
    """Render a tiny solid PNG via PyMuPDF so we have real, decodable image bytes."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, (255, 0, 0))
    return pix.tobytes('png')


def _bloated_png(width: int, height: int) -> bytes:
    """Build a PNG whose uncompressed payload exceeds the OCR file-size limit."""
    img = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    for y in range(height):
        for x in range(width):
            img.set_pixel(x, y, (x % 256, y % 256, (x + y) % 256))
    return img.tobytes('png')


class TestPrepareImageForOcr:
    def test_small_jpeg_returned_unchanged(self):
        png = _png_bytes(20, 10)
        img = Image.open(io.BytesIO(png))
        original = _to_jpeg_bytes(img, quality=settings.OCR_JPEG_QUALITY)
        prepared = prepare_image_for_ocr(original, 'jpeg')
        assert prepared.image_bytes is original
        assert prepared.ext == 'jpeg'
        assert prepared.action == 'unchanged'

    def test_small_png_is_reencoded_as_jpeg(self):
        original = _png_bytes(20, 10)
        prepared = prepare_image_for_ocr(original, 'png')
        assert prepared.ext == 'jpeg'
        assert prepared.action == 'jpeg_reencode'
        assert prepared.image_bytes[:2] == b'\xff\xd8'

    def test_describe_reports_conversion(self):
        prepared = prepare_image_for_ocr(_png_bytes(20, 10), 'png')
        assert 'png→jpeg' in prepared.describe(index=1, total=3)
        assert '[1/3]' in prepared.describe(index=1, total=3)
        original = _png_bytes(20, 10)
        prepared = prepare_image_for_ocr(original, 'png')
        assert prepared.ext == 'jpeg'
        assert prepared.action == 'jpeg_reencode'
        assert prepared.image_bytes[:2] == b'\xff\xd8'

    def test_oversized_file_is_reencoded_as_jpeg(self, monkeypatch):
        limit = 100_000
        monkeypatch.setattr(settings, 'OCR_MAX_FILE_SIZE_BYTES', limit)
        oversized = _bloated_png(800, 600)
        assert len(oversized) > limit
        prepared = prepare_image_for_ocr(oversized, 'png')
        assert prepared.ext == 'jpeg'
        assert prepared.action in ('jpeg_reencode', 'jpeg_downscale')
        assert prepared.image_bytes[:2] == b'\xff\xd8'
        assert prepared.output_bytes <= limit

    def test_oversized_dimensions_are_downscaled_to_jpeg(self):
        oversized = _png_bytes(_MAX_DIMENSION_PX + 500, 100)
        prepared = prepare_image_for_ocr(oversized, 'png')
        assert prepared.ext == 'jpeg'
        assert prepared.action == 'jpeg_downscale'
        assert prepared.output_width <= _MAX_DIMENSION_PX
        assert prepared.output_height <= _MAX_DIMENSION_PX

    def test_aspect_ratio_preserved_when_downscaling(self):
        oversized = _png_bytes(_MAX_DIMENSION_PX + 1000, (_MAX_DIMENSION_PX + 1000) // 2)
        prepared = prepare_image_for_ocr(oversized, 'png')
        assert prepared.ext == 'jpeg'
        ratio = prepared.output_width / prepared.output_height
        assert abs(ratio - 2.0) < 0.1

    def test_assembled_pdf_pages_within_dimension_limit(self):
        big = _png_bytes(_MAX_DIMENSION_PX + 200, _MAX_DIMENSION_PX + 200)
        images = [PageImage(image_bytes=big, ext='png')]
        pdf_bytes = assemble_image_pdf(images)
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            page = doc[0]
            assert page.rect.width <= _MAX_DIMENSION_PX
            assert page.rect.height <= _MAX_DIMENSION_PX


class TestAssembleImagePDF:
    def test_one_page_per_image(self):
        images = [
            PageImage(image_bytes=_png_bytes(), ext='png'),
            PageImage(image_bytes=_png_bytes(), ext='png'),
            PageImage(image_bytes=_png_bytes(), ext='png'),
        ]
        pdf_bytes = assemble_image_pdf(images)
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            assert doc.page_count == 3

    def test_returns_valid_pdf_bytes(self):
        pdf_bytes = assemble_image_pdf([PageImage(image_bytes=_png_bytes(), ext='png')])
        assert pdf_bytes[:5] == b'%PDF-'

    def test_empty_list_returns_empty_bytes(self):
        assert assemble_image_pdf([]) == b''

    def test_page_order_matches_input_order(self):
        images = [PageImage(image_bytes=_png_bytes(), ext='png') for _ in range(4)]
        pdf_bytes = assemble_image_pdf(images)
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            assert doc.page_count == 4

    def test_bloated_png_produces_small_pdf(self, monkeypatch):
        limit = 100_000
        monkeypatch.setattr(settings, 'OCR_MAX_FILE_SIZE_BYTES', limit)
        images = [PageImage(image_bytes=_bloated_png(800, 600), ext='png')]
        pdf_bytes = assemble_image_pdf(images)
        assert pdf_bytes[:5] == b'%PDF-'
        assert len(pdf_bytes) <= limit
