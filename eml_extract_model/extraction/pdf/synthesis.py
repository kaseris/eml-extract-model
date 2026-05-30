import io
import logging
from dataclasses import dataclass
from typing import List, Optional

import fitz
from PIL import Image

from ...config import settings
from ...schemas.definitions import PageImage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedImage:
    """Image bytes ready for OCR, plus metadata describing any conversion applied."""

    image_bytes: bytes
    ext: str
    action: str
    source_ext: str
    source_bytes: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int

    @property
    def output_bytes(self) -> int:
        return len(self.image_bytes)

    def describe(self, *, index: Optional[int] = None, total: Optional[int] = None) -> str:
        prefix = f'[{index}/{total}] ' if index is not None and total is not None else ''
        src_dims = f'{self.source_width}x{self.source_height}'
        out_dims = f'{self.output_width}x{self.output_height}'
        src_size = _format_bytes(self.source_bytes)
        out_size = _format_bytes(self.output_bytes)

        if self.action == 'unchanged':
            return (
                f'{prefix}unchanged {self.source_ext} {src_dims} {src_size}'
            )
        if self.action == 'jpeg_reencode':
            return (
                f'{prefix}{self.source_ext}→jpeg {src_dims} {src_size}→{out_size}'
            )
        return (
            f'{prefix}{self.source_ext}→jpeg+downscale '
            f'{src_dims}→{out_dims} {src_size}→{out_size}'
        )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f'{num_bytes / (1024 * 1024):.2f} MB'
    if num_bytes >= 1024:
        return f'{num_bytes / 1024:.1f} KB'
    return f'{num_bytes} B'


def _to_jpeg_bytes(img: Image.Image, *, quality: int) -> bytes:
    """Encode a Pillow image as JPEG, flattening alpha onto white if needed."""
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=True)
    return buf.getvalue()


def _resize_scale(img: Image.Image, max_bytes: int, max_dim: int) -> Image.Image:
    """Return a copy of *img* downscaled to fit byte and dimension limits."""
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_dim:
        scale = min(scale, max_dim / max(w, h))

    trial = _to_jpeg_bytes(img, quality=settings.OCR_JPEG_QUALITY)
    if len(trial) > max_bytes and scale >= 1.0:
        scale = min(scale, (max_bytes / len(trial)) ** 0.5 * 0.95)

    if scale >= 1.0:
        return img

    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def prepare_image_for_ocr(image_bytes: bytes, ext: str) -> PreparedImage:
    """Prepare image bytes for OCR: JPEG re-encode, then downscale if still too large.

    JPEG inputs within byte and dimension limits are returned unchanged.  All
    other formats are re-encoded as JPEG first to avoid PNG inflation when
    images are packed into a PDF for batch OCR.
    """
    max_bytes = settings.OCR_MAX_FILE_SIZE_BYTES
    max_dim = settings.OCR_MAX_IMAGE_DIMENSION_PX
    quality = settings.OCR_JPEG_QUALITY

    img = Image.open(io.BytesIO(image_bytes))
    src_w, src_h = img.size
    is_jpeg = ext.lower() in ('jpg', 'jpeg')
    if is_jpeg and len(image_bytes) <= max_bytes and src_w <= max_dim and src_h <= max_dim:
        return PreparedImage(
            image_bytes=image_bytes,
            ext=ext,
            action='unchanged',
            source_ext=ext,
            source_bytes=len(image_bytes),
            source_width=src_w,
            source_height=src_h,
            output_width=src_w,
            output_height=src_h,
        )

    jpeg_bytes = _to_jpeg_bytes(img, quality=quality)
    img = Image.open(io.BytesIO(jpeg_bytes))
    out_w, out_h = img.size
    if len(jpeg_bytes) <= max_bytes and out_w <= max_dim and out_h <= max_dim:
        return PreparedImage(
            image_bytes=jpeg_bytes,
            ext='jpeg',
            action='jpeg_reencode',
            source_ext=ext,
            source_bytes=len(image_bytes),
            source_width=src_w,
            source_height=src_h,
            output_width=out_w,
            output_height=out_h,
        )

    img = _resize_scale(img, max_bytes, max_dim)
    jpeg_bytes = _to_jpeg_bytes(img, quality=quality)
    out_w, out_h = img.size
    return PreparedImage(
        image_bytes=jpeg_bytes,
        ext='jpeg',
        action='jpeg_downscale',
        source_ext=ext,
        source_bytes=len(image_bytes),
        source_width=src_w,
        source_height=src_h,
        output_width=out_w,
        output_height=out_h,
    )


def _log_prepared(prepared: PreparedImage, *, index: int, total: int) -> None:
    logger.info('prepare_image_for_ocr: %s', prepared.describe(index=index, total=total))


def assemble_image_pdf(images: List[PageImage]) -> bytes:
    """Pack extracted page images into a single PDF, one image per page.

    The output preserves input order with a strict 1:1 page-to-image mapping,
    so a single OCR pass over the result yields one OCR page per source image.
    Images are re-encoded to JPEG and/or downscaled when they exceed Azure
    Document Intelligence's file-size or dimension limits.
    """
    total = len(images)
    logger.info('assemble_image_pdf: packing %d images', total)
    if not images:
        return b''

    src_total = 0
    out_total = 0
    out = fitz.open()
    try:
        for index, image in enumerate(images, start=1):
            prepared = prepare_image_for_ocr(image.image_bytes, image.ext)
            _log_prepared(prepared, index=index, total=total)
            src_total += prepared.source_bytes
            out_total += prepared.output_bytes

            src = fitz.open(stream=prepared.image_bytes, filetype=prepared.ext)
            try:
                rect = src[0].rect
                page = out.new_page(width=rect.width, height=rect.height)
                page.insert_image(rect, stream=prepared.image_bytes)
            finally:
                src.close()
        pdf_bytes = out.tobytes()
    finally:
        out.close()

    logger.info(
        'assemble_image_pdf: produced %s PDF from %s raw image bytes (%d→%d bytes)',
        _format_bytes(len(pdf_bytes)),
        _format_bytes(src_total),
        src_total,
        out_total,
    )
    return pdf_bytes
