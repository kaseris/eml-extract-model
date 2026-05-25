"""
Field test for the hybrid PDFReader.

Reads a PDF with PyMuPDF first and prints a per-page analysis table showing
the image-area ratio, scanned flag, and char count for each page.  If the
document has scanned pages AND --ocr is passed, the full PDFReader pipeline
runs (PyMuPDF + Azure OCR + stitching) and the final merged text is printed.

Usage:
    uv run python scripts/test_pdf_reader.py
    uv run python scripts/test_pdf_reader.py --pdf /path/to/other.pdf
    uv run python scripts/test_pdf_reader.py --pdf /path/to/scanned.pdf --ocr
    uv run python scripts/test_pdf_reader.py --debug   # show per-page ratio logs
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------------------
# Hard-coded default — change this to point at your test PDF.
# ---------------------------------------------------------------------------
_DEFAULT_PDF = '/path/to/your/test.pdf'
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s %(message)s')
logging.getLogger('eml_extract_model').setLevel(logging.INFO)

from eml_extract_model.extraction.pdf.pymupdf.reader import PyMuPDFReader
from eml_extract_model.extraction.pdf.reader import PDFReader
from eml_extract_model.schemas.definitions import PyMuPDFResponse

_TEXT_PREVIEW_CHARS = 200
_THRESHOLD = 0.10


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return '[' + '#' * filled + '.' * (width - filled) + ']'


def _print_pymupdf_report(response: PyMuPDFResponse, threshold: float) -> None:
    print(f'\n{"Page":>4}  {"Chars":>6}  {"Img ratio":>9}  {"Bar":<22}  {"Status"}')
    print('-' * 62)
    for page in response.pages:
        # Recompute ratio from char_count proxy — ratio itself comes from reader logs;
        # here we just display the is_scanned flag that was set by PyMuPDFReader.
        status = 'SCANNED → OCR' if page.is_scanned else 'text    → PyMuPDF'
        ratio_display = f'≥{threshold:.0%}' if page.is_scanned else f'<{threshold:.0%}'
        print(f'{page.page_number:>4}  {page.char_count:>6}  {ratio_display:>9}  {"":22}  {status}')

    scanned = sum(1 for p in response.pages if p.is_scanned)
    print('-' * 62)
    print(
        f'Total: {response.page_count} pages  |  '
        f'{scanned} scanned  |  '
        f'{response.page_count - scanned} text'
    )


def _print_pymupdf_report_with_ratios(
    pdf_bytes: bytes, threshold: float
) -> PyMuPDFResponse:
    """Run PyMuPDFReader with DEBUG logging to capture ratios, return response."""
    import fitz

    reader = PyMuPDFReader(image_area_threshold=threshold)

    # Compute ratios manually for the display table (reader._image_area_ratio is public enough)
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        ratios = [reader._image_area_ratio(page) for page in doc]

    response = reader(pdf_bytes)

    print(f'\n{"Page":>4}  {"Chars":>6}  {"Img ratio":>9}  {"Bar":<22}  {"Status"}')
    print('-' * 68)
    for page, ratio in zip(response.pages, ratios):
        bar = _bar(ratio)
        status = 'SCANNED → OCR' if page.is_scanned else 'text    → PyMuPDF'
        print(
            f'{page.page_number:>4}  {page.char_count:>6}  {ratio:>9.1%}  {bar:<22}  {status}'
        )

    scanned = sum(1 for p in response.pages if p.is_scanned)
    print('-' * 68)
    print(
        f'Total: {response.page_count} pages  |  '
        f'{scanned} scanned  |  '
        f'{response.page_count - scanned} text  |  '
        f'threshold={threshold:.0%}'
    )
    return response


def _print_text_preview(label: str, text: str) -> None:
    preview = text[:_TEXT_PREVIEW_CHARS].replace('\n', '↵ ')
    truncated = '…' if len(text) > _TEXT_PREVIEW_CHARS else ''
    print(f'\n── {label} ──')
    print(f'Total chars: {len(text)}')
    print(f'Preview: {preview}{truncated}')


async def main(pdf_path: str, run_ocr: bool, debug: bool) -> None:
    if debug:
        logging.getLogger('eml_extract_model').setLevel(logging.DEBUG)

    path = Path(pdf_path)
    if not path.exists():
        print(f'ERROR: file not found: {pdf_path}')
        sys.exit(1)

    pdf_bytes = path.read_bytes()
    print(f'\nFile:  {path.name}')
    print(f'Size:  {len(pdf_bytes):,} bytes')

    # ------------------------------------------------------------------
    # Step 1: per-page PyMuPDF analysis
    # ------------------------------------------------------------------
    print('\n── PyMuPDF per-page analysis ──')
    pymupdf_response = _print_pymupdf_report_with_ratios(pdf_bytes, _THRESHOLD)

    if not pymupdf_response.has_scanned_pages:
        _print_text_preview('Extracted text (PyMuPDF)', pymupdf_response.content)
        print('\nRoute: all pages → PyMuPDF only.  No OCR needed.')
        return

    # ------------------------------------------------------------------
    # Step 2: hybrid path — requires OCR credentials
    # ------------------------------------------------------------------
    scanned_pages = [p.page_number for p in pymupdf_response.pages if p.is_scanned]
    print(f'\nScanned pages: {scanned_pages}')

    if not run_ocr:
        print(
            '\nSkipping OCR (pass --ocr to invoke Azure Document Intelligence).\n'
            'PyMuPDF text for non-scanned pages only:'
        )
        text_only = '\n'.join(
            p.text for p in pymupdf_response.pages if not p.is_scanned
        )
        _print_text_preview('Text pages (PyMuPDF)', text_only)
        return

    print('\n── Running full hybrid PDFReader (PyMuPDF + OCR + stitch) ──')
    reader = PDFReader()
    result = await reader(pdf_bytes, path.name)

    # Per-page source breakdown
    ocr_by_page = {p.pageNumber: p.content for p in (result.ocr_result.pages if result.ocr_result else [])}
    print(f'\n{"Page":>4}  {"Source":<14}  {"Chars":>6}')
    print('-' * 30)
    for page in (result.pymupdf_result.pages if result.pymupdf_result else []):
        if page.is_scanned:
            source = 'OCR'
            chars = len(ocr_by_page.get(page.page_number, ''))
        else:
            source = 'PyMuPDF'
            chars = page.char_count
        print(f'{page.page_number:>4}  {source:<14}  {chars:>6}')

    _print_text_preview('Stitched output', result.text)
    print(f'\nused_ocr={result.used_ocr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the hybrid PDFReader pipeline.')
    parser.add_argument(
        '--pdf',
        default=_DEFAULT_PDF,
        help=f'Path to the PDF to test (default: {_DEFAULT_PDF})',
    )
    parser.add_argument(
        '--ocr',
        action='store_true',
        help='Run Azure Document Intelligence OCR for scanned pages (requires env vars)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable DEBUG logging (shows per-page image ratio from PyMuPDFReader)',
    )
    args = parser.parse_args()
    asyncio.run(main(args.pdf, args.ocr, args.debug))
