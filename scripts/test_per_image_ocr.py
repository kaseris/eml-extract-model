"""
Field test for the per-image positional OCR path in PDFReader.

Reads the PDF at _DEFAULT_PDF and shows:
  - PyMuPDF per-page analysis (image ratio, blocks, embedded images)
  - Per-image extraction summary
  - Assembled OCR input PDF stats
  - Positional merge result (with mock OCR or live Azure)

Usage:
    uv run python scripts/test_per_image_ocr.py
    uv run python scripts/test_per_image_ocr.py --ocr     # live Azure OCR
    uv run python scripts/test_per_image_ocr.py --debug
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Hard-coded default — change this to point at your test PDF.
# ---------------------------------------------------------------------------
_DEFAULT_PDF = "/Users/michaliskaseris/Downloads/ToF Classification-2.pdf"
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
logging.getLogger("eml_extract_model").setLevel(logging.INFO)

import fitz

from eml_extract_model.extraction.pdf.pymupdf.reader import PyMuPDFReader
from eml_extract_model.extraction.pdf.reader import PDFReader
from eml_extract_model.extraction.pdf.synthesis import (
    _format_bytes,
    assemble_image_pdf,
    prepare_image_for_ocr,
)
from eml_extract_model.schemas.definitions import OCRPage, OCRResponse, PyMuPDFResponse

_THRESHOLD = 0.10
_TEXT_PREVIEW_CHARS = 300


# ---------------------------------------------------------------------------
# Mock OCR — returns plausible per-image content without Azure credentials
# ---------------------------------------------------------------------------


def _build_mock_ocr(images) -> AsyncMock:
    pages = []
    combined = []
    for i, _ in enumerate(images):
        text = f"[OCR text extracted from embedded image {i + 1}]"
        pages.append(OCRPage(pageNumber=i + 1, content=text))
        combined.append(text)
    ocr_response = OCRResponse(content="\n".join(combined), pages=pages)
    return AsyncMock(return_value=ocr_response)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_pymupdf_report(pymupdf_response: PyMuPDFResponse, ratios: list) -> None:
    print(
        f'\n{"Page":>4}  {"Chars":>6}  {"Blocks":>6}  {"Images":>6}  {"Img ratio":>9}  {"Bar":<22}  {"Route"}'
    )
    print("-" * 80)
    for page, ratio in zip(pymupdf_response.pages, ratios):
        route = "SCANNED → per-image OCR" if page.is_scanned else "text    → PyMuPDF"
        print(
            f"{page.page_number:>4}  {page.char_count:>6}  "
            f"{len(page.blocks):>6}  {len(page.images):>6}  "
            f"{ratio:>9.1%}  {_bar(ratio):<22}  {route}"
        )
    scanned = sum(1 for p in pymupdf_response.pages if p.is_scanned)
    print("-" * 80)
    print(
        f"Total: {pymupdf_response.page_count} pages  |  "
        f"{scanned} scanned  |  "
        f"{pymupdf_response.page_count - scanned} text  |  "
        f"threshold={_THRESHOLD:.0%}"
    )


def _print_image_extraction_report(pymupdf_response: PyMuPDFResponse) -> list:
    print("\n── Embedded images extracted for OCR ──")
    all_images = [
        (page.page_number, idx, img)
        for page in pymupdf_response.pages
        if page.is_scanned
        for idx, img in enumerate(page.images)
        if img.image_bytes
    ]
    if not all_images:
        print("  (none — will fall back to whole-document OCR)")
        return []
    print(f'  {"Src page":>8}  {"Img #":>5}  {"BBox":>30}  {"Bytes":>8}  {"Ext"}')
    print("  " + "-" * 60)
    for page_no, idx, img in all_images:
        bbox = (
            f"({img.bbox[0]:.0f},{img.bbox[1]:.0f},{img.bbox[2]:.0f},{img.bbox[3]:.0f})"
        )
        print(
            f"  {page_no:>8}  {idx:>5}  {bbox:>30}  {len(img.image_bytes):>8}  {img.ext}"
        )
    print(f"\n  {len(all_images)} image(s) → will be assembled into one OCR-input PDF")
    return [img for _, _, img in all_images]


def _print_image_prep_report(all_images: list) -> None:
    print("\n── OCR image preparation ──")
    if not all_images:
        return
    print(
        f'  {"#":>3}  {"Action":<18}  {"Src ext":<7}  {"Out ext":<7}  '
        f'{"Dimensions":>15}  {"Src size":>10}  {"Out size":>10}'
    )
    print("  " + "-" * 78)
    src_total = 0
    out_total = 0
    for index, img in enumerate(all_images, start=1):
        prepared = prepare_image_for_ocr(img.image_bytes, img.ext)
        src_total += prepared.source_bytes
        out_total += prepared.output_bytes
        dims = f'{prepared.source_width}x{prepared.source_height}'
        if (prepared.output_width, prepared.output_height) != (
            prepared.source_width,
            prepared.source_height,
        ):
            dims = f'{dims}→{prepared.output_width}x{prepared.output_height}'
        print(
            f'  {index:>3}  {prepared.action:<18}  {prepared.source_ext:<7}  '
            f'{prepared.ext:<7}  {dims:>15}  '
            f'{_format_bytes(prepared.source_bytes):>10}  '
            f'{_format_bytes(prepared.output_bytes):>10}'
        )
    print("  " + "-" * 78)
    print(
        f'  {"":>3}  {"TOTAL":<18}  {"":<7}  {"":<7}  {"":>15}  '
        f'{_format_bytes(src_total):>10}  {_format_bytes(out_total):>10}'
    )


def _print_merge_report(pymupdf_response: PyMuPDFResponse) -> None:
    print("\n── Per-page merge (reading order) ──")
    for page in pymupdf_response.pages:
        source = "per-image OCR + blocks" if page.is_scanned else "PyMuPDF verbatim"
        print(f"\n  Page {page.page_number}  [{source}]")
        if page.is_scanned:
            block_items = [(round(b.bbox[1], 1), b.text[:60]) for b in page.blocks]
            img_items = [
                (round(i.bbox[1], 1), f"[image @ y={i.bbox[1]:.0f}]")
                for i in page.images
            ]
            for y, label in sorted(block_items + img_items, key=lambda x: x[0]):
                print(f"    y={y:>6}  {label}")


def _print_text_preview(label: str, text: str) -> None:
    preview = text[:_TEXT_PREVIEW_CHARS].replace("\n", " ↵ ")
    truncated = "…" if len(text) > _TEXT_PREVIEW_CHARS else ""
    print(f"\n── {label} ──")
    print(f"Total chars: {len(text)}")
    print(f"Preview: {preview}{truncated}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(run_live_ocr: bool, debug: bool) -> None:
    if debug:
        logging.getLogger("eml_extract_model").setLevel(logging.DEBUG)

    path = Path(_DEFAULT_PDF)
    if not path.exists():
        print(f"ERROR: file not found: {_DEFAULT_PDF}")
        print("Update _DEFAULT_PDF at the top of this script.")
        sys.exit(1)

    pdf_bytes = path.read_bytes()
    print(f"\nFile:  {path.name}")
    print(f"Size:  {len(pdf_bytes):,} bytes")

    # ------------------------------------------------------------------
    # Step 1: PyMuPDF per-page analysis
    # ------------------------------------------------------------------
    print("\n── PyMuPDF per-page analysis ──")
    pymupdf_reader = PyMuPDFReader(image_area_threshold=_THRESHOLD)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        ratios = [
            pymupdf_reader._image_area_ratio(page, page.get_image_info(xrefs=True))
            for page in doc
        ]
    pymupdf_response = pymupdf_reader(pdf_bytes)
    _print_pymupdf_report(pymupdf_response, ratios)

    if not pymupdf_response.has_scanned_pages:
        _print_text_preview("Extracted text (PyMuPDF only)", pymupdf_response.content)
        print("\nRoute: all pages → PyMuPDF.  No OCR needed.")
        return

    # ------------------------------------------------------------------
    # Step 2: show what will be sent to OCR
    # ------------------------------------------------------------------
    all_images = _print_image_extraction_report(pymupdf_response)

    _print_image_prep_report(all_images)

    if all_images:
        ocr_pdf = assemble_image_pdf(all_images)
        print(f"\n── Assembled OCR-input PDF ──")
        with fitz.open(stream=ocr_pdf, filetype="pdf") as d:
            print(f"  Pages: {d.page_count}  |  Size: {len(ocr_pdf):,} bytes")

    # ------------------------------------------------------------------
    # Step 3: run PDFReader with mock or live OCR
    # ------------------------------------------------------------------
    if run_live_ocr:
        print("\n── Running PDFReader with live Azure OCR ──")
        reader = PDFReader()
    else:
        print("\n── Running PDFReader with mock OCR (pass --ocr for live Azure) ──")
        mock_ocr = _build_mock_ocr(all_images)
        reader = PDFReader(pymupdf_reader=pymupdf_reader, ocr=mock_ocr)

    result = await reader(pdf_bytes, path.name)

    # ------------------------------------------------------------------
    # Step 4: display results
    # ------------------------------------------------------------------
    if result.pymupdf_result:
        _print_merge_report(result.pymupdf_result)
    _print_text_preview("Final stitched output", result.text)

    print(f"\nused_ocr={result.used_ocr}")
    ocr_page_count = len(result.ocr_result.pages) if result.ocr_result else 0
    print(f"OCR pages: {ocr_page_count}  (one per extracted image)")

    # ------------------------------------------------------------------
    # Step 5: write JSON output
    # ------------------------------------------------------------------
    output = {
        "file": path.name,
        "size_bytes": len(pdf_bytes),
        "used_ocr": result.used_ocr,
        "total_chars": len(result.text),
        "ocr_pages": ocr_page_count,
        "pages": [
            {
                "page_number": page.page_number,
                "is_scanned": page.is_scanned,
                "char_count": page.char_count,
                "image_ratio": ratios[page.page_number - 1],
                "blocks": [
                    {"bbox": list(b.bbox), "text": b.text} for b in page.blocks
                ],
                "images": [
                    {"bbox": list(i.bbox), "ext": i.ext, "size_bytes": len(i.image_bytes)}
                    for i in page.images
                ],
            }
            for page in (result.pymupdf_result.pages if result.pymupdf_result else [])
        ],
        "stitched_text": result.text,
    }

    out_path = path.with_suffix(".ocr_result.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nJSON written → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the per-image positional OCR path in PDFReader."
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use live Azure Document Intelligence instead of mock OCR",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()
    asyncio.run(main(args.ocr, args.debug))
