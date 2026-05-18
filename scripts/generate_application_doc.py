"""Generate a sample insurance application document PDF at the repo root."""
import pathlib
import struct

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = ROOT / 'application_document.pdf'

LINES = [
    'INSURANCE APPLICATION DOCUMENT',
    '',
    'Policy Number:    POL-2025-001234',
    'Application Date: 05/17/2025',
    'Coverage Type:    Auto',
    'Premium Amount:   $1,200.00',
    '',
    'APPLICANT INFORMATION',
    'Applicant Name:   Mary Davis',
    'Date of Birth:    01/15/1985',
    'Address:          123 Maple Street, Springfield, IL 62701',
    'Phone:            (217) 555-0198',
    'Email:            mary.davis@example.com',
    '',
    'AGENT INFORMATION',
    'Agent Name:       Robert Wilson',
    'Agency:           Midwest Insurance Group',
    'License No:       IL-AG-2019-004421',
    '',
    'VEHICLE INFORMATION',
    'Make / Model:     2022 Toyota Camry',
    'VIN:              1HGBH41JXMN109186',
    'Year:             2022',
    '',
    'DECLARATION',
    'I, the undersigned, certify that the information provided in this',
    'application is true and complete to the best of my knowledge.',
    '',
    'Applicant Signature: _______________________  Date: 05/17/2025',
    'Agent Signature:     _______________________  Date: 05/17/2025',
]


def _pdf_string(lines: list[str], x: float, y_start: float, line_height: float) -> bytes:
    ops = []
    y = y_start
    for line in lines:
        safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        ops.append(f'{x} {y} Td ({safe}) Tj 0 0 Td'.encode())
        y -= line_height
    return b'\n'.join(ops)


def build_pdf(lines: list[str]) -> bytes:
    font_obj = (
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier '
        b'/Encoding /WinAnsiEncoding >>'
    )

    stream_content = (
        b'BT\n'
        b'/F1 11 Tf\n'
        b'50 750 Td\n'
        b'11 TL\n'
    )
    for line in lines:
        safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        stream_content += f'({safe}) Tj T*\n'.encode()
    stream_content += b'ET\n'

    stream_len = len(stream_content)

    objects: list[bytes] = []

    catalog = b'<< /Type /Catalog /Pages 2 0 R >>'
    objects.append(catalog)

    pages = b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>'
    objects.append(pages)

    page = (
        b'<< /Type /Page /Parent 2 0 R '
        b'/MediaBox [0 0 612 792] '
        b'/Contents 4 0 R '
        b'/Resources << /Font << /F1 5 0 R >> >> >>'
    )
    objects.append(page)

    content = (
        f'<< /Length {stream_len} >>\nstream\n'.encode()
        + stream_content
        + b'\nendstream'
    )
    objects.append(content)

    objects.append(font_obj)

    body = b'%PDF-1.4\n'
    offsets: list[int] = []

    for i, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f'{i} 0 obj\n'.encode() + obj + b'\nendobj\n'

    xref_offset = len(body)
    xref = f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode()
    for off in offsets:
        xref += f'{off:010d} 00000 n \n'.encode()

    trailer = (
        f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
        f'startxref\n{xref_offset}\n%%EOF\n'
    ).encode()

    return body + xref + trailer


pdf_bytes = build_pdf(LINES)
OUTPUT.write_bytes(pdf_bytes)
print(f'Written: {OUTPUT}')
