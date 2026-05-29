from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field, model_validator

# Axis-aligned bounding box on a PDF page: (x0, y0, x1, y1) in page units.
BBox = Tuple[float, float, float, float]


class ClassificationResult(BaseModel):
    label: str
    confidence: float


class GPTClassificationResponse(BaseModel):
    label: str
    confidence: float


class OCRSpan(BaseModel):
    offset: int = 0
    length: int = 0


class OCRPage(BaseModel):
    pageNumber: int
    angle: float = 0.0
    width: float = 0.0
    height: float = 0.0
    spans: List[OCRSpan] = Field(default_factory=list)
    content: str = ""


class OCRResponse(BaseModel):
    content: str = ""
    pages: List[OCRPage] = Field(default_factory=list)

    @model_validator(mode='after')
    def _populate_page_content(self) -> 'OCRResponse':
        """Derive per-page text from span offsets into the top-level content string.

        Only runs when spans are present; pages with no spans keep whatever
        content value was set at construction time (typically the default "").
        """
        for page in self.pages:
            if not page.spans:
                continue
            parts = [
                self.content[s.offset : s.offset + s.length]
                for s in page.spans
                if s.offset + s.length <= len(self.content)
            ]
            page.content = "\n".join(parts)
        return self


class TextBlock(BaseModel):
    """A positioned run of text extracted from a page, used for reading-order merges."""
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    text: str = ""


class PageImage(BaseModel):
    """An embedded raster image on a page: its position plus the raw image bytes."""
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    image_bytes: bytes = b""
    ext: str = "png"


class PyMuPDFPage(BaseModel):
    page_number: int
    text: str = ""
    is_scanned: bool = False
    blocks: List[TextBlock] = Field(default_factory=list)
    images: List[PageImage] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


class PyMuPDFResponse(BaseModel):
    pages: List[PyMuPDFPage] = Field(default_factory=list)
    page_count: int = 0

    @computed_field
    @property
    def content(self) -> str:
        """Unified text content concatenated from all pages."""
        return "\n".join(p.text for p in self.pages)

    @computed_field
    @property
    def has_scanned_pages(self) -> bool:
        """True if at least one page was flagged as image-dominant by PyMuPDF."""
        return any(p.is_scanned for p in self.pages)


class PDFReadResult(BaseModel):
    text: str
    used_ocr: bool
    pymupdf_result: Optional[PyMuPDFResponse] = None
    ocr_result: Optional[OCRResponse] = None


class ExtractedField(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0


class IDCardExtractionResult(BaseModel):
    first_name: ExtractedField = Field(default_factory=ExtractedField)
    last_name: ExtractedField = Field(default_factory=ExtractedField)
    date_of_birth: ExtractedField = Field(default_factory=ExtractedField)
    expiration_date: ExtractedField = Field(default_factory=ExtractedField)
    sex: ExtractedField = Field(default_factory=ExtractedField)
    height: ExtractedField = Field(default_factory=ExtractedField)


class ApplicationDocumentExtractionResult(BaseModel):
    policy_number: ExtractedField = Field(default_factory=ExtractedField)
    applicant_name: ExtractedField = Field(default_factory=ExtractedField)
    application_date: ExtractedField = Field(default_factory=ExtractedField)
    coverage_type: ExtractedField = Field(default_factory=ExtractedField)
    premium_amount: ExtractedField = Field(default_factory=ExtractedField)
    agent_name: ExtractedField = Field(default_factory=ExtractedField)
