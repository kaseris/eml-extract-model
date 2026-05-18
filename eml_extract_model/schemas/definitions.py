from typing import List, Optional

from pydantic import BaseModel, Field, computed_field


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


class OCRResponse(BaseModel):
    content: str = ""
    pages: List[OCRPage] = Field(default_factory=list)


class PyMuPDFPage(BaseModel):
    page_number: int
    text: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


class PyMuPDFResponse(BaseModel):
    pages: List[PyMuPDFPage] = Field(default_factory=list)
    page_count: int = 0
    is_scanned: bool = False

    @computed_field
    @property
    def content(self) -> str:
        """Unified text content concatenated from all pages."""
        return "\n".join(p.text for p in self.pages)


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
