import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

from .errors import ConfigurationError

load_dotenv()

# Azure Document Intelligence max document size per tier (see service limits docs).
_OCR_MAX_FILE_SIZE_F0 = 4 * 1024 * 1024
_OCR_MAX_FILE_SIZE_S0 = 500 * 1024 * 1024


@dataclass
class Settings:
    OCR_MAX_IMAGE_DIMENSION_PX: int = 2048
    # F0 (free) tier limit; set to _OCR_MAX_FILE_SIZE_S0 for Standard (S0).
    OCR_MAX_FILE_SIZE_BYTES: int = _OCR_MAX_FILE_SIZE_F0
    OCR_JPEG_QUALITY: int = 85
    OCR_IMAGE_DEBUG_DIR: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / 'debug'
    )
    MATCH_CONFIDENCE: float = 1.00
    NO_MATCH_CONFIDENCE: float = 0.0
    LOW_CONFIDENCE_THRESHOLD: float = 0.5
    CHEAP_MODEL: str = "gpt-4o-mini"
    CAPABLE_MODEL: str = "gpt-4o"
    SUPPORTED_PDF_EXTENSIONS: frozenset = frozenset({".pdf"})
    SUPPORTED_IMAGE_EXTENSIONS: frozenset = frozenset(
        {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    )
    OPENAI_API_KEY: SecretStr = field(
        default_factory=lambda: SecretStr(os.environ["OPENAI_API_KEY"])
    )
    DOC_INTEL_ENDPOINT: str = field(
        default_factory=lambda: os.environ["DOC_INTEL_ENDPOINT"]
    )
    DOC_INTEL_API_KEY: str = field(
        default_factory=lambda: os.environ["DOC_INTEL_API_KEY"]
    )
    DOC_INTEL_LAYOUT: str = field(
        default_factory=lambda: os.environ["DOC_INTEL_LAYOUT"]
    )


try:
    settings = Settings()
except KeyError as exc:
    raise ConfigurationError(f"Missing required environment variable: {exc}") from exc
