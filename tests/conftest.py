import asyncio
import os
from unittest.mock import AsyncMock

import pytest

# Must be set before any project module is imported so config.py can read them.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("DOC_INTEL_ENDPOINT", "https://test.cognitiveservices.azure.com/")
os.environ.setdefault("DOC_INTEL_API_KEY", "test-key")
os.environ.setdefault("DOC_INTEL_LAYOUT", "prebuilt-layout")


# Prevent tenacity from sleeping during tests. Without this, rate-limit
# retry tests would wait up to 1+2=3 seconds per test case.
@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(asyncio, 'sleep', AsyncMock())
