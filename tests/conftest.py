import os

# Must be set before any project module is imported so config.py can read them.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("DOC_INTEL_ENDPOINT", "https://test.cognitiveservices.azure.com/")
os.environ.setdefault("DOC_INTEL_API_KEY", "test-key")
os.environ.setdefault("DOC_INTEL_LAYOUT", "prebuilt-layout")
