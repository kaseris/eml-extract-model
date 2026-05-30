import pytest

from eml_extract_model.schemas.definitions import OCRPage, OCRResponse, OCRSpan


# ---------------------------------------------------------------------------
# OCRSpan
# ---------------------------------------------------------------------------

class TestOCRSpan:
    def test_default_values(self):
        span = OCRSpan()
        assert span.offset == 0
        assert span.length == 0

    def test_custom_values(self):
        span = OCRSpan(offset=5, length=10)
        assert span.offset == 5
        assert span.length == 10


# ---------------------------------------------------------------------------
# OCRPage per-page content
# ---------------------------------------------------------------------------

class TestOCRPage:
    def test_content_defaults_empty(self):
        page = OCRPage(pageNumber=1)
        assert page.content == ''

    def test_page_metadata_fields(self):
        page = OCRPage(pageNumber=2, angle=1.5, width=612.0, height=792.0)
        assert page.pageNumber == 2
        assert page.angle == 1.5
        assert page.width == 612.0
        assert page.height == 792.0

    def test_spans_default_empty(self):
        page = OCRPage(pageNumber=1)
        assert page.spans == []


# ---------------------------------------------------------------------------
# OCRResponse model_validator populates per-page content
# ---------------------------------------------------------------------------

class TestOCRResponsePageContent:
    def test_single_page_single_span(self):
        response = OCRResponse(
            content='Hello world',
            pages=[
                OCRPage(
                    pageNumber=1,
                    spans=[OCRSpan(offset=0, length=11)],
                )
            ],
        )
        assert response.pages[0].content == 'Hello world'

    def test_single_page_partial_span(self):
        response = OCRResponse(
            content='Hello world',
            pages=[
                OCRPage(
                    pageNumber=1,
                    spans=[OCRSpan(offset=6, length=5)],
                )
            ],
        )
        assert response.pages[0].content == 'world'

    def test_single_page_multiple_spans(self):
        response = OCRResponse(
            content='Hello world',
            pages=[
                OCRPage(
                    pageNumber=1,
                    spans=[
                        OCRSpan(offset=0, length=5),
                        OCRSpan(offset=6, length=5),
                    ],
                )
            ],
        )
        assert response.pages[0].content == 'Hello\nworld'

    def test_two_pages_non_overlapping_spans(self):
        content = 'First page text\nSecond page text'
        response = OCRResponse(
            content=content,
            pages=[
                OCRPage(pageNumber=1, spans=[OCRSpan(offset=0, length=15)]),
                OCRPage(pageNumber=2, spans=[OCRSpan(offset=16, length=16)]),
            ],
        )
        assert response.pages[0].content == 'First page text'
        assert response.pages[1].content == 'Second page text'

    def test_page_with_no_spans_has_empty_content(self):
        response = OCRResponse(
            content='Some document text',
            pages=[OCRPage(pageNumber=1, spans=[])],
        )
        assert response.pages[0].content == ''

    def test_span_exceeding_content_length_is_skipped(self):
        # Span starts beyond the content — should not raise, just produce empty
        response = OCRResponse(
            content='Hi',
            pages=[
                OCRPage(pageNumber=1, spans=[OCRSpan(offset=100, length=10)])
            ],
        )
        assert response.pages[0].content == ''

    def test_empty_content_all_pages_empty(self):
        response = OCRResponse(
            content='',
            pages=[
                OCRPage(pageNumber=1, spans=[OCRSpan(offset=0, length=0)]),
            ],
        )
        assert response.pages[0].content == ''

    def test_no_pages(self):
        response = OCRResponse(content='some text', pages=[])
        assert response.pages == []

    def test_total_content_unchanged(self):
        response = OCRResponse(
            content='full document',
            pages=[OCRPage(pageNumber=1, spans=[OCRSpan(offset=0, length=13)])],
        )
        assert response.content == 'full document'


# ---------------------------------------------------------------------------
# model_validate from dict (simulating Azure DI SDK AnalyzeResult.as_dict())
# ---------------------------------------------------------------------------

class TestOCRResponseModelValidate:
    def test_validate_from_dict(self):
        raw = {
            'content': 'Page one text\nPage two text',
            'pages': [
                {
                    'pageNumber': 1,
                    'angle': 0.0,
                    'width': 612.0,
                    'height': 792.0,
                    'spans': [{'offset': 0, 'length': 13}],
                },
                {
                    'pageNumber': 2,
                    'angle': 0.0,
                    'width': 612.0,
                    'height': 792.0,
                    'spans': [{'offset': 14, 'length': 13}],
                },
            ],
        }
        response = OCRResponse.model_validate(raw)
        assert response.pages[0].content == 'Page one text'
        assert response.pages[1].content == 'Page two text'

    def test_validate_from_dict_missing_spans_key(self):
        # Azure may omit 'spans' for some pages — should default to []
        raw = {
            'content': 'text',
            'pages': [{'pageNumber': 1, 'angle': 0.0, 'width': 100, 'height': 100}],
        }
        response = OCRResponse.model_validate(raw)
        assert response.pages[0].content == ''
