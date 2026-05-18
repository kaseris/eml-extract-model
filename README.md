# eml-extract-model

Email and attachment classifier using rule-based and GPT-based models.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in the required environment variables:

```
OPENAI_API_KEY=
DOC_INTEL_ENDPOINT=
DOC_INTEL_API_KEY=
DOC_INTEL_LAYOUT=
```

## Building the wheel

Use `uv build` to produce a `.whl` file in the `dist/` directory:

```bash
uv build --wheel
```

The wheel will be written to `dist/eml_extract_model-<version>-py3-none-any.whl`.

To install the wheel in another environment:

```bash
pip install dist/eml_extract_model-*.whl
```

Or with `uv`:

```bash
uv pip install dist/eml_extract_model-*.whl
```

## Running the classifiers

```bash
# Rule-based classifier field test
uv run python scripts/test_rbc.py

# GPT-based email classifier (cheap model by default)
uv run python scripts/test_gbc.py

# GPT-based email classifier (capable model)
uv run python scripts/test_gbc.py --capable

# Print compiled regex patterns for a category
uv run python scripts/print_cancellation_patterns.py
uv run python scripts/print_policy_issuance_patterns.py
```
