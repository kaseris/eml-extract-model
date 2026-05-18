# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for environment and dependency management.

```bash
# Run the field test for the rule-based classifier
uv run python scripts/test_rbc.py

# Run the field test for the GPT-based email classifier (cheap model by default)
uv run python scripts/test_gbc.py
uv run python scripts/test_gbc.py --capable

# Print compiled regex patterns for a category
uv run python scripts/print_cancellation_patterns.py
uv run python scripts/print_policy_issuance_patterns.py
```

The Python source root is `eml-extract-model/eml-extract-model/`. Scripts in `scripts/` inject this path via `sys.path.insert` to resolve package imports.

## Architecture

The system classifies email bodies and attachments into categories (currently `cancellation`, `policy_issuance`) using two classifier tracks: rule-based and GPT-based. A shared `core/` layer provides the GPT chain machinery used by all GPT-based classifiers.

### Package layout

```
eml-extract-model/
├── classifier/
│   ├── email/
│   │   ├── rule_based/       # Pattern-matching classifier for email bodies
│   │   │   ├── patterns/     # Per-category regex modules
│   │   │   ├── rbc.py        # RuleBasedClassifier entry point
│   │   │   └── registry.py   # CategoryRule registry
│   │   └── gpt_based/
│   │       ├── gbc.py        # GPTBasedClassifier (thin wrapper over GPTChain)
│   │       └── prompts.py    # EMAIL_CLASSIFICATION_PROMPT
│   └── attachment/
│       └── gpt_based/
│           ├── gbc.py        # GPTBasedAttachmentClassifier (thin wrapper over GPTChain)
│           └── prompts.py    # ATTACHMENT_CLASSIFICATION_PROMPT
├── core/
│   └── gpt_chain.py          # Shared GPTChain: LLM construction, structured output, label validation
├── schemas/
│   ├── categories.py         # EMailCategories enum
│   └── definitions.py        # ClassificationResult, GPTClassificationResponse
└── config.py                 # Settings dataclass (model names, confidence thresholds, env vars)
```

### Rule-Based Classifier (`classifier/email/rule_based/`)

The entry point is `RuleBasedClassifier` in `rbc.py`. It iterates a registry of `CategoryRule` instances defined in `registry.py`. Each rule pairs an `EMailCategories` enum value with a tuple of compiled `re.Pattern` objects. The first rule whose any pattern matches the text wins, returning confidence `1.0`; no match returns `""` / `0.0`.

Patterns for each category live in `patterns/`. Each module exposes two compiled patterns:
- A **simple pattern** — alternation of strong keywords (fast, high-precision).
- A **context pattern** — verb within N characters of a subject noun (broader recall).

To add a new category: add a `patterns/<category>.py` module, add the label to `EMailCategories`, and append a `CategoryRule` to `RULES` in `registry.py`.

### GPT-Based Classifiers (`classifier/*/gpt_based/`)

Each GPT-based classifier is a thin wrapper that pairs a prompt with an invoke key and delegates to `GPTChain`. The only differences between classifiers are the prompt import, the invoke key, and the class name.

- `classifier/email/gpt_based/gbc.py` — classifies email bodies, uses `EMAIL_CLASSIFICATION_PROMPT`, invoke key `email_body`
- `classifier/attachment/gpt_based/gbc.py` — classifies attachment content, uses `ATTACHMENT_CLASSIFICATION_PROMPT`, invoke key `attachment_content`

Both classifiers expose an `async def __call__` so they can be awaited directly in FastAPI route handlers without blocking the event loop.

Prompts live in `prompts.py` alongside their classifier and are subject-specific. To add a new GPT-based classifier: add a `prompts.py` with a `ChatPromptTemplate`, create a `gbc.py` that instantiates `GPTChain` with that prompt and an invoke key, and make `__call__` async.

### Core (`core/gpt_chain.py`)

`GPTChain` owns all shared GPT machinery: `ChatOpenAI` construction, `.with_structured_output(GPTClassificationResponse)`, label validation against `EMailCategories`, and confidence passthrough. It takes a `prompt`, `invoke_key`, and `model` at construction time and exposes `async def run(text) -> ClassificationResult`, which uses LangChain's `ainvoke()` to avoid blocking the event loop during the OpenAI API call.

**Sync vs async:** `RuleBasedClassifier` remains synchronous (pure regex, no I/O). All GPT-based classifiers are async. Never call a GPT-based classifier without `await` inside an async context.

### Configuration (`config.py`)

`Settings` is a plain dataclass with defaults for float constants and model names, and `field(default_factory=lambda: os.environ[...])` for required env vars. `load_dotenv()` is called at import time. The singleton `settings` is imported wherever needed.

- `CHEAP_MODEL` / `CAPABLE_MODEL` — model names used by the hybrid strategy
- `MATCH_CONFIDENCE` / `NO_MATCH_CONFIDENCE` — confidence values for rule-based results

Required env vars: `OPENAI_API_KEY`, `DOC_INTEL_ENDPOINT`, `DOC_INTEL_API_KEY`, `DOC_INTEL_LAYOUT`.

### Schemas (`schemas/`)

- `categories.py` — `EMailCategories(str, Enum)` is the source of truth for valid labels.
- `definitions.py` — `ClassificationResult(label, confidence)` is the shared return type for all classifiers; `GPTClassificationResponse(label, confidence)` is the structured output schema used by `GPTChain`.

## Development Workflow

### Adding a new business rule

When the user asks for a new business rule or feature, follow this order strictly:

1. **Understand requirements first** — Ask clarifying questions before writing any code. Do not assume scope.
2. **Write tests second** — Implement the test cases that define the expected behaviour. Tests must fail before implementation exists.
3. **Implement third** — Write the minimum code needed to make the tests pass.
4. **Verify** — Run the full test suite; confirm no regressions.
5. **Propagate to the API** — Update the sibling project at `../eml-extract-api`:
   - `app/pipeline.py` — add/update singleton instances and pipeline logic
   - `app/schemas.py` — add/update API-facing Pydantic response models
   - `tests/test_pipeline.py` and `tests/test_routes.py` — keep tests in sync

Never write implementation code before the tests for that behaviour exist.

## Design Patterns

### Callable classifier interface

All classifiers are callable objects (`__call__`), not functions or methods on a service class. This keeps instantiation (model choice, chain construction) separate from invocation and makes classifiers easy to swap or compose.

```python
# rule-based: sync
result = classifier(text)

# GPT-based: async
result = await classifier(text)
```

### Rule-based: registry + pattern modules

Categories are registered as `CategoryRule(label, patterns)` instances in a central `RULES` tuple (`registry.py`). Each category owns its pattern logic in an isolated `patterns/<category>.py` module that exposes a simple pattern (keyword alternation) and a context pattern (verb–noun proximity). Adding a category never requires touching existing patterns.

### GPT-based: prompt separation

Prompt templates are defined in `prompts.py`, separate from classifier logic in `gbc.py`. Each classifier owns its prompt — email and attachment prompts are never shared. `gbc.py` contains no prompt strings; it only imports and wires the prompt into `GPTChain`.

### GPT-based: thin wrapper over shared chain

All GPT-based classifiers delegate entirely to `GPTChain` in `core/gpt_chain.py`. A classifier's `gbc.py` only declares: which prompt to use, which invoke key to pass, and the default model. All LLM construction, structured output binding, label validation, and confidence handling live exclusively in `GPTChain`.

### Uniform return type

Every classifier — rule-based or GPT-based, email or attachment — returns `ClassificationResult(label, confidence)`. The pipeline layer never needs to know which classifier produced a result.
