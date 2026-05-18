"""
Prints the cancellation regex patterns for external testing (e.g. regexr.com).
"""
import importlib.util
import os

_patterns_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "eml_extract_model",
    "classifier",
    "email",
    "rule_based",
    "patterns",
    "cancellation.py",
)

spec = importlib.util.spec_from_file_location("cancellation", _patterns_path)
cancellation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cancellation)

if __name__ == "__main__":
    simple = cancellation.build_simple_pattern()
    context = cancellation.build_context_pattern()

    print("=== Simple pattern ===")
    print(simple.pattern)
    print()
    print("=== Context (proximity) pattern ===")
    print(context.pattern)
