"""
Prints the policy issuance regex patterns for external testing (e.g. regexr.com).
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
    "policy_issuance.py",
)

spec = importlib.util.spec_from_file_location("policy_issuance", _patterns_path)
policy_issuance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_issuance)

if __name__ == "__main__":
    simple = policy_issuance.build_simple_pattern()
    context = policy_issuance.build_context_pattern()

    print("=== Simple pattern ===")
    print(simple.pattern)
    print()
    print("=== Context (proximity) pattern ===")
    print(context.pattern)
