import re

import pytest

from eml_extract_model.classifier.email.rule_based.patterns.policy_issuance import (
    ISSUANCE_VERBS,
    POLICY_ISSUANCE_CONTEXT_PATTERN,
    POLICY_ISSUANCE_PATTERN,
    STRONG_KEYWORDS,
    SUBJECT_NOUNS,
    build_context_pattern,
    build_simple_pattern,
)


class TestBuildSimplePattern:
    def test_returns_compiled_pattern(self):
        assert isinstance(build_simple_pattern(), re.Pattern)

    def test_matches_underwriting(self):
        assert build_simple_pattern().search("The underwriting team approved it")

    def test_matches_declaration_page(self):
        assert build_simple_pattern().search("Please find the declaration page attached")

    def test_matches_certificate_of_insurance(self):
        assert build_simple_pattern().search("Enclosed is the certificate of insurance")

    def test_matches_policy_number(self):
        assert build_simple_pattern().search("Your policy number is POL-123456")

    def test_matches_policy_effective(self):
        assert build_simple_pattern().search("Your coverage is policy effective immediately")

    def test_matches_new_policy(self):
        assert build_simple_pattern().search("Your new policy starts today")

    def test_case_insensitive(self):
        p = build_simple_pattern()
        assert p.search("POLICY NUMBER 99")
        assert p.search("UNDERWRITING review")

    def test_no_match_unrelated_text(self):
        p = build_simple_pattern()
        assert not p.search("I would like to cancel my account please")


class TestBuildContextPattern:
    def test_returns_compiled_pattern(self):
        assert isinstance(build_context_pattern(), re.Pattern)

    def test_issued_policy(self):
        assert build_context_pattern().search("We have issued your policy")

    def test_policy_was_created(self):
        assert build_context_pattern().search("Your policy was created today")

    def test_sent_the_policy(self):
        assert build_context_pattern().search("We have sent the policy document")

    def test_noun_before_verb(self):
        assert build_context_pattern().search("The coverage was confirmed")

    def test_case_insensitive(self):
        assert build_context_pattern().search("ISSUED THE POLICY")

    def test_dotall_matches_across_newlines(self):
        assert build_context_pattern().search("issued\nyour\npolicy")

    def test_no_match_when_too_far_apart(self):
        p = build_context_pattern(proximity=5)
        assert not p.search("issued this very long unrelated sentence before policy")


class TestModuleLevelPatterns:
    def test_policy_issuance_pattern_is_compiled(self):
        assert isinstance(POLICY_ISSUANCE_PATTERN, re.Pattern)

    def test_policy_issuance_context_pattern_is_compiled(self):
        assert isinstance(POLICY_ISSUANCE_CONTEXT_PATTERN, re.Pattern)

    def test_module_pattern_matches_underwriting(self):
        assert POLICY_ISSUANCE_PATTERN.search("underwriting")

    def test_module_context_pattern_matches(self):
        assert POLICY_ISSUANCE_CONTEXT_PATTERN.search("issued the policy")

    def test_constants_are_non_empty_lists(self):
        assert isinstance(STRONG_KEYWORDS, list) and len(STRONG_KEYWORDS) > 0
        assert isinstance(ISSUANCE_VERBS, list) and len(ISSUANCE_VERBS) > 0
        assert isinstance(SUBJECT_NOUNS, list) and len(SUBJECT_NOUNS) > 0
