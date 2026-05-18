import re

import pytest

from eml_extract_model.classifier.email.rule_based.patterns.cancellation import (
    CANCEL_VERBS,
    CANCELLATION_CONTEXT_PATTERN,
    CANCELLATION_PATTERN,
    STRONG_KEYWORDS,
    SUBJECT_NOUNS,
    build_context_pattern,
    build_simple_pattern,
)


class TestBuildSimplePattern:
    def test_returns_compiled_pattern(self):
        assert isinstance(build_simple_pattern(), re.Pattern)

    def test_matches_cancel(self):
        assert build_simple_pattern().search("I want to cancel my subscription")

    def test_matches_cancellation(self):
        assert build_simple_pattern().search("Please process my cancellation")

    def test_matches_unsubscribe(self):
        assert build_simple_pattern().search("Please unsubscribe me from all emails")

    def test_matches_terminate(self):
        assert build_simple_pattern().search("We need to terminate this agreement")

    def test_matches_discontinue(self):
        assert build_simple_pattern().search("Kindly discontinue my plan")

    def test_matches_opt_out(self):
        assert build_simple_pattern().search("I want to opt out of this service")

    def test_matches_remove_me(self):
        assert build_simple_pattern().search("Please remove me from the list")

    def test_matches_close_account_compound(self):
        # Pattern is `close.?account` — requires ≤1 char between words
        assert build_simple_pattern().search("close account immediately")

    def test_matches_delete_account_compound(self):
        # Pattern is `delete.?account` — requires ≤1 char between words
        assert build_simple_pattern().search("delete account now")

    def test_case_insensitive(self):
        p = build_simple_pattern()
        assert p.search("CANCEL MY ACCOUNT")
        assert p.search("Cancellation Request")

    def test_no_match_unrelated_text(self):
        p = build_simple_pattern()
        assert not p.search("I am very happy with my service and want to renew")

    def test_custom_keywords(self):
        p = build_simple_pattern(["quit", "stop"])
        assert p.search("Please stop the service")
        assert p.search("I want to quit")
        assert not p.search("cancel this")


class TestBuildContextPattern:
    def test_returns_compiled_pattern(self):
        assert isinstance(build_context_pattern(), re.Pattern)

    def test_verb_before_noun(self):
        assert build_context_pattern().search("cancel the subscription")

    def test_noun_before_verb(self):
        assert build_context_pattern().search("subscription cancellation requested")

    def test_close_my_account(self):
        assert build_context_pattern().search("close my account immediately")

    def test_stop_the_service(self):
        assert build_context_pattern().search("stop the service please")

    def test_case_insensitive(self):
        assert build_context_pattern().search("CANCEL THE SUBSCRIPTION")

    def test_dotall_matches_across_newlines(self):
        assert build_context_pattern().search("cancel\nthe\nsubscription")

    def test_no_match_when_too_far_apart(self):
        p = build_context_pattern(proximity=5)
        assert not p.search("cancel this very long unrelated text before subscription")

    def test_custom_verb_noun_lists(self):
        p = build_context_pattern(verbs=[r"end"], nouns=["plan"])
        assert p.search("end the plan")
        assert not p.search("cancel the subscription")


class TestModuleLevelPatterns:
    def test_cancellation_pattern_is_compiled(self):
        assert isinstance(CANCELLATION_PATTERN, re.Pattern)

    def test_cancellation_context_pattern_is_compiled(self):
        assert isinstance(CANCELLATION_CONTEXT_PATTERN, re.Pattern)

    def test_module_pattern_matches_cancel(self):
        assert CANCELLATION_PATTERN.search("cancel")

    def test_module_context_pattern_matches(self):
        assert CANCELLATION_CONTEXT_PATTERN.search("cancel the subscription")

    def test_constants_are_non_empty_lists(self):
        assert isinstance(STRONG_KEYWORDS, list) and len(STRONG_KEYWORDS) > 0
        assert isinstance(CANCEL_VERBS, list) and len(CANCEL_VERBS) > 0
        assert isinstance(SUBJECT_NOUNS, list) and len(SUBJECT_NOUNS) > 0
