import pytest

from eml_extract_model.classifier.email.rule_based.rbc import RuleBasedClassifier
from eml_extract_model.errors import EmptyInputError
from eml_extract_model.schemas.definitions import ClassificationResult


@pytest.fixture
def classifier():
    return RuleBasedClassifier()


class TestEmptyInput:
    def test_empty_string_raises(self, classifier):
        with pytest.raises(EmptyInputError):
            classifier("")

    def test_whitespace_only_raises(self, classifier):
        with pytest.raises(EmptyInputError):
            classifier("   ")

    def test_newline_tab_only_raises(self, classifier):
        with pytest.raises(EmptyInputError):
            classifier("\n\t")


class TestCancellationMatches:
    def test_cancel_keyword(self, classifier):
        result = classifier("I would like to cancel my subscription")
        assert result.label == "cancellation"
        assert result.confidence == 1.0

    def test_cancellation_noun(self, classifier):
        result = classifier("Please process my cancellation request")
        assert result.label == "cancellation"

    def test_unsubscribe(self, classifier):
        result = classifier("Please unsubscribe me from all communications")
        assert result.label == "cancellation"

    def test_terminate(self, classifier):
        result = classifier("We need to terminate the contract immediately")
        assert result.label == "cancellation"

    def test_discontinue(self, classifier):
        result = classifier("Kindly discontinue my plan effective next month")
        assert result.label == "cancellation"

    def test_close_account_context(self, classifier):
        result = classifier("I want to close my account")
        assert result.label == "cancellation"

    def test_opt_out(self, classifier):
        result = classifier("I'd like to opt out of this service")
        assert result.label == "cancellation"


class TestPolicyIssuanceMatches:
    def test_underwriting(self, classifier):
        result = classifier("The underwriting team has approved your application")
        assert result.label == "policy_issuance"
        assert result.confidence == 1.0

    def test_declaration_page(self, classifier):
        result = classifier("Please find your declaration page attached")
        assert result.label == "policy_issuance"

    def test_certificate_of_insurance(self, classifier):
        result = classifier("Enclosed is your certificate of insurance")
        assert result.label == "policy_issuance"

    def test_policy_number(self, classifier):
        result = classifier("Your policy number is POL-789012")
        assert result.label == "policy_issuance"

    def test_issued_policy_context(self, classifier):
        result = classifier("We have issued your policy effective today")
        assert result.label == "policy_issuance"

    def test_new_policy_context(self, classifier):
        result = classifier("Your new policy has been created and sent")
        assert result.label == "policy_issuance"

    def test_request_new_policy_not_cancellation(self, classifier):
        # "request" + "policy" must not fire the cancellation context pattern
        result = classifier(
            "I am writing to request a new auto insurance policy. "
            "I recently purchased a 2024 Honda Civic and need coverage to begin immediately."
        )
        assert result.label == "policy_issuance"


class TestNoMatch:
    def test_unrelated_text(self, classifier):
        result = classifier("Hello, I have a question about my billing statement")
        assert result.label == ""
        assert result.confidence == 0.0

    def test_greeting_only(self, classifier):
        result = classifier("Good morning, I hope this email finds you well")
        assert result.label == ""
        assert result.confidence == 0.0


class TestReturnType:
    def test_match_returns_classification_result(self, classifier):
        result = classifier("cancel my subscription")
        assert isinstance(result, ClassificationResult)

    def test_no_match_returns_classification_result(self, classifier):
        result = classifier("some generic unrelated text here")
        assert isinstance(result, ClassificationResult)
