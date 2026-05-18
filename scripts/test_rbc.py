"""
Field tests for RuleBasedClassifier against representative email bodies.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eml_extract_model.classifier.email.rule_based.rbc import RuleBasedClassifier

_SAMPLES: list[tuple[str, str]] = [
    (
        'cancellation – strong keyword',
        'Hi, I would like to cancel my subscription effective immediately. Please confirm.',
    ),
    (
        'cancellation – context proximity',
        'Could you please terminate the contract we have in place? I no longer need it.',
    ),
    (
        'cancellation – opt-out',
        'Please opt out my account from all future billing. I want to unsubscribe.',
    ),
    (
        'policy_issuance – strong keyword',
        'Your policy issuance has been completed. Please find your declaration page attached.',
    ),
    (
        'policy_issuance – context proximity',
        'We are pleased to confirm that we have issued the new insurance policy for your vehicle.',
    ),
    (
        'policy_issuance – certificate',
        'Attached is your certificate of insurance along with the policy number for your records.',
    ),
    (
        'no match',
        'Just following up on our meeting from yesterday. Let me know when you are free to chat.',
    ),
]


def _label(expected: str, actual: str) -> str:
    if expected == 'no match':
        return 'PASS' if actual == '' else 'FAIL'
    return 'PASS' if actual == expected else 'FAIL'


if __name__ == '__main__':
    classifier = RuleBasedClassifier()

    header = f"{'Case':<45} {'Expected':<20} {'Got':<20} {'Conf':>6}  {'Result'}"
    print(header)
    print('-' * len(header))

    for description, body in _SAMPLES:
        expected_label = description.split(' – ')[0]
        result = classifier(body)
        status = _label(expected_label, result.label)
        print(
            f'{description:<45} {expected_label:<20} {result.label or "<none>":<20}'
            f' {result.confidence:>6.2f}  {status}'
        )
