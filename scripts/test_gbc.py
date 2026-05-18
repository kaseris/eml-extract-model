"""
Field tests for GPTBasedClassifier against representative email bodies.
Uses the cheap model by default; pass --capable to use the full model.
"""
import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s %(message)s')
logging.getLogger('eml_extract_model').setLevel(logging.INFO)

from eml_extract_model.classifier.email.gpt_based.gbc import GPTBasedClassifier
from eml_extract_model.config import settings

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
        'cancellation – ambiguous',
        'I am writing to inform you that we will not be renewing the policy upon its expiry next month.',
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
        'policy_issuance – binding confirmation',
        'This email confirms that coverage has been bound effective today at 12:01 AM.',
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


async def main(model: str) -> None:
    print(f'Model: {model}\n')

    classifier = GPTBasedClassifier(model=model)

    header = f"{'Case':<45} {'Expected':<20} {'Got':<20} {'Conf':>6}  {'Result'}"
    print(header)
    print('-' * len(header))

    for description, body in _SAMPLES:
        expected_label = description.split(' – ')[0]
        result = await classifier(body)
        status = _label(expected_label, result.label)
        print(
            f'{description:<45} {expected_label:<20} {result.label or "<none>":<20}'
            f' {result.confidence:>6.2f}  {status}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--capable',
        action='store_true',
        help='Use the capable model instead of the cheap model',
    )
    args = parser.parse_args()

    asyncio.run(main(settings.CAPABLE_MODEL if args.capable else settings.CHEAP_MODEL))
