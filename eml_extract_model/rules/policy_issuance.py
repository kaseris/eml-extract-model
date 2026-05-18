import logging

from ..errors import (
    ApplicantNameMismatchError,
    MissingApplicationDocumentError,
    MultipleApplicationDocumentsError,
    PolicyIssuanceMissingIDCardError,
    PolicyIssuanceMultipleIDCardsError,
)
from ..schemas.categories import AttachmentCategories
from ..schemas.definitions import (
    ApplicationDocumentExtractionResult,
    ClassificationResult,
    IDCardExtractionResult,
)

logger = logging.getLogger(__name__)

_ID_CARD_LABEL = AttachmentCategories.ID_CARD.value
_APP_DOC_LABEL = AttachmentCategories.APPLICATION_DOCUMENT.value


def validate_policy_issuance_attachments(
    attachment_results: list[ClassificationResult],
) -> None:
    """Assert exactly one id_card and one application_document attachment.

    Raises PolicyIssuanceMissingIDCardError if no id_card is found.
    Raises PolicyIssuanceMultipleIDCardsError if more than one id_card is found.
    Raises MissingApplicationDocumentError if no application_document is found.
    Raises MultipleApplicationDocumentsError if more than one application_document is found.
    """
    id_cards = [r for r in attachment_results if r.label == _ID_CARD_LABEL]
    app_docs = [r for r in attachment_results if r.label == _APP_DOC_LABEL]

    if len(id_cards) == 0:
        raise PolicyIssuanceMissingIDCardError(
            'Policy issuance email must have exactly one ID card attachment; none found.'
        )
    if len(id_cards) > 1:
        raise PolicyIssuanceMultipleIDCardsError(
            f'Policy issuance email must have exactly one ID card attachment; found {len(id_cards)}.'
        )
    if len(app_docs) == 0:
        raise MissingApplicationDocumentError(
            'Policy issuance email must have exactly one application document attachment; none found.'
        )
    if len(app_docs) > 1:
        raise MultipleApplicationDocumentsError(
            f'Policy issuance email must have exactly one application document attachment; found {len(app_docs)}.'
        )

    logger.info(
        'validate_policy_issuance_attachments: id_card confidence=%.2f app_doc confidence=%.2f',
        id_cards[0].confidence,
        app_docs[0].confidence,
    )


def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(token.lower().strip(',') for token in name.split() if token.strip(','))


def validate_applicant_name_match(
    id_card: IDCardExtractionResult,
    app_doc: ApplicationDocumentExtractionResult,
) -> None:
    """Assert that the applicant name on the application document matches the ID card.

    Comparison is case-insensitive and order-independent (handles 'Davis, Mary' == 'Mary Davis').

    Raises ApplicantNameMismatchError if names do not match or either is absent.
    """
    first = id_card.first_name.value
    last = id_card.last_name.value
    applicant = app_doc.applicant_name.value

    if not first or not last or not applicant:
        raise ApplicantNameMismatchError(
            'Applicant name could not be verified: one or more name fields are absent.'
        )

    id_tokens = _name_tokens(f'{first} {last}')
    app_tokens = _name_tokens(applicant)

    if id_tokens != app_tokens:
        raise ApplicantNameMismatchError(
            f'Applicant name mismatch: ID card has "{first} {last}" but application document has "{applicant}".'
        )

    logger.info(
        'validate_applicant_name_match: names match id_card="%s %s" app_doc="%s"',
        first,
        last,
        applicant,
    )
