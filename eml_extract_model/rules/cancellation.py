import logging

from ..errors import MissingIDCardAttachmentError, MultipleIDCardAttachmentsError
from ..schemas.categories import AttachmentCategories
from ..schemas.definitions import ClassificationResult

logger = logging.getLogger(__name__)

_ID_CARD_LABEL = AttachmentCategories.ID_CARD.value


def validate_cancellation_attachments(
    attachment_results: list[ClassificationResult],
) -> ClassificationResult:
    """Assert that exactly one attachment is classified as id_card.

    Returns the matching ClassificationResult.
    Raises MissingIDCardAttachmentError if none is id_card.
    Raises MultipleIDCardAttachmentsError if more than one is id_card.
    """
    id_cards = [r for r in attachment_results if r.label == _ID_CARD_LABEL]

    if len(id_cards) == 0:
        raise MissingIDCardAttachmentError(
            'Cancellation email must have exactly one ID card attachment; none found.'
        )
    if len(id_cards) > 1:
        raise MultipleIDCardAttachmentsError(
            f'Cancellation email must have exactly one ID card attachment; found {len(id_cards)}.'
        )

    logger.info('validate_cancellation_attachments: id_card found confidence=%.2f', id_cards[0].confidence)
    return id_cards[0]
