import logging

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from ....errors import OCRAuthError

logger = logging.getLogger(__name__)


def get_client(endpoint: str, key: str) -> DocumentIntelligenceClient:
    """Construct and return an Azure DocumentIntelligenceClient.

    Raises OCRAuthError if the client cannot be created due to invalid credentials.
    """
    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )
        logger.info('DocumentIntelligenceClient created: endpoint=%s', endpoint)
        return client
    except Exception as exc:
        raise OCRAuthError(
            f'Failed to create DocumentIntelligenceClient: {exc}'
        ) from exc
