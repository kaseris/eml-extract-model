from enum import Enum


class EMailCategories(str, Enum):
    CANCELLATION = "cancellation"
    POLICY_ISSUANCE = "policy_issuance"


class AttachmentCategories(str, Enum):
    ID_CARD = "id_card"
    APPLICATION_DOCUMENT = "application_document"
