from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are an expert identity document analyst. Extract the following fields from \
the provided identity document content. For each field, return the value exactly \
as it appears on the document and a confidence score between 0.0 and 1.0 \
reflecting how certain you are about the extracted value.

If a field is not present or cannot be determined from the content, set its \
value to null and its confidence to 0.0.

Fields to extract:
- first_name: The person's first (given) name
- last_name: The person's last (family) name
- date_of_birth: Date of birth as shown on the document
- expiration_date: Document expiration date as shown on the document
- sex: Sex or gender as shown (e.g. "M", "F", "Male", "Female", "X")
- height: Height as shown (e.g. "180 cm", "5'11\"", "182")\
"""

ID_CARD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ('human', 'Identity document content:\n{id_card_content}'),
])
