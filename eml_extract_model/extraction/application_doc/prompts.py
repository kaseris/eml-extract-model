from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are an expert insurance document analyst. Extract the following fields from \
the provided insurance application document content. For each field, return the \
value exactly as it appears in the document and a confidence score between 0.0 \
and 1.0 reflecting how certain you are about the extracted value.

If a field is not present or cannot be determined from the content, set its \
value to null and its confidence to 0.0.

Fields to extract:
- policy_number: The policy or application reference number
- applicant_name: The full name of the applicant as written in the document
- application_date: The date the application was submitted or signed
- coverage_type: The type of insurance coverage being applied for (e.g. "Auto", "Home", "Life")
- premium_amount: The quoted or agreed premium amount (e.g. "1200.00", "$1,200")
- agent_name: The name of the insurance agent or broker handling the application\
"""

APPLICATION_DOC_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ('human', 'Insurance application document content:\n{application_doc_content}'),
])
