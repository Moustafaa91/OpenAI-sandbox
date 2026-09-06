from validation_models import UserInput


EXAMPLE_RESPONSE_STRUCTURE = """{
    name="Example User",
    email="user@example.com",
    query="I ordered a new computer monitor and it arrived with the screen cracked. I need to exchange it for a new one.",
    order_id="ABC-12345",
    purchase_date="2025-12-31",
    priority="medium",
    category="refund_request",
    is_complaint=True,
    tags=["monitor", "support", "exchange"]
}"""

USER_INPUT_JSON = """
{
    "name": "Joe User",
    "email": "joe@example.com",
    "query": "When can I expect delivery of the headphones I ordered?",
    "order_id": "ABC-12345",
    "purchase_date": "2025-12-01"
}
"""


def build_sample_prompt(user_json: str = USER_INPUT_JSON) -> str:
    user_input = UserInput.model_validate_json(user_json)
    return f"""
Please analyze this user query
{user_input.model_dump_json(indent=2)}:

Return your analysis as a JSON object matching this exact structure
and data types:
{EXAMPLE_RESPONSE_STRUCTURE}

Respond ONLY with valid JSON. Do not include any explanations or
other text or formatting before or after the JSON object.
"""


PROMPT = build_sample_prompt()
