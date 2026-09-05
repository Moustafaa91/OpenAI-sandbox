from openai import OpenAI
from pydantic import BaseModel, ValidationError, EmailStr, Field
from typing import List, Literal, Optional
from datetime import date
import json

from openai_config import DEFAULT_MODEL, get_openai_client


class UserInput(BaseModel):
    name: str
    email: EmailStr
    query: str
    order_id: Optional[int] = Field(
        None,
        description="Optional order ID 5-digit number.",
        ge=10000,
        le=99999,
    )
    purchase_date: Optional[date] = None

class CustomerQuery(UserInput):
    priority: str = Field(
        ..., description="Priority level: low, medium, high"
    )
    category: Literal[
        'refund_request', 'information_request', 'other'
    ] = Field(..., description="Query category")
    is_complaint: bool = Field(
        ..., description="Whether this is a complaint"
    )
    tags: List[str] = Field(..., description="Relevant keyword tags")

example_response_structure = f"""{{
    name="Example User",
    email="user@example.com",
    query="I ordered a new computer monitor and it arrived with the screen cracked. I need to exchange it for a new one.",
    order_id=12345,
    purchase_date="2025-12-31",
    priority="medium",
    category="refund_request",
    is_complaint=True,
    tags=["monitor", "support", "exchange"] 
}}"""

user_input_json = '''
{
    "name": "Joe User",
    "email": "joe.user@example.com",
    "query": "I forgot my password.",
    "order_number": null,
    "purchase_date": null
}
'''

user_input = UserInput.model_validate_json(user_input_json)

prompt = f"""
Please analyze this user query\n {user_input.model_dump_json(indent=2)}:

Return your analysis as a JSON object matching this exact structure 
and data types:
{example_response_structure}

Respond ONLY with valid JSON. Do not include any explanations or 
other text or formatting before or after the JSON object.
"""

def call_llm(prompt,client: OpenAI | None = None, model=DEFAULT_MODEL):
     client = client or get_openai_client()
     response = client.chat.completions.create(
         model=model,
         messages=[{"role": "user", "content": prompt}]
    )
     return response.choices[0].message.content


def validate_with_model(data_model, llm_response):
    try:
        validated_data = data_model.model_validate_json(llm_response)
        print("data validation successful!")
        print(validated_data.model_dump_json(indent=2))
        return validated_data, None
    except ValidationError as e:
        print(f"error validating data: {e}")
        error_message = (
            f"This response generated a validation error: {e}."
        )
        return None, error_message

def create_retry_prompt(
    original_prompt, original_response, error_message
):
    retry_prompt = f"""
        This is a request to fix an error in the structure of an llm_response.
        Here is the original request:
        <original_prompt>
        {original_prompt}
        </original_prompt>

        Here is the original llm_response:
        <llm_response>
        {original_response}
        </llm_response>

        This response generated an error: 
        <error_message>
        {error_message}
        </error_message>

        Compare the error message and the llm_response and identify what 
        needs to be fixed or removed
        in the llm_response to resolve this error. 

        Respond ONLY with valid JSON. Do not include any explanations or 
        other text or formatting before or after the JSON string.
        """
    return retry_prompt

def validate_llm_response(
    prompt, data_model, n_retry=5
):
    # Initial LLM call
    response_content = call_llm(prompt)
    current_prompt = prompt

    # Try to validate with the model
    # attempt: 0=initial, 1=first retry, ...
    for attempt in range(n_retry + 1):

        validated_data, validation_error = validate_with_model(
            data_model, response_content
        )

        if validation_error:
            if attempt < n_retry:
                print(f"retry {attempt} of {n_retry} failed, trying again...")
            else:
                print(f"Max retries reached. Last error: {validation_error}")
                return None, (
                    f"Max retries reached. Last error: {validation_error}"
                )

            validation_retry_prompt = create_retry_prompt(
                original_prompt=current_prompt,
                original_response=response_content,
                error_message=validation_error
            )
            response_content = call_llm(validation_retry_prompt)
            current_prompt = validation_retry_prompt
            continue

        # If you get here, both parsing and validation succeeded
        return validated_data, None

def call_llm_once():
    """Call the LLM once and return the response content."""
    response_content = call_llm(prompt)
    print(response_content)
    return response_content

def call_llm_with_retry_and_validation(n_retry=5):
    """Call the LLM with retry mechanism and validate the response."""
    validated_data, error = validate_llm_response(
        prompt, CustomerQuery, n_retry=n_retry
    )
    if error:
        print(f"Error after {n_retry} retries: {error}")
    else:
        print("Validated data:")
        print(validated_data.model_dump_json(indent=2))
    return validated_data, error