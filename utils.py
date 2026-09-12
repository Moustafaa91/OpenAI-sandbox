from typing import Type

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from openai_config import DEFAULT_MODEL, get_openai_client


def call_llm(
    prompt: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    client = client or get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def validate_with_model(
    data_model: Type[BaseModel], llm_response: str
):
    try:
        validated_data = data_model.model_validate_json(llm_response)
        print("data validation successful!")
        print(validated_data.model_dump_json(indent=2))
        return validated_data, None
    except ValidationError as error:
        print(f"error validating data: {error}")
        return None, f"This response generated a validation error: {error}."


def create_retry_prompt(
    original_prompt: str,
    original_response: str,
    error_message: str,
) -> str:
    return f"""
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
needs to be fixed or removed in the llm_response to resolve this error.

Respond ONLY with valid JSON. Do not include any explanations or
other text or formatting before or after the JSON string.
"""


def validate_llm_response(
    prompt: str,
    data_model: Type[BaseModel],
    n_retry: int = 5,
):
    response_content = call_llm(prompt)
    current_prompt = prompt

    for attempt in range(n_retry + 1):
        validated_data, validation_error = validate_with_model(
            data_model, response_content
        )
        if validation_error is None:
            return validated_data, None

        if attempt == n_retry:
            print(f"Max retries reached. Last error: {validation_error}")
            return None, f"Max retries reached. Last error: {validation_error}"

        print(f"retry {attempt} of {n_retry} failed, trying again...")
        current_prompt = create_retry_prompt(
            original_prompt=current_prompt,
            original_response=response_content,
            error_message=validation_error,
        )
        response_content = call_llm(current_prompt)
