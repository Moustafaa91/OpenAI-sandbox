from sample_data import PROMPT
from utils import (
    call_llm,
    create_retry_prompt,
    validate_llm_response,
    validate_with_model,
)
from validation_models import CustomerQuery


def call_llm_once(prompt_text: str = PROMPT):
    """Call the LLM once and return the response content."""
    return call_llm(prompt_text)


def call_llm_with_retry_and_validation(
    prompt_text: str = PROMPT,
    n_retry: int = 5,
):
    """Call the LLM with retries and validate the response."""
    return validate_llm_response(
        prompt_text, CustomerQuery, n_retry=n_retry
    )
