"""Backward-compatible exports for the learning examples.

The implementations live in focused modules. This module keeps the original
import path working for existing examples.
"""

from llm_validation import (
    call_llm,
    call_llm_once,
    call_llm_with_retry_and_validation,
    create_retry_prompt,
    validate_llm_response,
    validate_with_model,
)
from sample_data import (
    EXAMPLE_RESPONSE_STRUCTURE,
    PROMPT,
    USER_INPUT_JSON,
    build_sample_prompt,
)
from mock_data import FAQ_DB, ORDER_DB
from support_agent import (
    TOOL_DEFINITIONS,
    check_order_status,
    create_customer_query,
    decide_next_action_with_tools,
    generate_structured_support_ticket,
    get_tool_outputs,
    lookup_faq_answer,
    run_agent_workflow,
    validate_user_input,
)
from validation_models import (
    CheckOrderStatusArgs,
    CustomerQuery,
    FAQLookupArgs,
    OrderDetails,
    SupportTicket,
    UserInput,
)


# Original lowercase names remain available for existing code.
example_response_structure = EXAMPLE_RESPONSE_STRUCTURE
user_input_json = USER_INPUT_JSON
user_input = UserInput.model_validate_json(USER_INPUT_JSON)
prompt = PROMPT
faq_db = FAQ_DB
order_db = ORDER_DB
tool_definitions = TOOL_DEFINITIONS


if __name__ == "__main__":
    print(run_agent_workflow().model_dump_json(indent=2))
