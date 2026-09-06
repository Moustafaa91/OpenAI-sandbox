import json
from datetime import datetime

from pydantic_ai import Agent

from mock_data import FAQ_DB, ORDER_DB
from openai_config import DEFAULT_MODEL, get_openai_client
from sample_data import USER_INPUT_JSON
from validation_models import (
    CheckOrderStatusArgs,
    CustomerQuery,
    FAQLookupArgs,
    SupportTicket,
    UserInput,
)


def lookup_faq_answer(args: FAQLookupArgs) -> str:
    query_words = {word.lower() for word in args.query.split()}
    tag_set = {tag.lower() for tag in args.tags}
    best_match = None
    best_score = 0

    for faq in FAQ_DB:
        keywords = {keyword.lower() for keyword in faq["keywords"]}
        score = len(keywords & tag_set) + len(keywords & query_words)
        if score > best_score:
            best_score = score
            best_match = faq

    if best_match and best_score > 0:
        return best_match["answer"]
    return "Sorry, I couldn't find an FAQ answer for your question."


def check_order_status(args: CheckOrderStatusArgs):
    order = ORDER_DB.get(args.order_id)
    if not order:
        return {
            "order_id": args.order_id,
            "status": "not found",
            "estimated_delivery": None,
            "note": "order_id not found",
        }

    if args.email.lower() != order.get("email", "").lower():
        return {
            "order_id": args.order_id,
            "status": order["status"],
            "estimated_delivery": order["estimated_delivery"],
            "note": "order_id found but email mismatch",
        }

    return {
        "order_id": args.order_id,
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"],
        "note": "order_id and email match",
    }


def get_tool_outputs(tool_calls):
    tool_outputs = []
    for tool_call in tool_calls or []:
        if tool_call.function.name == "lookup_faq_answer":
            args = FAQLookupArgs.model_validate_json(
                tool_call.function.arguments
            )
            result = lookup_faq_answer(args)
        elif tool_call.function.name == "check_order_status":
            args = CheckOrderStatusArgs.model_validate_json(
                tool_call.function.arguments
            )
            result = check_order_status(args)
        else:
            continue

        tool_outputs.append({"tool_call_id": tool_call.id, "output": result})
    return tool_outputs


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_faq_answer",
            "description": "Look up an FAQ answer by matching tags to FAQ entry keywords.",
            "parameters": FAQLookupArgs.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the status of a customer's order.",
            "parameters": CheckOrderStatusArgs.model_json_schema(),
        },
    },
]


def validate_user_input(user_json: str):
    try:
        return UserInput.model_validate_json(user_json)
    except Exception as error:
        print(f"Unexpected error: {error}")
        return None


def create_customer_query(valid_user_json: str) -> CustomerQuery:
    customer_query_agent = Agent(
        model=f"openai-chat:{DEFAULT_MODEL}",
        output_type=CustomerQuery,
    )
    return customer_query_agent.run_sync(valid_user_json).output


def decide_next_action_with_tools(customer_query: CustomerQuery):
    client = get_openai_client()
    support_ticket_schema = json.dumps(
        SupportTicket.model_json_schema(), indent=2
    )
    system_prompt = f"""
You are a helpful customer support agent. Your job is to determine what support
action should be taken for the customer, based on the customer query and the
expected fields in the SupportTicket schema below. If more information on an
order_id or FAQ response would be helpful and can be obtained by calling a
tool, call the appropriate tool. If an order_id is present, always look up
its status.

Here is the JSON schema for the SupportTicket model:
{support_ticket_schema}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(customer_query.model_dump())},
    ]
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )
    message = response.choices[0].message
    return message, getattr(message, "tool_calls", None), messages


def generate_structured_support_ticket(
    customer_query: CustomerQuery,
    message,
    tool_outputs: list,
):
    client = get_openai_client()
    tool_results = "\n".join(
        f"Tool: {output['tool_call_id']} Output: {json.dumps(output['output'])}"
        for output in tool_outputs
    ) if tool_outputs else "No tool calls were made."
    prompt = f"""
You are a support agent. Use all information below to generate a support ticket
as a validated Pydantic model.
Customer query: {customer_query.model_dump_json(indent=2)}
LLM message: {message.content}
Tool results: {tool_results}
"""
    response = client.chat.completions.parse(
        model=DEFAULT_MODEL,
        max_completion_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        response_format=SupportTicket,
    )
    response.creation_date = datetime.now()
    return response


def run_agent_workflow(user_json: str = USER_INPUT_JSON) -> SupportTicket:
    """Validate input, run the support agent, and return a support ticket."""
    valid_data = validate_user_input(user_json)
    if valid_data is None:
        raise ValueError("The user input could not be validated.")

    customer_query = create_customer_query(valid_data.model_dump_json())
    message, tool_calls, _ = decide_next_action_with_tools(customer_query)
    return generate_structured_support_ticket(
        customer_query, message, get_tool_outputs(tool_calls)
    )
