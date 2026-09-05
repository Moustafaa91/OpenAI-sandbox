
# OpenAI Sandbox

A small learning project for experimenting with the OpenAI API.\
**This is not a vibe coding project; it is a demo/sandbox project for learning LLM API integrations.**

## Setup

Install the dependencies:

```powershell
python -m pip install openai python-dotenv "pydantic[email]"
```

Add your API key to `.env`:

```text
OPENAI_API_KEY=your-api-key
```

The client uses the `gpt-5.4-mini` model by default. That setting is defined in
`openai_config.py` and can be changed there.

## Run the validation example

From the repository directory, run:

```powershell
python main.py
```

`main.py` imports the sample customer request from `pydantic_validation.py`
and calls `call_llm_with_retry_and_validation()`.

## What is validated

The sample input is first parsed as `UserInput`. The model requires:

- `name` and `query` to be strings.
- `email` to be a valid email address.
- `order_id`, when supplied, to be an integer from `10000` through `99999`.
- `purchase_date`, when supplied, to be an ISO date such as `2025-12-31`.

The LLM response is then validated as `CustomerQuery`, which adds:

- `priority` as a required string. Its field description recommends `low`,
  `medium`, or `high`, but the current model does not enforce those values.
- `category` as `refund_request`, `information_request`, or `other`.
- `is_complaint` as a boolean.
- `tags` as a list of strings.

Validation checks both that the response is valid JSON and that its values match
the Pydantic model. This prevents the application from treating an incomplete,
malformed, or incorrectly typed LLM response as trusted application data.

## Retry behavior

The first request asks the model to return only JSON matching `CustomerQuery`.
If Pydantic raises a `ValidationError`, the code sends another request that
includes the original prompt, the rejected response, and the validation error.
The model is asked to correct only the parts needed to produce valid JSON.

The default `n_retry=5` allows one initial request plus up to five correction
requests. Each failed attempt prints its validation error and a retry message.
If all attempts fail, the function returns `(None, error_message)` and prints
the final error. On success it returns `(validated_data, None)` and prints the
validated JSON.

## Expected output

The exact values depend on the model response, but a successful run has this
shape:

```text
data validation successful!
{
  "name": "Joe User",
  "email": "joe.user@example.com",
  "query": "I forgot my password.",
  "order_id": null,
  "purchase_date": null,
  "priority": "medium",
  "category": "information_request",
  "is_complaint": false,
  "tags": ["password", "account"]
}
Validated data:
{
  "name": "Joe User",
  "email": "joe.user@example.com",
  "query": "I forgot my password.",
  "order_id": null,
  "purchase_date": null,
  "priority": "medium",
  "category": "information_request",
  "is_complaint": false,
  "tags": ["password", "account"]
}
```

When the initial response is invalid, the output also includes messages such as
`error validating data: ...` and `retry 0 of 5 failed, trying again...` before a
successful validation or the final `Max retries reached` error.
