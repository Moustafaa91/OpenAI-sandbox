
# OpenAI Sandbox

A small project for learning common OpenAI API patterns with Python.\
**This is not a vibe coding project; it is a demo/sandbox project for learning LLM API integrations.**

## Setup

Run these commands from the project folder:

```powershell
python -m pip install openai llama-index llama-index-llms-openai llama-index-embeddings-huggingface pydantic-ai python-dotenv "pydantic[email]"
```

Create a file named `.env` in the project folder and add your OpenAI API key:

```text
OPENAI_API_KEY=your-api-key
```

## Examples

Choose one of the four examples:

### Single LLM call

```powershell
python main.py single
```

Sends one prompt to the LLM and prints its response.

### LLM call with retries

```powershell
python main.py retry --retries 5
```

Asks the LLM for structured JSON, validates the response with Pydantic, and
tries again when the response is invalid. The `--retries` option controls how
many correction attempts are allowed.

### Customer support agent

```powershell
python main.py agent
```

Analyzes a customer query and decides whether to look up an FAQ, check an
order, escalate the request, or take no action. It then creates a structured
support ticket.

### Review summaries

```powershell
python main.py reviews
```

Summarizes the sample product reviews and gives each one a score from 1 to 10.

## Why Pydantic is used

The LLM is asked to return structured JSON, but its response still needs to be
checked. Pydantic verifies that the response has the expected fields and data
types before the program uses it.

## How retries work

When Pydantic finds an invalid response, the program sends the original prompt,
the response, and the validation error back to the LLM. The LLM is asked to
correct the response, and Pydantic checks it again. This continues until the
response is valid or the retry limit is reached.
