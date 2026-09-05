# OpenAI Sandbox

A small learning project for experimenting with the OpenAI API.

## Setup

Install the dependencies:

```powershell
python -m pip install openai python-dotenv
```

Add your API key to `.env`:

```text
OPENAI_API_KEY=your-api-key
```

## Run the review workflow

```powershell
python main.py
```

The entry point is intentionally small. Shared OpenAI configuration lives in
`openai_config.py`, review-specific API code lives in `review_summary.py`, and
the sample input data lives in `reviews.py`.
