from openai import OpenAI

from openai_config import DEFAULT_MODEL, get_openai_client


def build_prompt(review: str) -> str:
    """Build the prompt used to summarize one product review."""
    return f"""
Your task is to generate a short summary of a product review from an ecommerce site
and give it a score from 1 to 10, where 1 is the worst and 10 is the best.

Summarize the review below, delimited by triple backticks, in at most 20 words.
Return JSON in this format:
{{"summary": "<summary>", "score": <score>}}

Review: ```{review}```
"""


def summarize_review(
    review: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize one review using the OpenAI API."""
    client = client or get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(review)}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def summarize_reviews(reviews: list[str]) -> None:
    """Print a summary for each review, continuing if one request fails."""
    client = get_openai_client()
    for index, review in enumerate(reviews, start=1):
        try:
            print(f"Review {index}: {summarize_review(review, client)}")
        except Exception as error:
            print(f"Review {index} failed: {error}")