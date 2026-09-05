from reviews import all_reviews
from review_summary import summarize_reviews
from pydantic_validation import call_llm_once, call_llm_with_retry_and_validation


def main() -> None:
    """Run the current review-summary workflow."""
    #summarize_reviews(all_reviews)

    #call_llm_once()
    call_llm_with_retry_and_validation()


if __name__ == "__main__":
    main()

