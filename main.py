from reviews import all_reviews
from review_summary import summarize_reviews


def main() -> None:
    """Run the current review-summary workflow."""
    summarize_reviews(all_reviews)


if __name__ == "__main__":
    main()

