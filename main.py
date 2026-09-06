import argparse

from llm_validation import (
    call_llm_once,
    call_llm_with_retry_and_validation,
)
from review_summary import summarize_reviews
from reviews import all_reviews
from support_agent import run_agent_workflow


def main() -> None:
    """Run one of the LLM learning examples."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario",
        choices=("single", "retry", "agent", "reviews"),
        nargs="?",
        default="None",
        help="Example to run (single, retry, agent, reviews)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum validation retries for the retry scenario",
    )
    args = parser.parse_args()

    if args.scenario == "single":
        print(call_llm_once())
    elif args.scenario == "retry":
        validated_data, error = call_llm_with_retry_and_validation(
            n_retry=args.retries
        )
        if error:
            print(error)
        else:
            print(validated_data.model_dump_json(indent=2))
    elif args.scenario == "agent":
        print(run_agent_workflow().model_dump_json(indent=2))
    elif args.scenario == "reviews":
        summarize_reviews(all_reviews)
    else:
        print(f"Unknown scenario: {args.scenario}, you can choose from: single, retry, agent, reviews")


if __name__ == "__main__":
    main()

