#!/usr/bin/env python3
"""
Hiver AI Email Response System — Main Entry Point

End-to-end pipeline:
1. Generate a synthetic dataset of customer support emails
2. Generate suggested replies using RAG + Gemini
3. Evaluate replies with multi-metric scoring
4. Report per-response and overall system scores

Usage:
    python main.py generate-dataset     # Step 1: Create the dataset
    python main.py generate-responses   # Step 2: Generate replies for test emails
    python main.py evaluate             # Step 3: Score the generated replies
    python main.py run-all              # Steps 2+3 in sequence
"""

import argparse
import json
import os
import sys
import time


def cmd_generate_dataset(args):
    """Generate the synthetic email dataset using Gemini."""
    from data.generate_dataset import generate_dataset

    print("\n=== Generating Synthetic Dataset ===\n")
    generate_dataset(
        output_path="data/dataset.json",
        emails_per_category=args.per_category,
    )


def cmd_generate_responses(args):
    """Generate suggested replies for test emails using RAG."""
    from src.embeddings import EmailRetriever
    from src.responder import EmailResponder

    print("\n=== Generating Suggested Responses ===\n")

    # Load full dataset
    with open("data/dataset.json") as f:
        full_dataset = json.load(f)

    total = len(full_dataset)
    split_idx = int(total * (1 - args.test_split))

    train_data = full_dataset[:split_idx]
    test_data = full_dataset[split_idx:]

    print(f"  Dataset: {total} total emails")
    print(f"  Train (retrieval corpus): {len(train_data)}")
    print(f"  Test (generate + evaluate): {len(test_data)}")

    # Build retriever on training data only (no data leakage)
    retriever = EmailRetriever(dataset=train_data)
    responder = EmailResponder(retriever)

    results = []
    for i, item in enumerate(test_data):
        print(f"\n  [{i + 1}/{len(test_data)}] Generating reply for: {item['id']}")
        print(f"    Email: {item['incoming_email'][:80]}...")

        generated = responder.generate_reply(item["incoming_email"])

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "incoming_email": item["incoming_email"],
                "expected_reply": item["expected_reply"],
                "generated_reply": generated,
            }
        )

        print(f"    Reply: {generated[:80]}...")

        # Rate limiting
        time.sleep(0.5)

    os.makedirs("results", exist_ok=True)
    with open("results/generated_responses.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Generated {len(results)} responses")
    print(f"  Saved to: results/generated_responses.json")

    return results


def cmd_evaluate(args):
    """Evaluate generated responses with multi-metric scoring."""
    from src.embeddings import EmailRetriever
    from src.evaluator import EmailEvaluator
    from src.reporter import Reporter

    print("\n=== Evaluating Generated Responses ===\n")

    # Load generated responses
    responses_path = "results/generated_responses.json"
    if not os.path.exists(responses_path):
        print(f"  ✗ No generated responses found at {responses_path}")
        print(f"    Run 'python main.py generate-responses' first.")
        sys.exit(1)

    with open(responses_path) as f:
        responses = json.load(f)

    # Load training data for the retriever (used for semantic embeddings)
    with open("data/dataset.json") as f:
        full_dataset = json.load(f)

    split_idx = int(len(full_dataset) * (1 - args.test_split))
    train_data = full_dataset[:split_idx]

    retriever = EmailRetriever(dataset=train_data)
    evaluator = EmailEvaluator(retriever)
    reporter = Reporter()

    results = []
    for i, item in enumerate(responses):
        print(f"  Evaluating [{i + 1}/{len(responses)}]: {item['id']}...")

        scores = evaluator.evaluate_response(
            item["incoming_email"],
            item["generated_reply"],
            item["expected_reply"],
        )

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "incoming_email": item["incoming_email"],
                "expected_reply": item["expected_reply"],
                "generated_reply": item["generated_reply"],
                "scores": scores,
            }
        )

        # Rate limiting for LLM judge calls
        time.sleep(0.5)

    # Generate and display report
    report = reporter.format_results(results)
    reporter.print_summary(report)
    reporter.save_report(report)

    return report


def cmd_run_all(args):
    """Run the full pipeline: generate responses → evaluate."""
    print("\n" + "=" * 72)
    print("   HIVER AI EMAIL RESPONSE SYSTEM — Full Pipeline")
    print("=" * 72)

    start = time.time()

    # Check dataset exists
    if not os.path.exists("data/dataset.json"):
        print("\n  ⚠ No dataset found. Generating one first...")
        cmd_generate_dataset(args)

    print("\n--- Step 1/2: Generating Responses ---")
    cmd_generate_responses(args)

    print("\n--- Step 2/2: Evaluating Responses ---")
    cmd_evaluate(args)

    elapsed = time.time() - start
    print(f"\n  ✓ Pipeline complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Hiver AI Email Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate-dataset                  # Create synthetic dataset
  python main.py generate-responses                # Generate AI replies
  python main.py evaluate                          # Score the replies
  python main.py run-all                           # Full pipeline
  python main.py run-all --test-split 0.3          # Use 30% for testing
  python main.py generate-dataset --per-category 5 # Smaller dataset (faster)
        """,
    )

    parser.add_argument(
        "command",
        choices=["generate-dataset", "generate-responses", "evaluate", "run-all"],
        help="Command to run",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of dataset for testing (default: 0.2)",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=10,
        help="Email pairs per category for dataset generation (default: 10)",
    )

    args = parser.parse_args()

    # Validate API key early
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n  ✗ Error: No API key found.")
        print("  Set one of these environment variables:")
        print("    export GEMINI_API_KEY='your-key-here'")
        print("    export GOOGLE_API_KEY='your-key-here'")
        print("  Get a free key at: https://aistudio.google.com/apikey\n")
        sys.exit(1)

    commands = {
        "generate-dataset": cmd_generate_dataset,
        "generate-responses": cmd_generate_responses,
        "evaluate": cmd_evaluate,
        "run-all": cmd_run_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
