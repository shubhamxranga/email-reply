"""
Results reporting — per-response and overall system scores.
"""

import json
import os
import numpy as np
from datetime import datetime


class Reporter:
    """Formats and outputs evaluation results as structured reports."""

    def format_results(self, results):
        """
        Transform raw evaluation results into a structured report.

        Args:
            results: List of dicts, each with id, category, emails, and scores.

        Returns:
            A report dict with overall, per-category, and per-response sections.
        """
        # Per-response summary
        per_response = []
        for r in results:
            per_response.append(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "incoming_email_preview": r["incoming_email"][:120] + "..."
                    if len(r["incoming_email"]) > 120
                    else r["incoming_email"],
                    "generated_reply_preview": r["generated_reply"][:120] + "..."
                    if len(r["generated_reply"]) > 120
                    else r["generated_reply"],
                    "scores": r["scores"],
                }
            )

        # Aggregate metrics
        composites = [r["scores"]["composite_score"] for r in results]
        rouges = [r["scores"]["rouge_l"] for r in results]
        semantics = [r["scores"]["semantic_similarity"] for r in results]
        llm_avgs = [r["scores"]["llm_judge"]["average_score"] for r in results]

        overall = {
            "total_evaluated": len(results),
            "composite_score": {
                "mean": round(float(np.mean(composites)), 4),
                "median": round(float(np.median(composites)), 4),
                "std": round(float(np.std(composites)), 4),
                "min": round(float(np.min(composites)), 4),
                "max": round(float(np.max(composites)), 4),
            },
            "rouge_l": {
                "mean": round(float(np.mean(rouges)), 4),
                "median": round(float(np.median(rouges)), 4),
            },
            "semantic_similarity": {
                "mean": round(float(np.mean(semantics)), 4),
                "median": round(float(np.median(semantics)), 4),
            },
            "llm_judge_avg": {
                "mean": round(float(np.mean(llm_avgs)), 4),
                "median": round(float(np.median(llm_avgs)), 4),
            },
        }

        # Per-category breakdown
        categories = {}
        for r in results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["scores"]["composite_score"])

        category_scores = {}
        for cat, scores in sorted(categories.items()):
            category_scores[cat] = {
                "count": len(scores),
                "mean_composite": round(float(np.mean(scores)), 4),
                "median_composite": round(float(np.median(scores)), 4),
            }

        report = {
            "timestamp": datetime.now().isoformat(),
            "scoring_weights": {
                "rouge_l": 0.15,
                "semantic_similarity": 0.20,
                "llm_judge": 0.65,
            },
            "overall": overall,
            "per_category": category_scores,
            "per_response": per_response,
        }

        return report

    def print_summary(self, report):
        """Print a human-readable summary to the console."""
        print("\n" + "=" * 72)
        print("   EVALUATION REPORT — Hiver AI Email Response System")
        print("=" * 72)

        overall = report["overall"]
        print(f"\n   Emails Evaluated: {overall['total_evaluated']}")

        cs = overall["composite_score"]
        print(f"\n   Overall Composite Score: {cs['mean']:.4f}")
        print(f"     median: {cs['median']:.4f} | std: {cs['std']:.4f} | "
              f"range: [{cs['min']:.4f}, {cs['max']:.4f}]")

        print(f"\n   Metric Breakdown (means):")
        print(f"     ROUGE-L:             {overall['rouge_l']['mean']:.4f}")
        print(f"     Semantic Similarity:  {overall['semantic_similarity']['mean']:.4f}")
        print(f"     LLM Judge (norm):     {overall['llm_judge_avg']['mean']:.4f}")

        print(f"\n   Scoring Weights:")
        w = report["scoring_weights"]
        print(f"     ROUGE-L: {w['rouge_l']:.0%} | Semantic: {w['semantic_similarity']:.0%} | "
              f"LLM Judge: {w['llm_judge']:.0%}")

        print(f"\n   Per-Category Breakdown:")
        print(f"   {'Category':<25} {'Mean':>8} {'Median':>8} {'Count':>6}")
        print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*6}")
        for cat, scores in report["per_category"].items():
            print(f"   {cat:<25} {scores['mean_composite']:>8.4f} "
                  f"{scores['median_composite']:>8.4f} {scores['count']:>6}")

        print("\n" + "-" * 72)
        print("   Per-Response Details (all responses):")
        print("-" * 72)

        for r in report["per_response"]:
            s = r["scores"]
            print(f"\n   [{r['id']}] Category: {r['category']}")
            print(f"   Email: {r['incoming_email_preview']}")
            print(f"   Reply: {r['generated_reply_preview']}")
            print(f"   Composite: {s['composite_score']:.4f} | "
                  f"ROUGE: {s['rouge_l']:.4f} | "
                  f"Semantic: {s['semantic_similarity']:.4f} | "
                  f"LLM: {s['llm_judge']['average_score']:.4f}")

            # Print LLM judge dimension breakdown
            judge = s["llm_judge"]
            dims = ["relevance", "completeness", "tone", "accuracy", "actionability"]
            for dim in dims:
                if dim in judge and isinstance(judge[dim], dict):
                    print(f"     {dim.capitalize():15s}: {judge[dim]['score']}/5 — "
                          f"{judge[dim]['justification']}")

        print("\n" + "=" * 72)

    def save_report(self, report, path="results/evaluation_report.json"):
        """Save the full report as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n   Full report saved to: {path}")
