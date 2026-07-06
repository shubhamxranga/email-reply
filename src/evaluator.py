"""
Multi-metric evaluation system for generated email replies.

Metrics:
1. ROUGE-L    — Lexical overlap (longest common subsequence)
2. Semantic   — Cosine similarity between sentence embeddings
3. LLM Judge  — Multi-dimensional quality assessment via Gemini

Composite = 0.15 * ROUGE-L + 0.20 * Semantic + 0.65 * LLM-Judge
"""

import google.generativeai as genai
from rouge_score import rouge_scorer
import numpy as np
import os
import json
import time


class EmailEvaluator:
    """Evaluates generated email replies using multiple complementary metrics."""

    # Composite score weights — LLM Judge weighted highest because
    # it's the only metric that evaluates tone, completeness, and actionability,
    # which automated metrics fundamentally cannot capture.
    WEIGHT_ROUGE = 0.15
    WEIGHT_SEMANTIC = 0.20
    WEIGHT_LLM_JUDGE = 0.65

    def __init__(self, retriever):
        """
        Initialize the evaluator.

        Args:
            retriever: An EmailRetriever instance (used for encoding text to embeddings).
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        genai.configure(api_key=api_key)
        self.judge_model = genai.GenerativeModel("gemini-2.0-flash")
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.retriever = retriever

    def evaluate_response(self, incoming_email, generated_reply, reference_reply):
        """
        Evaluate a generated reply against a reference reply.

        Returns a dict with per-metric scores and a composite score.
        """
        # Metric 1: ROUGE-L (lexical overlap)
        rouge_score = self._compute_rouge(generated_reply, reference_reply)

        # Metric 2: Semantic Similarity (embedding cosine)
        semantic_score = self._compute_semantic_similarity(
            generated_reply, reference_reply
        )

        # Metric 3: LLM-as-Judge (multi-dimensional quality)
        judge_result = self._llm_judge(incoming_email, generated_reply, reference_reply)

        # Composite Score
        composite = (
            self.WEIGHT_ROUGE * rouge_score
            + self.WEIGHT_SEMANTIC * semantic_score
            + self.WEIGHT_LLM_JUDGE * judge_result["average_score"]
        )

        return {
            "rouge_l": round(rouge_score, 4),
            "semantic_similarity": round(semantic_score, 4),
            "llm_judge": judge_result,
            "composite_score": round(composite, 4),
        }

    def _compute_rouge(self, generated, reference):
        """
        Compute ROUGE-L F-measure between generated and reference text.

        ROUGE-L measures the longest common subsequence, capturing structural
        similarity. It's useful for detecting when generated replies follow
        the same general template as the reference.

        Limitation: Misses semantic paraphrases
        ("I'll help you" vs "Let me assist you" scores low).
        """
        scores = self.rouge.score(reference, generated)
        return scores["rougeL"].fmeasure

    def _compute_semantic_similarity(self, generated, reference):
        """
        Compute cosine similarity between sentence embeddings.

        Uses the same embedding model as the retriever to measure semantic
        alignment between generated and reference replies.

        Strength: Captures meaning beyond exact word overlap.
        Limitation: Can score high for topically similar but functionally
        different replies (e.g., both mention "billing" but one is a refund
        and the other is an invoice).
        """
        gen_emb = self.retriever.encode_text(generated)
        ref_emb = self.retriever.encode_text(reference)

        # Cosine similarity
        dot = np.dot(gen_emb, ref_emb)
        norm = np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb)
        similarity = dot / norm if norm > 0 else 0.0

        return float(max(0.0, min(1.0, similarity)))  # Clamp to [0, 1]

    def _llm_judge(self, incoming_email, generated_reply, reference_reply):
        """
        Use Gemini as a judge to evaluate the generated reply on 5 dimensions.

        This is the most important metric because it evaluates qualities that
        automated metrics fundamentally cannot capture:
        - Tone and empathy (is the reply appropriate for the situation?)
        - Completeness (are all questions answered?)
        - Actionability (does it give clear next steps?)

        Each dimension is scored 1-5 with a justification.
        """
        prompt = f"""You are an expert customer support quality evaluator. Your job is to rate a generated email reply on 5 quality dimensions.

ORIGINAL CUSTOMER EMAIL:
{incoming_email}

REFERENCE REPLY (what a skilled human agent sent):
{reference_reply}

GENERATED REPLY (the one you are evaluating):
{generated_reply}

Rate the GENERATED REPLY on each dimension using a 1-5 scale:
- 1 = Very poor
- 2 = Below average
- 3 = Acceptable
- 4 = Good
- 5 = Excellent

Dimensions to evaluate:

1. RELEVANCE: Does the reply directly address the specific concerns, questions, or issues raised in the customer's email? (Not just topically related, but specifically responsive)

2. COMPLETENESS: Are ALL questions and issues in the customer's email addressed? Does the reply leave anything unanswered?

3. TONE: Is the tone professional, empathetic, and appropriate for the customer's emotional state? (A frustrated customer needs acknowledgment; a simple question needs a clear answer)

4. ACCURACY: Is the information in the reply factually consistent with the reference reply? Does it avoid making up features, policies, or procedures?

5. ACTIONABILITY: Does the reply provide clear, concrete next steps? Can the customer act on this reply without needing to write back for clarification?

Respond in EXACTLY this JSON format with no markdown formatting, no code blocks, no extra text:
{{"relevance": {{"score": 4, "justification": "brief reason"}}, "completeness": {{"score": 3, "justification": "brief reason"}}, "tone": {{"score": 5, "justification": "brief reason"}}, "accuracy": {{"score": 4, "justification": "brief reason"}}, "actionability": {{"score": 4, "justification": "brief reason"}}}}"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.judge_model.generate_content(prompt)
                text = response.text.strip()

                # Clean up markdown code blocks if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if "```" in text:
                        text = text.rsplit("```", 1)[0]
                    text = text.strip()

                result = json.loads(text)

                # Extract scores and compute average
                dimensions = ["relevance", "completeness", "tone", "accuracy", "actionability"]
                scores = {}
                raw_scores = []

                for dim in dimensions:
                    if dim in result:
                        scores[dim] = {
                            "score": int(result[dim]["score"]),
                            "justification": str(result[dim]["justification"]),
                        }
                        raw_scores.append(int(result[dim]["score"]))
                    else:
                        scores[dim] = {"score": 3, "justification": "Dimension not evaluated"}
                        raw_scores.append(3)

                # Normalize average to 0-1 range (scores are 1-5)
                avg = np.mean(raw_scores) / 5.0
                scores["average_score"] = round(float(avg), 4)

                return scores

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                # Fallback on final failure
                return self._fallback_judge_result(str(e))
            except Exception as e:
                time.sleep(2)
                if attempt < max_retries - 1:
                    continue
                return self._fallback_judge_result(str(e))

    def _fallback_judge_result(self, error_msg):
        """Return a neutral fallback result when the LLM judge fails."""
        fallback = {
            "score": 3,
            "justification": f"LLM judge error — default score. Error: {error_msg}",
        }
        return {
            "relevance": fallback.copy(),
            "completeness": fallback.copy(),
            "tone": fallback.copy(),
            "accuracy": fallback.copy(),
            "actionability": fallback.copy(),
            "average_score": 0.6,
            "error": error_msg,
        }
