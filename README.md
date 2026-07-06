# Hiver AI Email Suggested-Response System

An AI-powered system that generates suggested email replies for customer support, using **RAG (Retrieval-Augmented Generation)** with semantic search over past email interactions. Includes a comprehensive **multi-metric evaluation framework** that measures reply quality across lexical, semantic, and human-judgment dimensions.

Built for the [Hiver Open Challenge](https://hiver.com).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Response Generator](#response-generator)
- [Accuracy & Evaluation](#accuracy--evaluation-the-core)
- [Trade-offs & Design Decisions](#trade-offs--design-decisions)
- [How AI Tools Were Used](#how-ai-tools-were-used)

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one free](https://aistudio.google.com/apikey))

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hiver-email-ai.git
cd hiver-email-ai

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY='your-key-here'
```

### Run Everything

```bash
# Generate dataset + generate responses + evaluate (full pipeline)
python main.py run-all

# Or run each step individually:
python main.py generate-dataset       # Step 1: Create synthetic email dataset
python main.py generate-responses     # Step 2: Generate AI replies for test emails
python main.py evaluate               # Step 3: Score all generated replies
```

### Options

```bash
python main.py run-all --test-split 0.3          # Use 30% of dataset for testing
python main.py generate-dataset --per-category 5 # Faster: 5 emails per category
```

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Incoming    │────▶│  Embedding       │────▶│  FAISS Retrieval  │
│  Email       │     │  (MiniLM-L6-v2)  │     │  (Top-3 Similar)  │
└──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Few-Shot Prompt  │
                                              │  Construction     │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Gemini 2.0 Flash │
                                              │  (Generation)     │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Suggested Reply  │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Multi-Metric     │
                                              │  Evaluation       │
                                              │  ┌──────────────┐ │
                                              │  │ ROUGE-L      │ │
                                              │  │ Semantic Sim  │ │
                                              │  │ LLM Judge    │ │
                                              │  └──────────────┘ │
                                              └──────────────────┘
```

**Pipeline:**
1. **Embed** the incoming email using `all-MiniLM-L6-v2`
2. **Retrieve** the 3 most similar past emails from the dataset via FAISS
3. **Construct** a few-shot prompt with those email-reply pairs as examples
4. **Generate** a reply using Gemini 2.0 Flash, grounded in the retrieved examples
5. **Evaluate** the generated reply against the reference using 3 complementary metrics

---

## Dataset

### What It Is

50 synthetic customer support email-reply pairs, generated using Gemini, across 5 categories:

| Category | Count | Examples |
|---|---|---|
| **Billing** | 10 | Refunds, unexpected charges, plan changes, invoice requests |
| **Technical** | 10 | Bugs, sync failures, integration issues, login problems |
| **Feature Request** | 10 | New features, workflow improvements, API access |
| **Account** | 10 | Password resets, team changes, permissions, data export |
| **General Inquiry** | 10 | Pricing, onboarding, migration, security questions |

### Why Synthetic?

1. **Privacy**: Real customer emails contain PII and are not publicly shareable
2. **Control**: We can ensure diversity across tones (frustrated → polite), company sizes (startup → enterprise), and technical sophistication
3. **Ground Truth**: Each pair has a known-good reference reply, essential for evaluation
4. **Transparency**: The generation prompt is in `data/generate_dataset.py` — fully reproducible

### How It Was Built

The script `data/generate_dataset.py` uses Gemini to generate 10 email pairs per category. Each prompt specifies:
- Diverse company sizes and customer personas
- Varied emotional tones and email lengths
- Specific product details (shared inboxes, assignment rules, collision detection)
- Professional reply standards (empathy, specificity, clear next steps)

### Representativeness

The dataset simulates a B2B SaaS email collaboration tool's support inbox. While synthetic, the emails:
- Cover the most common support categories (billing, technical, account, features, general)
- Include realistic product-specific terminology
- Vary in complexity from simple password resets to complex enterprise integration issues
- Reflect real customer emotions (frustration, urgency, confusion, satisfaction)

---

## Response Generator

### Approach: RAG + Few-Shot Prompting

I chose **Retrieval-Augmented Generation (RAG)** over alternatives for these reasons:

| Approach | Pros | Cons | Why Not? |
|---|---|---|---|
| **RAG + Few-Shot** ✅ | Grounded in real data, no training needed, adapts to dataset | Retrieval quality matters | **Chosen**: best accuracy-to-effort ratio |
| Zero-shot prompting | Simple, no data needed | Generic, not grounded in company patterns | Replies won't match company voice |
| Fine-tuning | Best quality ceiling | Needs 1000+ examples, compute, time | Not feasible in 100 minutes |
| Template matching | Fast, predictable | Rigid, can't handle novel situations | Too limited for diverse emails |

### How It Works

1. **Semantic Search**: Encode the incoming email with `all-MiniLM-L6-v2`, then find the 3 most similar past emails using FAISS (cosine similarity on L2-normalized vectors)
2. **Few-Shot Prompt**: Construct a prompt that includes the retrieved email-reply pairs as examples, teaching the model the expected tone, structure, and level of detail
3. **Generation**: Gemini 2.0 Flash generates a reply grounded in those examples

### Train/Test Split

The dataset is split **80/20** (40 train, 10 test):
- **Training set**: Used as the retrieval corpus for RAG — these are the "past emails" the system learns from
- **Test set**: Held out completely — replies are generated and then scored against the reference

This prevents **data leakage**: the system never sees the test email's reference reply during generation.

---

## Accuracy & Evaluation — The Core

### What Does "Accurate" Mean for Email Replies?

This is the key question. Email reply accuracy is **fundamentally different** from classification accuracy:

- There's **no single correct answer** — many valid replies exist for any email
- A reply can be **factually correct but tonally wrong** (too formal for a frustrated customer)
- A reply can be **well-written but incomplete** (misses one of three questions asked)
- **Exact match is meaningless** — "I'll help you" and "Let me assist you" are equivalent

This is why I use **three complementary metrics** that capture different aspects of quality:

### Metric 1: ROUGE-L (Weight: 15%)

**What it measures**: Longest Common Subsequence (LCS) between generated and reference text — lexical overlap.

**Why include it**: Detects when the generated reply uses the same key phrases, template structure, or specific terminology as the reference. Good at catching structural similarity.

**Why only 15% weight**: ROUGE has well-documented limitations:
- "I'll refund your payment" vs "Your payment will be refunded" scores low despite being equivalent
- Penalizes paraphrasing and alternative valid phrasings
- Can't distinguish helpful content from verbose padding

### Metric 2: Semantic Similarity (Weight: 20%)

**What it measures**: Cosine similarity between sentence embeddings (`all-MiniLM-L6-v2`) of the generated and reference replies.

**Why include it**: Captures meaning beyond exact words. "I've escalated this to our billing team" and "This has been forwarded to the payments department" score high, as they should.

**Why only 20% weight**: Semantic similarity can be deceived:
- Two replies about "billing" might score high even if one offers a refund and the other requests payment
- It measures topical alignment, not functional correctness
- Doesn't evaluate tone, completeness, or actionability

### Metric 3: LLM-as-Judge (Weight: 65%)

**What it measures**: Gemini evaluates the generated reply on 5 dimensions, each scored 1-5:

| Dimension | What It Captures |
|---|---|
| **Relevance** | Does the reply address the *specific* concerns raised? |
| **Completeness** | Are *all* questions and issues answered? |
| **Tone** | Professional, empathetic, and appropriate for the situation? |
| **Accuracy** | Factually consistent with the reference reply? |
| **Actionability** | Clear next steps the customer can follow? |

**Why 65% weight**: This is the **most important metric** because:
1. It evaluates **functional qualities** that automated metrics fundamentally cannot capture (tone, completeness, actionability)
2. It provides **per-dimension justifications**, making scores interpretable and debuggable
3. It aligns closest with how a human QA reviewer would evaluate a support reply
4. Research (e.g., [Zheng et al., 2023 "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685)) shows strong correlation between LLM judges and human preferences

**Potential concern — self-evaluation bias**: The same model family (Gemini) generates and judges replies. I mitigate this by:
- Providing the reference reply as an anchor for judgment
- Using structured scoring with specific dimensions (not just "is this good?")
- Requiring justifications (forces the model to reason, reducing inflated scores)

### Composite Score

```
composite = 0.15 × ROUGE-L + 0.20 × Semantic Similarity + 0.65 × LLM Judge (normalized)
```

The weighting reflects that **email quality is primarily about functional correctness** (did it help the customer?), with lexical and semantic similarity as supporting signals.

### Validation: Why These Metrics Reflect Real Quality

1. **ROUGE catches template adherence**: When the company has standard phrases ("Thanks for reaching out"), ROUGE detects whether the AI uses them
2. **Semantic similarity catches topical alignment**: Ensures the reply is about the right subject (billing vs. technical)
3. **LLM Judge catches what matters most**: Did the reply actually address the customer's problem? Is the tone right? Are next steps clear?

Together, they form a **multi-signal evaluation** where each metric compensates for the others' blind spots.

### Output

The system produces:
- **Per-response scores**: All 3 metrics + 5 LLM judge dimensions with justifications
- **Per-category breakdown**: Mean and median composite scores by category
- **Overall system score**: Mean, median, std, min, max across all responses
- **Full JSON report**: `results/evaluation_report.json`

---

## Trade-offs & Design Decisions

| Decision | Alternative | Why I Chose This |
|---|---|---|
| Gemini 2.0 Flash | GPT-4, Claude, local LLM | Free, fast, good quality — maximizes iteration speed |
| all-MiniLM-L6-v2 | Larger embedding models | Fast loading, good quality for this task, widely validated |
| FAISS (flat index) | Approximate NN, vector DB | Dataset is small (50 items) — exact search is fast and correct |
| 80/20 train/test split | Cross-validation | Simple, sufficient for 50 items, avoids complexity |
| Synthetic dataset | Enron corpus, public datasets | Enron is corporate email (not support), synthetic gives us ground truth |
| 3 metrics, not 1 | Single score | Each metric has known failure modes — combining them is more robust |

---

## Project Structure

```
hiver-email-ai/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── main.py                    # CLI entry point (4 commands)
├── data/
│   ├── generate_dataset.py    # Dataset generation script (uses Gemini)
│   └── dataset.json           # Generated dataset (50 email pairs)
├── src/
│   ├── __init__.py
│   ├── embeddings.py          # Sentence embeddings + FAISS retrieval
│   ├── responder.py           # RAG-based response generator
│   ├── evaluator.py           # Multi-metric evaluation engine
│   └── reporter.py            # Results formatting and output
└── results/
    ├── generated_responses.json   # AI-generated replies (runtime)
    └── evaluation_report.json     # Full evaluation report (runtime)
```

---

## How AI Tools Were Used

- **Google Antigravity (AGY)**: Used as a pair-programming assistant to scaffold the project structure, write code, debug issues, and iterate quickly within the 100-minute window. All code was reviewed, understood, and directed by me.
- **Google Gemini 2.0 Flash**: Powers three components:
  1. **Dataset generation** — Creates synthetic email-reply pairs
  2. **Response generation** — Generates suggested replies (the core feature)
  3. **LLM-as-Judge evaluation** — Scores generated replies on 5 quality dimensions

---

## Dependencies

| Package | Purpose |
|---|---|
| `google-generativeai` | Gemini API client (generation + evaluation) |
| `sentence-transformers` | Text embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Fast vector similarity search |
| `rouge-score` | ROUGE-L metric computation |
| `numpy` | Numerical operations |

---

*Built for the Hiver Open Challenge — 100 minutes on the clock.*
