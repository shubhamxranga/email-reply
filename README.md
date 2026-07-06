# Hiver AI Email Suggested-Response System

This is an end-to-end system built for the Hiver Open Challenge. It takes incoming customer support emails, matches them against past interactions using semantic search (RAG), drafts suggested replies using Gemini, and evaluates the suggestions using a robust multi-metric scoring framework.

---

### How It Was Built

The dataset was generated using structured, category-specific prompts (matching the logic in `data/generate_dataset.py`) to ensure high quality and variation. The raw outputs were then collected and saved directly into `data/dataset.json`. 

The `data/generate_dataset.py` script is included in this repository as a utility to programmatically reproduce or scale the dataset generation process.

---

## Quick Start

### Setup

Make sure you have Python 3.10+ and a Gemini API Key.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hiver-email-ai.git
cd hiver-email-ai

# Set up a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure your Gemini API key
export GEMINI_API_KEY='your-key-here'
```

### Run

```bash
# Run the full pipeline (RAG Retrieval -> Response Generation -> Multi-metric Evaluation)
python main.py run-all
```

To run individual steps:
```bash
python main.py generate-dataset       # Step 1: Generate synthetic support emails (if data/dataset.json is missing)
python main.py generate-responses     # Step 2: Generate suggestions for test emails
python main.py evaluate               # Step 3: Evaluate generated replies
```

---

## System Architecture & Pipeline

```
[Incoming Email] ──▶ [Embed (MiniLM-L6-v2)] ──▶ [FAISS Similarity Search (Top 3 examples)]
                                                              │
[Suggested Reply] ◀── [Gemini 2.5 Flash] ◀── [Few-Shot Prompt Construction]
        │
        └──▶ [Multi-Metric Evaluation (ROUGE-L + Semantic Similarity + LLM-as-Judge)]
```

---

## Dataset

- **Origin**: 100 synthetic customer support email-reply pairs created using Gemini (found in `data/dataset.json`).
- **Categories**: 20 pairs each across **Billing**, **Technical**, **Feature Request**, **Account Management**, and **General Inquiries**.
- **Why Synthetic?**: It bypasses PII/privacy issues inherent in real customer emails, while allowing us to programmatically inject diverse tones (angry, polite, urgent), company sizes (startup to enterprise), and specific product scenarios.

---

## Suggested-Response Generator (Gen AI)

We chose **RAG (Retrieval-Augmented Generation) with Few-Shot prompting** over fine-tuning or zero-shot prompts:
- **Zero-shot** is too generic and lacks contextual grounding in our "company policy."
- **Fine-tuning** requires far more compute, time, and data than what's feasible in a 100-minute challenge.
- **RAG + Few-shot** dynamically finds the most relevant past responses, giving Gemini the specific context, tone, and resolution steps to follow for the new email.

**Data Split**: The system uses a stratified **80/20 train/test split** (taking 4 test emails and 16 training emails from each category). We build the FAISS search index strictly on the training set to prevent data leakage.

---

## Evaluating Accuracy — The Core Challenge

Measuring the quality of an email reply is hard. An exact string match is useless since there are many ways to write a good response. We use three metrics to evaluate each reply:

### 1. ROUGE-L (15% Weight)
- **What it does**: Measures lexical overlap (longest common subsequence).
- **Role**: Ensures the generated response follows the structure and standard templates of our reference replies.
- **Limitation**: Misses paraphrasing (e.g., scoring "Let me look into this" lower than "I will check this").

### 2. Semantic Similarity (20% Weight)
- **What it does**: Computes cosine similarity of sentence embeddings.
- **Role**: Captures the underlying meaning of the reply even when different words are used.
- **Limitation**: Can be fooled if the topic is identical but the action is wrong (e.g., discussing a refund but refusing it instead of approving it).

### 3. LLM-as-a-Judge (65% Weight)
- **What it does**: Prompts Gemini 2.5 Flash to score the reply on a 1-5 scale across 5 custom dimensions:
  1. **Relevance**: Does it directly address the customer's core issues?
  2. **Completeness**: Are all questions answered?
  3. **Tone**: Is it professional, empathetic, and appropriate?
  4. **Accuracy**: Is it factually consistent with the reference reply?
  5. **Actionability**: Are next steps clear?
- **Role**: This is the heavy lifter. Email quality is subjective, and only an LLM can parse human traits like tone, empathy, completeness, and actionability.
- **Self-bias mitigation**: We provide the human-written reference reply as a strong anchor and force the model to output a justification for every score.

### Overall Score
We combine these into a weighted composite score:
$$\text{Composite} = 0.15 \times \text{ROUGE-L} + 0.20 \times \text{Semantic Sim} + 0.65 \times \text{LLM Judge}$$

---

## Project Structure

```
hiver-email-ai/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── main.py                    # CLI entry point
├── data/
│   ├── generate_dataset.py    # Generates dataset using Gemini
│   └── dataset.json           # 100 email-reply pairs
├── src/
│   ├── embeddings.py          # Embedding & FAISS retrieval
│   ├── responder.py           # RAG responder
│   ├── evaluator.py           # Multi-metric evaluation
│   └── reporter.py            # Console & JSON reports
└── results/
    ├── generated_responses.json   # Generated outputs (run-time)
    └── evaluation_report.json     # Detailed evaluation scores (run-time)
```

---

## AI Usage Disclosure
- **Gemini 2.5 Flash**: Handled synthetic dataset generation, email reply draft generation, and LLM-as-judge evaluation.
- **Google Antigravity**: Served as the interactive coding assistant to write the code structure, handle setup, and debug pipeline issues.
