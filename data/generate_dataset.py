"""
Synthetic dataset generator for customer support email pairs.

Generates diverse email-reply pairs across 5 categories using Gemini,
simulating realistic customer support interactions for a B2B SaaS
email collaboration tool (similar to Hiver).

Why synthetic?
- Real customer emails contain PII and are not publicly available
- Synthetic data lets us control diversity, edge cases, and ground truth
- We can ensure coverage across categories, tones, and complexity levels
- The generation prompt is transparent and reproducible
"""

import google.generativeai as genai
import json
import os
import time


# Each category maps to a description used in the generation prompt
CATEGORIES = {
    "billing": (
        "billing and payment issues — refunds, unexpected charges, plan upgrades "
        "or downgrades, payment method failures, invoice requests, subscription "
        "cancellations, prorated charges, enterprise pricing questions"
    ),
    "technical": (
        "technical problems — bugs, error messages, app crashes, email sync "
        "failures, integration issues (Slack, Salesforce, Zapier), login problems, "
        "slow performance, shared inbox not updating, notification issues, "
        "mobile app glitches"
    ),
    "feature_request": (
        "feature requests and product feedback — new feature suggestions, "
        "workflow improvements, integration wishes, UI/UX feedback, automation "
        "requests, reporting enhancements, API access, customization options"
    ),
    "account": (
        "account management — password resets, adding/removing team members, "
        "permission and role changes, account merging, data export requests, "
        "SSO setup, account ownership transfer, GDPR data requests, "
        "onboarding new departments"
    ),
    "general_inquiry": (
        "general inquiries — pricing comparisons, product capabilities, "
        "onboarding and setup help, best practices for shared inboxes, "
        "migration from other tools, security and compliance questions, "
        "training resources, SLA details"
    ),
}


def generate_dataset(output_path="data/dataset.json", emails_per_category=10):
    """
    Generate a synthetic customer support email dataset using Gemini.

    Args:
        output_path: Path to save the generated dataset JSON.
        emails_per_category: Number of email pairs per category.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    dataset = []

    for category, description in CATEGORIES.items():
        print(f"  Generating {emails_per_category} emails for: {category}...")

        prompt = f"""Generate {emails_per_category} realistic customer support email exchanges for a B2B SaaS company called "CollabMail" that provides shared inbox and email collaboration tools (similar to Hiver, Front, or Help Scout — teams share inboxes like support@, sales@ inside Gmail).

Category: {category}
Topics to cover: {description}

Requirements for DIVERSITY — each email pair should be different:
- Vary company sizes: startups (1-10 people), mid-market (50-500), enterprise (1000+)
- Vary tone: frustrated/angry, polite/professional, confused, urgent/panicked, casual
- Vary length: some are 2-3 sentences, others are detailed paragraphs
- Vary technical sophistication: some customers are tech-savvy, others are not
- Include specific product details (e.g., "shared inbox for support@", "assignment rules", "collision detection", "email templates", "analytics dashboard")
- Some emails should mention competitor tools or context about their setup

Requirements for REPLIES:
- Professional, empathetic, and helpful
- Address every specific point the customer raised
- Provide concrete next steps or solutions
- Appropriate level of detail for the question
- Include things like: links to help articles (use realistic URLs), steps to follow, timelines for resolution
- Some replies should apologize, some should educate, some should escalate

Output EXACTLY a JSON array (no markdown, no code blocks, no explanatory text):
[
    {{
        "incoming_email": "Subject: ... \\n\\nHi team,\\n\\n...",
        "expected_reply": "Hi [Name],\\n\\n..."
    }}
]

Generate exactly {emails_per_category} diverse exchanges."""

        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()

                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if "```" in text:
                        text = text.rsplit("```", 1)[0]
                    text = text.strip()

                emails = json.loads(text)

                # Validate structure
                for email in emails:
                    assert "incoming_email" in email, "Missing incoming_email"
                    assert "expected_reply" in email, "Missing expected_reply"
                    assert len(email["incoming_email"]) > 20, "Email too short"
                    assert len(email["expected_reply"]) > 20, "Reply too short"

                for i, email in enumerate(emails):
                    dataset.append(
                        {
                            "id": f"{category}_{i + 1:02d}",
                            "category": category,
                            "incoming_email": email["incoming_email"],
                            "expected_reply": email["expected_reply"],
                        }
                    )

                print(f"    ✓ Generated {len(emails)} email pairs")
                break

            except (json.JSONDecodeError, AssertionError, KeyError) as e:
                wait = [3, 8, 15, 30][min(attempt, 3)]
                print(f"    ✗ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"    ⚠ Skipping category {category} after {max_retries} failures")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait = [10, 30, 60, 120][min(attempt, 3)]
                    print(f"    ⏳ Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                else:
                    wait = [3, 8, 15, 30][min(attempt, 3)]
                    print(f"    ✗ API error on attempt {attempt + 1}: {error_str[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    print(f"    ⚠ Skipping category {category} after {max_retries} failures")

        # Rate limiting — generous pause between categories
        time.sleep(5)

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n  Dataset saved to {output_path}")
    print(f"  Total email pairs: {len(dataset)}")
    print(f"  Categories: {len(CATEGORIES)}")

    # Print category distribution
    from collections import Counter
    cats = Counter(item["category"] for item in dataset)
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    return dataset


if __name__ == "__main__":
    generate_dataset()
