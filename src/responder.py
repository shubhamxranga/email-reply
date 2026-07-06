"""
RAG-based email response generator using retrieval-augmented few-shot prompting with Gemini.
"""

import google.generativeai as genai
import os
import time


class EmailResponder:
    """Generates suggested email replies using RAG + Gemini few-shot prompting."""

    def __init__(self, retriever):
        """
        Initialize the responder.

        Args:
            retriever: An EmailRetriever instance for finding similar past emails.
        """
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.retriever = retriever

    def generate_reply(self, incoming_email, top_k=3):
        """
        Generate a suggested reply for an incoming email.

        Uses RAG to find similar past email-reply pairs, then constructs a
        few-shot prompt for Gemini to generate a contextually grounded reply.

        Args:
            incoming_email: The customer's email text.
            top_k: Number of similar past emails to use as examples.

        Returns:
            Generated reply text.
        """
        similar_emails = self.retriever.retrieve_similar(incoming_email, top_k=top_k)
        prompt = self._build_prompt(incoming_email, similar_emails)

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # Retry once after a brief pause (rate limiting)
            time.sleep(2)
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as retry_error:
                return f"[Error generating reply: {retry_error}]"

    def _build_prompt(self, incoming_email, similar_emails):
        """Construct a few-shot prompt grounded in similar past interactions."""
        examples = ""
        for i, ex in enumerate(similar_emails, 1):
            examples += f"\n--- Example {i} (similarity: {ex['similarity']:.2f}) ---\n"
            examples += f"Customer Email:\n{ex['incoming_email']}\n\n"
            examples += f"Support Reply:\n{ex['expected_reply']}\n"

        prompt = f"""You are a professional customer support agent for a B2B SaaS company that provides email collaboration and shared inbox tools. Your job is to write helpful, empathetic, and accurate replies to customer emails.

Study these similar past interactions to learn the company's tone, level of detail, and response patterns:
{examples}

Now write a reply to this new customer email. Match the tone and helpfulness of the examples above.

--- New Customer Email ---
{incoming_email}

--- Your Reply ---
Write a professional support reply. Guidelines:
- Address every concern or question raised
- Be empathetic and acknowledge the customer's situation
- Provide specific, actionable next steps
- Be concise but thorough
- Use a warm, professional tone
- Do NOT include a subject line — just the reply body
- Do NOT use placeholder names — if a name is not obvious, use a generic greeting"""

        return prompt
