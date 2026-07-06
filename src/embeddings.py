"""
Embedding engine and similarity-based email retrieval using sentence-transformers + FAISS.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os


class EmailRetriever:
    """Retrieves similar past emails using semantic similarity search."""

    def __init__(self, dataset=None, dataset_path="data/dataset.json"):
        """
        Initialize the retriever.

        Args:
            dataset: List of email dicts to index. If None, loads from dataset_path.
            dataset_path: Path to the dataset JSON file.
        """
        print("  Loading embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if dataset is not None:
            self.dataset = dataset
        else:
            with open(dataset_path) as f:
                self.dataset = json.load(f)

        self._build_index()

    def _build_index(self):
        """Build a FAISS index over the incoming emails in the dataset."""
        emails = [item["incoming_email"] for item in self.dataset]
        print(f"  Building FAISS index over {len(emails)} emails...")
        self.embeddings = self.model.encode(emails, convert_to_numpy=True)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(self.embeddings)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve_similar(self, query_email, top_k=3):
        """
        Find the top-k most similar past emails to the query.

        Args:
            query_email: The incoming email text to match against.
            top_k: Number of similar emails to retrieve.

        Returns:
            List of dicts with similarity score and email data.
        """
        query_embedding = self.model.encode([query_email], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, min(top_k, len(self.dataset)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                {
                    "similarity": float(score),
                    "incoming_email": self.dataset[idx]["incoming_email"],
                    "expected_reply": self.dataset[idx]["expected_reply"],
                    "category": self.dataset[idx]["category"],
                }
            )
        return results

    def encode_text(self, text):
        """Encode text into an embedding vector."""
        return self.model.encode([text], convert_to_numpy=True)[0]
