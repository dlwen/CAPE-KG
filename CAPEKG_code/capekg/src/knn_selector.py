"""
KNN Few-Shot Example Selector.

CAPE-KG decomposes each multi-hop question with a similarity-based
dynamic few-shot prompt (paper Sec. 3.3 + Appendix D, Table 4): given an
incoming question q, the top-k most similar examples from a fixed
decomposition pool (built from MQuAKE-CF after dedup) are appended to a
base instruction template to form the final prompt.

Similarity is computed with frozen sentence-transformer embeddings
(all-MiniLM-L6-v2). Top-k examples are sorted by similarity ascending
so the most-similar example sits closest to the query, following common
ICL practice.
"""

import os
import pickle

from sentence_transformers import SentenceTransformer, util


class KNNFewShotSelector:
    """Dynamic KNN-based few-shot selector for question decomposition."""

    def __init__(self, pools_dir, base_prompt, encoder_name="all-MiniLM-L6-v2"):
        self.pools_dir = pools_dir
        self.base_prompt = base_prompt
        self.decomposition_pool = self._load_pool("decomposition_pool.pkl")
        self.similarity_model = SentenceTransformer(encoder_name)

    def _load_pool(self, filename):
        path = os.path.join(self.pools_dir, filename)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return []

    def select_knn_examples(self, query_text, k=6, text_key="original_question"):
        """Return the k most similar pool entries, sorted ascending by similarity."""
        pool = self.decomposition_pool
        if not pool or k <= 0:
            return []
        cand_texts = [item[text_key] for item in pool]
        q_emb = self.similarity_model.encode([query_text])
        c_emb = self.similarity_model.encode(cand_texts)
        sims = util.cos_sim(q_emb, c_emb)[0]
        indexed = sorted(enumerate(sims), key=lambda x: x[1].item(), reverse=True)[:k]
        # Re-sort ascending so the most similar example is placed last
        indexed.sort(key=lambda x: x[1].item())
        return [pool[i] for i, _ in indexed]

    def build_decomposition_prompt(self, query_question, k=6):
        """Compose the final decomposition prompt for one question.

        Layout:
            <base instruction (rules)>
            <demonstration 1>
            ...
            <demonstration k>
            Question: <query>
            Subquestion:
        """
        clean_query = (query_question
                       .replace("Divide the following multi-hop question into "
                                "sub-questions:\n\nQuestion: ", "")
                       .replace("\nSubquestion:", ""))

        examples = self.select_knn_examples(clean_query, k=k)
        lines = [self.base_prompt.strip(), ""]
        for ex in examples:
            lines.append(f"Question: {ex['original_question']}")
            lines.append("Subquestion:")
            for subq in ex["subquestions"]:
                lines.append(subq)
            lines.append("")
        lines.append(f"Question: {clean_query}")
        lines.append("Subquestion:")
        return "\n".join(lines)
