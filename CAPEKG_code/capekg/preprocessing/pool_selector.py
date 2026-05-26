#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot example pool selector.

Functions:
    1. Select few-shot examples from the prebuilt pools.
    2. Similarity-based selection (sentence-transformer cosine similarity).
    3. Random selection.
    4. Format the final decomposition / answer prompts.

Usage:
    from preprocessing.pool_selector import PoolSelector
    selector = PoolSelector()
    prompt = selector.select_decomposition_prompt(question, num_examples=3)
"""

import os
import pickle
import random
from typing import Dict, List

from sentence_transformers import SentenceTransformer, util


class PoolSelector:
    def __init__(self, pools_dir="preprocessing/pools", use_similarity=True):
        self.pools_dir = pools_dir
        self.use_similarity = use_similarity

        self.decomposition_pool = self.load_pool("decomposition_pool.pkl")
        self.answer_pool = self.load_pool("answer_pool.pkl")

        self.similarity_model = None
        if use_similarity:
            try:
                print("Loading similarity model ...")
                self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Similarity model loaded")
            except Exception as e:
                print(f"Similarity model failed to load, falling back to random: {e}")
                self.use_similarity = False

    def load_pool(self, filename):
        """Load a pickled example pool."""
        filepath = os.path.join(self.pools_dir, filename)
        try:
            with open(filepath, "rb") as f:
                pool = pickle.load(f)
            print(f"Loaded {filename}: {len(pool)} examples")
            return pool
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return []
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return []

    def calculate_similarity(self, query_text: str,
                             candidate_texts: List[str]) -> List[float]:
        """Cosine similarity between the query and each candidate text."""
        if not self.similarity_model or not candidate_texts:
            return [0.0] * len(candidate_texts)
        try:
            query_embedding = self.similarity_model.encode([query_text])
            candidate_embeddings = self.similarity_model.encode(candidate_texts)
            similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
            return similarities.tolist()
        except Exception as e:
            print(f"Similarity computation failed: {e}")
            return [0.0] * len(candidate_texts)

    def select_decomposition_examples(self, query_question: str,
                                      num_examples: int = 3,
                                      strategy: str = "random") -> List[Dict]:
        """Select few-shot examples for question decomposition."""
        if not self.decomposition_pool:
            return []
        num_examples = min(num_examples, len(self.decomposition_pool))

        if strategy == "similarity" and self.use_similarity:
            candidate_questions = [ex["original_question"]
                                   for ex in self.decomposition_pool]
            similarities = self.calculate_similarity(query_question, candidate_questions)
            indexed = list(enumerate(similarities))
            indexed.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in indexed[:num_examples]]
            return [self.decomposition_pool[idx] for idx in selected_indices]
        return random.sample(self.decomposition_pool, num_examples)

    def select_answer_examples(self, num_examples: int = 5,
                               strategy: str = "random") -> List[Dict]:
        """Select few-shot examples for answer generation."""
        if not self.answer_pool:
            return []
        num_examples = min(num_examples, len(self.answer_pool))
        # Currently only random sampling is supported.
        return random.sample(self.answer_pool, num_examples)

    def format_decomposition_prompt(self, selected_examples: List[Dict],
                                    query_question: str,
                                    api_type: str = "instruct") -> str:
        """Format the decomposition prompt."""
        if api_type == "instruct":
            prompt_lines = [
                "Break down multi-hop questions into sequential sub-questions. "
                "Use [ENT] as placeholder for previous answers.",
                "",
            ]
            for example in selected_examples:
                prompt_lines.append(f"Question: {example['original_question']}")
                prompt_lines.append("Subquestion:")
                for subq in example["subquestions"]:
                    prompt_lines.append(subq)
                prompt_lines.append("")
            prompt_lines.append(f"Question: {query_question}")
            prompt_lines.append("Subquestion:")
            return "\n".join(prompt_lines)

        examples_text = []
        for example in selected_examples:
            example_text = (
                f"Question: {example['original_question']}\nSubquestion:\n"
                + "\n".join(example["subquestions"])
            )
            examples_text.append(example_text)
        examples_str = "\n\n".join(examples_text)
        return (
            f"Examples:\n\n{examples_str}\n\n"
            f"Question: {query_question}\nSubquestion:"
        )

    def format_answer_prompt(self, selected_examples: List[Dict],
                             query_question: str,
                             api_type: str = "instruct") -> str:
        """Format the answer-generation prompt."""
        if api_type == "instruct":
            prompt_lines = [
                "Answer with precise entity names only. No explanations.",
                "",
            ]
            for example in selected_examples:
                prompt_lines.append(f"Question: {example['question']}")
                prompt_lines.append(f"Answer: {example['answer']}")
                prompt_lines.append("")
            prompt_lines.append(f"Question: {query_question}")
            prompt_lines.append("Answer:")
            return "\n".join(prompt_lines)

        examples_text = [
            f"Question: {example['question']}\nAnswer: {example['answer']}"
            for example in selected_examples
        ]
        examples_str = "\n\n".join(examples_text)
        return (
            "Answer each question with a short, accurate entity name only.\n\n"
            f"{examples_str}\n\nQuestion: {query_question}\nAnswer:"
        )

    def generate_decomposition_prompt(self, query_question: str,
                                      num_examples: int = 3,
                                      strategy: str = "random",
                                      api_type: str = "instruct") -> str:
        """Generate the full decomposition prompt for one query question."""
        selected_examples = self.select_decomposition_examples(
            query_question, num_examples, strategy
        )
        if not selected_examples:
            if api_type == "instruct":
                return (f"Break down the following multi-hop question into "
                        f"sub-questions:\n\nQuestion: {query_question}\n"
                        f"Subquestion:")
            return (f"Break down the following multi-hop question into "
                    f"sub-questions using [ENT] as placeholder:\n\n"
                    f"Question: {query_question}\nSubquestion:")
        return self.format_decomposition_prompt(selected_examples,
                                                query_question, api_type)

    def generate_answer_prompt(self, query_question: str,
                               num_examples: int = 5,
                               strategy: str = "random",
                               api_type: str = "instruct") -> str:
        """Generate the full answer-generation prompt for one sub-question."""
        selected_examples = self.select_answer_examples(num_examples, strategy)
        if not selected_examples:
            if api_type == "instruct":
                return (f"Answer with precise entity names only.\n\n"
                        f"Question: {query_question}\nAnswer:")
            return (f"Answer the following question with a short, accurate "
                    f"entity name:\n\nQuestion: {query_question}\nAnswer:")
        return self.format_answer_prompt(selected_examples,
                                         query_question, api_type)

    def get_pool_stats(self) -> Dict:
        """Return summary statistics for the loaded pools."""
        return {
            "decomposition_pool_size": len(self.decomposition_pool),
            "answer_pool_size": len(self.answer_pool),
            "similarity_enabled": self.use_similarity,
        }


def test_selector():
    """Smoke test for the pool selector."""
    print("=== PoolSelector smoke test ===")
    selector = PoolSelector()
    print(f"Pool stats: {selector.get_pool_stats()}")

    test_question = "Who is the spouse of the head of government of Japan?"
    print(f"\nDecomposition test (question: {test_question}):")
    print(selector.generate_decomposition_prompt(
        test_question, num_examples=2, strategy="random"
    ))

    test_subquestion = "Who is the head of government of Japan?"
    print(f"\nAnswer test (sub-question: {test_subquestion}):")
    print(selector.generate_answer_prompt(test_subquestion, num_examples=3))


if __name__ == "__main__":
    test_selector()
