#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot example pool builder, based on the MQuAKE-CF dataset.

Functions:
    1. Build the question-decomposition pool: original question -> subquestion sequence.
    2. Build the answer-generation pool: single-hop question -> answer.
    3. Replace entities with the [ENT] placeholder.
    4. Support multiple top-level question variants per case.

Usage:
    python preprocessing/build_fewshot_pools.py
"""

import json
import os
import pickle


class FewShotPoolBuilder:
    def __init__(
        self,
        cf_for_decomp_path="../datasets/MQuAKE-CF_nonduplicate.json",
        cf_for_answer_path="../datasets/MQuAKE-CF_answer_pool_nonduplicate.json",
        output_dir="preprocessing/preprocessing/pools",
    ):
        self.cf_for_decomp_path = cf_for_decomp_path
        self.cf_for_answer_path = cf_for_answer_path
        self.output_dir = output_dir
        self.decomposition_pool = []
        self.answer_pool = []

        os.makedirs(output_dir, exist_ok=True)

    def load_cf_data(self, path):
        """Load a dataset from the given path."""
        print(f"Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} cases")
        return data

    def replace_entity_with_placeholder(self, question, entity_to_replace):
        """Replace ``entity_to_replace`` in ``question`` with the [ENT] placeholder."""
        if not entity_to_replace or not question:
            return question
        if entity_to_replace in question:
            return question.replace(entity_to_replace, "[ENT]")
        return question

    def build_decomposition_examples(self, case):
        """Build decomposition examples for a single case."""
        examples = []
        original_questions = case.get("questions", [])
        single_hops = case.get("single_hops", [])
        if not original_questions or not single_hops:
            return examples

        subquestions = []
        for i, hop in enumerate(single_hops):
            hop_question = hop.get("question", "")
            if i == 0:
                subquestions.append(hop_question)
            else:
                prev_answer = single_hops[i - 1].get("answer", "")
                modified_question = self.replace_entity_with_placeholder(
                    hop_question, prev_answer
                )
                subquestions.append(modified_question)

        for original_q in original_questions:
            examples.append({
                "original_question": original_q,
                "subquestions": subquestions.copy(),
            })
        return examples

    def build_answer_examples(self, case):
        """Build answer-generation examples for a single case."""
        examples = []
        for hop in case.get("single_hops", []):
            question = hop.get("question", "")
            answer = hop.get("answer", "")
            if question and answer:
                examples.append({"question": question, "answer": answer})
        return examples

    def process_dataset(self):
        """Build the decomposition and answer pools from the configured sources."""
        print("\nBuilding few-shot example pools ...")

        decomp_data = self.load_cf_data(self.cf_for_decomp_path)
        answer_data = self.load_cf_data(self.cf_for_answer_path)

        decomposition_count = 0
        answer_count = 0
        for case in decomp_data:
            decomp_examples = self.build_decomposition_examples(case)
            self.decomposition_pool.extend(decomp_examples)
            decomposition_count += len(decomp_examples)
        for case in answer_data:
            answer_examples = self.build_answer_examples(case)
            self.answer_pool.extend(answer_examples)
            answer_count += len(answer_examples)

        print(f"Decomposition examples: {decomposition_count}")
        print(f"Answer examples: {answer_count}")
        return decomposition_count, answer_count

    def analyze_pools(self):
        """Print pool statistics."""
        print("\n=== Pool statistics ===")
        if self.decomposition_pool:
            print(f"Decomposition pool size: {len(self.decomposition_pool)}")
        if self.answer_pool:
            print(f"Answer pool size:        {len(self.answer_pool)}")

    def show_examples(self, num_examples=3):
        """Show a few examples from each pool."""
        print(f"\n=== Sample examples (first {num_examples} of each) ===")

        print("\nDecomposition examples:")
        for i, example in enumerate(self.decomposition_pool[:num_examples], 1):
            print(f"\n  Example {i}:")
            print(f"    Original question: {example['original_question']}")
            print(f"    Subquestion sequence:")
            for j, subq in enumerate(example["subquestions"], 1):
                print(f"      {j}. {subq}")

        print("\nAnswer examples:")
        for i, example in enumerate(self.answer_pool[:num_examples], 1):
            print(f"\n  Example {i}:")
            print(f"    Question: {example['question']}")
            print(f"    Answer:   {example['answer']}")

    def save_pools(self):
        """Save both pools as JSON (human-readable) and pickle (program-loadable)."""
        print(f"\nSaving pools to {self.output_dir} ...")

        decomp_json_path = os.path.join(self.output_dir, "decomposition_pool.json")
        answer_json_path = os.path.join(self.output_dir, "answer_pool.json")
        with open(decomp_json_path, "w", encoding="utf-8") as f:
            json.dump(self.decomposition_pool, f, indent=2, ensure_ascii=False)
        with open(answer_json_path, "w", encoding="utf-8") as f:
            json.dump(self.answer_pool, f, indent=2, ensure_ascii=False)

        decomp_pkl_path = os.path.join(self.output_dir, "decomposition_pool.pkl")
        answer_pkl_path = os.path.join(self.output_dir, "answer_pool.pkl")
        with open(decomp_pkl_path, "wb") as f:
            pickle.dump(self.decomposition_pool, f)
        with open(answer_pkl_path, "wb") as f:
            pickle.dump(self.answer_pool, f)

        print("Saved:")
        print(f"  Decomposition pool: {decomp_json_path}, {decomp_pkl_path}")
        print(f"  Answer pool:        {answer_json_path}, {answer_pkl_path}")

    def build_pools(self):
        """End-to-end pool construction."""
        print("=== Few-shot example pool builder ===")
        decomp_count, answer_count = self.process_dataset()
        self.analyze_pools()
        self.show_examples()
        self.save_pools()
        print("\nDone.")
        print(f"Final counts: decomposition={decomp_count}, answer={answer_count}")
        print(f"Output directory: {self.output_dir}")
        return self.decomposition_pool, self.answer_pool


def main():
    """Entry point with fixed input/output paths."""
    builder = FewShotPoolBuilder()
    return builder.build_pools()


if __name__ == "__main__":
    main()
