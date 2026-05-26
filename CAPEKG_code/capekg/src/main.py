"""
CAPE-KG End-to-End Pipeline.

This script wires together the modules in this package to reproduce the
paper's main experiment on MQuAKE-CF-3K / MQuAKE-T. The overall control
flow corresponds to Figure 1 in the paper:

  for each batch of edit cases:
      build the read-only Base KG B from MQuAKE-CF target_true / single_hops
      for each case c in the batch:
          build the per-case Overlay O_c from c.requested_rewrite
          compute the edit impact surface (S^c_edit, P^c_edit)
          for each question Q of case c:
              decompose Q into sub-questions {q_1, ..., q_n}
              for each sub-question q_i:
                  route to Overlay if q_i is edit-sensitive (Eq. 1)
                  else route to Base
                  run progressive retrieval (Sec. 3.3)
                  fall back to edit-aware LLM if retrieval fails
              compare predicted chain against gold path

API credentials must be set in the environment (OPENAI_API_KEY); no
secrets are hard-coded in this code.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from .detectors import Detector, DetectorScorer
from .entity_manager import EntityQIDManager
from .kg import (
    build_base_kg_from_dataset,
    build_case_overlay_from_edits,
    compute_edit_influence_sets,
    is_edit_sensitive_question,
)
from .knn_selector import KNNFewShotSelector
from .llm_interface import configure_openai, run_llm_answer, run_llm_divide
from .rebel_extractor import RebelExtractor
from .retrieval import retrieve_from_graph
from .utils import normalize_answer, verify_gold_path


# ---------------------------------------------------------------------------
# Edit-aware LLM fallback (paper Sec. 3.3 Failure Stage)
# ---------------------------------------------------------------------------

RELATION_DESC = {
    "P6": "head of government",
    "P27": "country of citizenship",
    "P39": "position held",
    "P35": "head of state",
    "P159": "headquarters location",
}


def build_edit_aware_fallback_prompt(answer_template, sub_question,
                                     case_edits):
    """Inject relevant edits into the LLM answer prompt (Table 6).

    Only edits whose subject or relation type is actually referenced in
    the current sub-question are injected; this keeps the fallback
    aligned with the edit intent without leaking unrelated edit
    information into other sub-questions.
    """
    facts = []
    q_low = sub_question.lower()
    for d in case_edits:
        subj = d["subject"]
        if subj.lower() not in q_low:
            continue
        rel = d["relation"]
        rel_desc = RELATION_DESC.get(rel, f"relation {rel}")
        new_target = d["target_new"]
        facts.append(f"Note: {subj} {rel_desc} {new_target}.")

    context = ("\n" + "\n".join(facts[:3]) + "\n") if facts else ""
    return f"{answer_template}\n\nQuestion: {context}{sub_question}\nAnswer:"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args):
    # ---- deterministic seeding ---------------------------------------------
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    configure_openai()

    # ---- load models -------------------------------------------------------
    rebel = RebelExtractor(args.rebel_path, device=args.device)
    entity_det = Detector(args.entity_detector_path, device=args.device)
    relation_det = Detector(args.relation_detector_path, device=args.device)
    scorer = DetectorScorer(entity_det, relation_det, threshold=args.tau)

    entity_manager = EntityQIDManager()

    # ---- load decomposition prompt + KNN selector --------------------------
    with open(args.divide_prompt_path, "r", encoding="utf-8") as f:
        divide_template = f.read()
    base_divide_prompt = divide_template.split("\nExamples:")[0].rstrip() + "\n\n"
    knn = KNNFewShotSelector(args.pools_dir, base_divide_prompt)

    with open(args.answer_prompt_path, "r", encoding="utf-8") as f:
        answer_template = f.read()

    # ---- Base KG B (paper Sec. 3.1) ----------------------------------------
    base_qid2name, base_triples, base_qid_relations = build_base_kg_from_dataset(
        args.base_kg_dataset, entity_manager
    )

    # ---- dataset -----------------------------------------------------------
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    # ---- evaluation counters ----------------------------------------------
    total, m_correct, h_correct = 0, 0, 0

    for batch_start in range(0, len(data), args.edit):
        batch = data[batch_start: batch_start + args.edit]

        for case in batch:
            case_edits = case.get("requested_rewrite", [])

            # Overlay + edit impact surface for this case (Sec. 3.1 + Eq. 1)
            overlay_qid2name, overlay_triples, overlay_qid_relations = \
                build_case_overlay_from_edits(case_edits, base_qid2name,
                                              entity_manager)
            s_edit, p_edit = compute_edit_influence_sets(case_edits)

            for question_item in case["questions"]:
                question = question_item if isinstance(question_item, str) \
                                          else question_item["question"]
                total += 1

                # ---- decompose multi-hop question into sub-questions -------
                divide_prompt = knn.build_decomposition_prompt(
                    f"Divide the following multi-hop question into "
                    f"sub-questions:\n\nQuestion: {question}\nSubquestion:",
                    k=args.knn_k,
                )
                divided = run_llm_divide(divide_prompt, model=args.gpt_model)
                sub_questions = [s.strip() for s in divided.split("\n")
                                 if s.strip()]

                # ---- iterate hops ------------------------------------------
                predicted_entities = []
                for sq in sub_questions:
                    processed_sq = sq
                    if "[ENT]" in sq and predicted_entities:
                        processed_sq = sq.replace("[ENT]", predicted_entities[-1])

                    # ---- edit-aware routing (paper Eq. 1) ------------------
                    use_overlay = is_edit_sensitive_question(
                        processed_sq, s_edit, p_edit
                    )

                    if use_overlay:
                        graph_answer = retrieve_from_graph(
                            processed_sq,
                            overlay_qid2name, overlay_triples, overlay_qid_relations,
                            rebel=rebel, scorer=scorer,
                            entity_manager=entity_manager,
                            current_edit_details=case_edits,
                            gpt_model=args.gpt_model, lam=args.lam,
                        )
                    else:
                        graph_answer = retrieve_from_graph(
                            processed_sq,
                            base_qid2name, base_triples, base_qid_relations,
                            rebel=rebel, scorer=scorer,
                            entity_manager=entity_manager,
                            current_edit_details=None,
                            gpt_model=args.gpt_model, lam=args.lam,
                        )

                    if graph_answer is None:
                        # Failure stage: LLM fallback with edit injection
                        prompt = build_edit_aware_fallback_prompt(
                            answer_template, processed_sq, case_edits,
                        )
                        graph_answer = run_llm_answer(prompt, model=args.gpt_model)

                    predicted_entities.append(graph_answer)

                # ---- score the case ----------------------------------------
                final_pred = predicted_entities[-1] if predicted_entities else ""
                gold_answer = normalize_answer(case["new_answer"])
                aliases = [a.lower() for a in case.get("new_answer_alias", [])]

                m_acc = (
                    normalize_answer(final_pred) == gold_answer
                    or gold_answer in normalize_answer(final_pred)
                    or any(normalize_answer(final_pred) in a for a in aliases)
                )
                if m_acc:
                    m_correct += 1

                h_acc = verify_gold_path(predicted_entities,
                                         len(case["single_hops"]),
                                         case["new_single_hops"])
                if h_acc:
                    h_correct += 1

    print(f"M-Acc: {m_correct / total:.4f}  ({m_correct}/{total})")
    print(f"H-Acc: {h_correct / total:.4f}  ({h_correct}/{total})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", default="../datasets/MQuAKE-CF-3k.json")
    p.add_argument("--base_kg_dataset", default="../datasets/MQuAKE-CF.json")
    p.add_argument("--edit", type=int, default=1,
                   help="Batch size (number of cases per editing batch).")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--knn_k", type=int, default=6)
    p.add_argument("--tau", type=float, default=0.4,
                   help="Detector confidence threshold (paper Sec. 3.3).")
    p.add_argument("--lam", type=float, default=1.0,
                   help="Outlier multiplier in K'' filter (paper Sec. 3.3).")
    p.add_argument("--gpt_model", default="gpt-3.5-turbo-instruct")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--rebel_path", default="../model/rebel-large")
    p.add_argument("--entity_detector_path",
                   default="../train/results_entity_judge/best_model_entity_judge")
    p.add_argument("--relation_detector_path",
                   default="../train/results/results/best_model")
    p.add_argument("--pools_dir",
                   default="../preprocessing/preprocessing/pools")
    p.add_argument("--divide_prompt_path", default="../prompts/divide_improved.txt")
    p.add_argument("--answer_prompt_path", default="../prompts/answer_improved.txt")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
