"""
Edit-Aware Progressive Retrieval (paper Sec. 3.3).

Implements the three-stage retrieval pipeline that drives intent
consistency:

  High Confidence Stage
    K' = {k_j in K | g_phi(q_i, k_j) >= tau}.  We then filter low outliers
    inside K' using mean/std around the score distribution
    (the K'' = {k_j in K' | g_phi(q_i, k_j) >= mu - lambda * sigma} rule in
    the paper). Candidates are tried in descending score order; edited
    targets unrelated to the sub-question are down-weighted to suppress
    spurious matches.

  Low Confidence Stage
    Triggered when K' is empty. The LLM is asked to pick the most
    plausible entity from the original candidate pool K.

  Failure Stage
    If no valid entity can be retrieved at all, the sub-question is sent to
    the LLM with the relevant edited triple injected into the prompt
    context (paper Sec. 3.3 + Appendix D, Table 6), so that even the LLM
    fallback stays aligned with the edit.

The function ``retrieve_from_graph`` below realises the High and Low
Confidence stages over a given KG layer (Base or Overlay, decided by the
routing rule in :func:`kg.is_edit_sensitive_question`). The Failure
Stage's LLM call is delegated to ``llm_interface.run_llm_answer``.
"""

import numpy as np

from .detectors import ENTITY, REL, DetectorScorer
from .entity_manager import EntityQIDManager
from .llm_interface import gpt_select_entity_from_candidates
from .rebel_extractor import RebelExtractor


# ---------------------------------------------------------------------------
# Outlier filtering (paper Sec. 3.3 High Confidence Stage, K'' rule)
# ---------------------------------------------------------------------------

def filter_entities_by_statistics(scores, lam=1.0):
    """Apply K'' = {k_j in K' | g_phi(q_i, k_j) >= mu - lam * sigma}.

    Returns the subset of (score, entity) tuples that pass the threshold,
    sorted descending by score. Falls back to the original list if the
    filter would empty it.
    """
    if not scores:
        return []
    values = np.array([s for s, _ in scores], dtype=float)
    mu, sigma = float(values.mean()), float(values.std())
    threshold = mu - lam * sigma
    kept = [(s, e) for s, e in scores if s >= threshold]
    if not kept:
        kept = list(scores)
    kept.sort(key=lambda x: x[0], reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Edit irrelevance suppression
# ---------------------------------------------------------------------------

def suppress_irrelevant_edits(scores, question_lower, current_edit_details):
    """Down-weight edit targets that are NOT directly referenced in the
    current sub-question. Prevents spurious overlay hits when an unrelated
    edited entity shares surface form with a candidate."""
    if not (scores and current_edit_details):
        return scores
    out = []
    for score, entity in scores:
        is_target = any(entity.lower() == d["target_new"].lower()
                        for d in current_edit_details)
        if is_target:
            asking_directly = any(kw in question_lower
                                  for kw in ("who is", "what is", entity.lower()))
            out.append((score * (1.0 if asking_directly else 0.3), entity))
        else:
            out.append((score, entity))
    return out


# ---------------------------------------------------------------------------
# Main retrieval driver
# ---------------------------------------------------------------------------

def retrieve_from_graph(q, qid2name, triples, qid_relations,
                        rebel: RebelExtractor,
                        scorer: DetectorScorer,
                        entity_manager: EntityQIDManager,
                        current_edit_details=None,
                        gpt_model="gpt-3.5-turbo-instruct",
                        lam=1.0):
    """Run High + Low Confidence stages of progressive retrieval over a KG.

    Args:
        q                    Sub-question text.
        qid2name             Entity-id -> surface-name mapping for the layer.
        triples              (qid, relation) -> set of (qid, name).
        qid_relations        qid -> set of relations available for that entity.
        rebel                REBEL triple extractor for candidate proposal.
        scorer               DetectorScorer wrapping g_phi and g_psi.
        entity_manager       Shared QID manager.
        current_edit_details Optional list of edit dicts so that retrieval
                             can prefer edited targets when the question
                             actually asks for them.
        gpt_model            Backbone for the Low Confidence stage's
                             entity selection.
        lam                  Multiplier in the K'' outlier filter rule.

    Returns:
        Predicted answer string, or ``None`` to signal a Failure Stage
        fallback to the LLM (handled by the caller).
    """
    # ---- propose candidate entities via REBEL ------------------------------
    kb = rebel.extract(q)
    entity_set = set()
    for r in kb.relations:
        entity_set.update([r["head"], r["type"], r["tail"]])
    if not entity_set:
        return None

    # ---- score and filter (High Confidence Stage) --------------------------
    scores = scorer.get_scores(q, list(entity_set), ENTITY)
    scores = suppress_irrelevant_edits(scores, q.lower(), current_edit_details)

    head = None
    head_qid = None

    if not scores:
        # ---- Low Confidence Stage: GPT picks from the full pool K ----------
        chosen = gpt_select_entity_from_candidates(q, list(entity_set), gpt_model)
        if chosen:
            cand_qid = entity_manager.get_consistent_qid(chosen, qid2name)
            if cand_qid in qid_relations:
                head, head_qid = chosen, cand_qid
    else:
        # ---- K'' outlier filter and sequential trial -----------------------
        filtered = filter_entities_by_statistics(scores, lam=lam)
        for _, candidate in filtered:
            cand_qid = entity_manager.get_consistent_qid(candidate, qid2name)
            if cand_qid in qid_relations:
                head, head_qid = candidate, cand_qid
                break

    if head is None:
        return None  # signal Failure Stage to caller

    # ---- relation scoring --------------------------------------------------
    available_relations = list(qid_relations.get(head_qid, []))
    if not available_relations:
        return None
    rel_scores = scorer.get_scores(q, available_relations, REL)
    if not rel_scores:
        return None
    rel_scores.sort(key=lambda x: x[0], reverse=True)
    relation = rel_scores[0][1]

    # ---- look up the object, preferring edited target when relevant -------
    targets = list(triples.get((head_qid, relation), []))
    if not targets:
        return None

    if current_edit_details:
        for d in current_edit_details:
            for tail_qid, tail_name in targets:
                if tail_name == d["target_new"] or \
                   qid2name.get(tail_qid, "") == d["target_new"]:
                    return tail_name

    return targets[0][1]
