"""
Utility helpers used by the main pipeline.

  - verify_gold_path: hop-wise accuracy check used by the H-Acc metric.
  - normalize_answer: lowercase / strip wrapper for final-answer matching.
"""


def normalize_answer(answer):
    """Lowercase + strip a final answer, handling dict-style answer fields."""
    if isinstance(answer, dict):
        answer = answer.get("str", "") or answer.get("name", "")
    return str(answer).lower().strip()


def verify_gold_path(predicted_entities, hops, gold_path):
    """Check whether every intermediate hop matches its annotated gold answer.

    Used to compute Hop-wise Answering Accuracy (H-Acc), the paper's
    secondary metric that rewards fully-correct reasoning chains.
    """
    if len(predicted_entities) != hops:
        return False
    for i in range(hops):
        if i >= len(gold_path):
            return False
        answer = gold_path[i]["answer"]
        aliases = gold_path[i].get("answer_alias", [])
        pe_low = predicted_entities[i].lower()
        a_low = answer.lower()
        if (pe_low == a_low or a_low in pe_low or pe_low in a_low or
                any(pe_low in al.lower() for al in aliases) or
                any(al.lower() in pe_low for al in aliases)):
            continue
        return False
    return True
