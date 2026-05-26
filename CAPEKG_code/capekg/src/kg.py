"""
Multi-Layer Knowledge Graph Construction and Update (paper Sec. 3.1 + 3.2).

The CAPE-KG framework maintains a two-tier knowledge graph for each batch
of edit cases:

  - Base layer  B   : built only from factual triples F (target_true facts
                      and single_hops chains derived from MQuAKE-CF).
                      Shared across all cases in the batch. Read-only.

  - Overlay  O_c    : built only from edits associated with case c. The
                      same (subject, relation) pair is overwritten with the
                      new target_new value (paper Sec. 3.2's deterministic
                      conflict arbitration). Visible only to case c.

The separation guarantees the knowledge-boundary consistency formalised in
the paper: factual queries hit B, edit-sensitive queries hit O_c, and edits
never propagate across cases.

Routing (paper Eq. 1) is implemented by ``is_edit_sensitive_question`` which
checks whether a sub-question references an entity in S^c_edit (edited
subjects/targets) or a relation in P^c_edit (edited relation ids).
"""

import json

from .entity_manager import EntityQIDManager


# ---------------------------------------------------------------------------
# Base layer construction
# ---------------------------------------------------------------------------

def build_base_kg_from_dataset(dataset_path, entity_manager: EntityQIDManager):
    """Build the read-only Base layer B from the MQuAKE-CF dataset.

    Two sources of factual triples are extracted:
      (i)  Each requested_rewrite's ``target_true`` field, which gives a
           canonical (subject, relation_id, target_true) triple before any
           edit is applied.
      (ii) Each case's ``single_hops`` chain, where the answer of each hop
           becomes the subject of the next.

    Returns:
        (qid2name, triples, qid_relations)
        where ``triples`` maps (subj_qid, relation) -> set of (obj_qid, name)
        and ``qid_relations`` maps subj_qid -> set of relations.
    """
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}, {}, {}

    qid2name, triples, qid_relations = {}, {}, {}
    entity_manager.update_from_existing_kg(qid2name)

    for item in data:
        # (i) target_true triples
        for edit in item.get("requested_rewrite", []):
            subject = edit.get("subject", "").strip()
            relation_id = edit.get("relation_id", "unknown")
            target_true = (edit.get("target_true") or {}).get("str", "").strip()
            if not (subject and relation_id and target_true):
                continue

            s_qid = entity_manager.get_consistent_qid(subject, qid2name)
            o_qid = entity_manager.get_consistent_qid(target_true, qid2name)
            qid2name.setdefault(s_qid, subject)
            qid2name.setdefault(o_qid, target_true)

            triples.setdefault((s_qid, relation_id), set()).add((o_qid, target_true))
            qid_relations.setdefault(s_qid, set()).add(relation_id)

        # (ii) single_hops chains
        hops = item.get("single_hops", [])
        if hops and isinstance(hops[0], dict):
            context_subject = (hops[0].get("answer") or "").strip() or None
        else:
            context_subject = None

        for hop in hops:
            q_text = (hop.get("cloze") or "").strip()
            a_text = (hop.get("answer") or "").strip()
            if not (context_subject and q_text and a_text):
                continue

            s_qid = entity_manager.get_consistent_qid(context_subject, qid2name)
            o_qid = entity_manager.get_consistent_qid(a_text, qid2name)
            qid2name.setdefault(s_qid, context_subject)
            qid2name.setdefault(o_qid, a_text)

            triples.setdefault((s_qid, q_text), set()).add((o_qid, a_text))
            qid_relations.setdefault(s_qid, set()).add(q_text)

    return qid2name, triples, qid_relations


# ---------------------------------------------------------------------------
# Overlay construction (per case)
# ---------------------------------------------------------------------------

def build_case_overlay_from_edits(edits, base_qid2name,
                                  entity_manager: EntityQIDManager):
    """Build the per-case Overlay O_c from this case's requested_rewrite.

    Implements paper Sec. 3.2's conflict-arbitration rule: for each
    (subject, relation_id) the prior overlay value is cleared and the
    new ``target_new`` overwrites it. Edits never propagate to B or
    to other cases (paper's case-isolation assumption).
    """
    overlay_qid2name, overlay_triples, overlay_qid_relations = {}, {}, {}
    entity_manager.update_from_existing_kg(base_qid2name)
    entity_manager.update_from_existing_kg(overlay_qid2name)

    for edit in edits:
        subject = edit.get("subject", "").strip()
        relation_id = edit.get("relation_id", "unknown")
        target_new = (edit.get("target_new") or {}).get("str", "").strip()
        if not (subject and relation_id and target_new):
            continue

        s_qid = entity_manager.get_consistent_qid(subject, base_qid2name)
        o_qid = entity_manager.get_consistent_qid(target_new, base_qid2name)
        overlay_qid2name.setdefault(s_qid, subject)
        overlay_qid2name.setdefault(o_qid, target_new)

        # Conflict arbitration: clear prior value for the same (s, r) and
        # write the new edit on top.
        overlay_triples.setdefault((s_qid, relation_id), set()).clear()
        overlay_triples[(s_qid, relation_id)].add((o_qid, target_new))
        overlay_qid_relations.setdefault(s_qid, set()).add(relation_id)

    return overlay_qid2name, overlay_triples, overlay_qid_relations


# ---------------------------------------------------------------------------
# Edit impact surface (paper Eq. 1)
# ---------------------------------------------------------------------------

def compute_edit_influence_sets(edits):
    """Return (S^c_edit, P^c_edit) for the case's edit set E^c.

    S^c_edit collects lowercase forms of every edited subject and every
    new target value; P^c_edit collects every edited relation id.
    These are the sets referenced in paper Eq. 1's routing function L(q_i).
    """
    s_edit, p_edit = set(), set()
    for edit in edits:
        s = (edit.get("subject") or "").strip()
        t = ((edit.get("target_new") or {}).get("str") or "").strip()
        r = (edit.get("relation_id") or "unknown").strip()
        if s:
            s_edit.add(s.lower())
        if t:
            s_edit.add(t.lower())
        if r:
            p_edit.add(r)
    return s_edit, p_edit


def is_edit_sensitive_question(question_text, s_edit, p_edit):
    """Implement paper Eq. 1's routing function L(q_i).

    Returns True iff the sub-question's surface form contains an edited
    subject/target (S^c_edit) or an edited relation id (P^c_edit). The
    retrieval pipeline routes True queries to the case Overlay and
    False queries to the read-only Base layer.
    """
    ql = (question_text or "").lower()
    if any(name and name in ql for name in s_edit):
        return True
    if any(pid and pid.lower() in ql for pid in p_edit):
        return True
    return False
