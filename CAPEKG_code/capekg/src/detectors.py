"""
Entity / Relation Detector Wrappers.

Two DistilBERT binary classifiers adopted from KEDKG (Lu et al., 2025):

  - Entity detector  g_phi(q, k) -> [0, 1]
        Probability that candidate entity k is the subject of sub-question q.
        Used in paper Eq. 1's edit-aware retrieval routing and in the
        High Confidence stage of progressive retrieval (Sec. 3.3).

  - Relation detector g_psi(q, r) -> [0, 1]
        Probability that candidate relation r is the predicate referenced
        in sub-question q. Used after entity scoring to select the edge
        followed in the KG.

Standalone classification accuracy on the held-out 80/20 split of the
detector training corpora is reported in the appendix.
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


ENTITY = 0  # tag passed by the retrieval pipeline to select the entity head
REL = 1     # tag passed by the retrieval pipeline to select the relation head


class Detector:
    """Single binary text-pair classifier (entity or relation)."""

    def __init__(self, model_path, device="cuda:0"):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, question, candidate):
        inputs = self.tokenizer(
            question, candidate,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return logits.argmax().item(), probs


class DetectorScorer:
    """Wraps both detectors and exposes paper Eq. 1's score function.

    ``get_scores(q, candidates, flag)`` returns the list of
    (probability, candidate) pairs with probability >= 0.5
    (the threshold tau used in the paper's main experiments).
    """

    def __init__(self, entity_detector: Detector, relation_detector: Detector,
                 threshold: float = 0.5):
        self.entity = entity_detector
        self.relation = relation_detector
        self.threshold = threshold

    def get_scores(self, question, candidates, flag):
        det = self.entity if flag == ENTITY else self.relation
        scored = []
        for cand in candidates:
            _, probs = det.predict(question, cand)
            score = probs[0, 1].item()
            if score >= self.threshold:
                scored.append((score, cand))
        return scored
