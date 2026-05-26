"""
REBEL Triple Extractor.

Wraps the pretrained REBEL seq2seq relation-extraction model (Cabot &
Navigli, 2021) to convert a natural-language sentence into a list of
(head, relation, tail) triples. Used in two places in CAPE-KG:

  (1) KG construction: extract triples from edit-prompt sentences
      to populate the overlay KG (paper Sec. 3.1).
  (2) Retrieval: extract candidate entities from sub-questions so that
      the entity detector g_phi can score them (paper Sec. 3.3).
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def extract_relations_from_model_output(text):
    """Parse REBEL's special-token-augmented output into triples.

    REBEL emits sequences of the form
        <triplet> <head> <subj> <tail> <obj> <relation>
    one triple at a time. This routine walks the token stream and
    accumulates (head, type, tail) dicts.
    """
    relations = []
    head, rel, tail = "", "", ""
    current = "x"
    text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").strip()

    for token in text.split():
        if token == "<triplet>":
            current = "t"
            if rel:
                relations.append({"head": head.strip(),
                                  "type": rel.strip(),
                                  "tail": tail.strip()})
                rel = ""
            head = ""
        elif token == "<subj>":
            current = "s"
            if rel:
                relations.append({"head": head.strip(),
                                  "type": rel.strip(),
                                  "tail": tail.strip()})
            tail = ""
        elif token == "<obj>":
            current = "o"
            rel = ""
        else:
            if current == "t":
                head += " " + token
            elif current == "s":
                tail += " " + token
            elif current == "o":
                rel += " " + token

    if head and rel and tail:
        relations.append({"head": head.strip(),
                          "type": rel.strip(),
                          "tail": tail.strip()})
    return relations


class KB:
    """Lightweight deduplicating triple store used as REBEL's output buffer."""

    def __init__(self):
        self.relations = []

    @staticmethod
    def _eq(r1, r2):
        return all(r1[k] == r2[k] for k in ("head", "type", "tail"))

    def add_relation(self, r):
        if not any(self._eq(r, existing) for existing in self.relations):
            self.relations.append(r)


class RebelExtractor:
    """Pretrained REBEL wrapper that returns a deduplicated triple set."""

    def __init__(self, model_path, device="cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.device = device

    def extract(self, text, num_beams=3, num_return_sequences=3):
        """Run beam search and return all unique triples across beams."""
        inputs = self.tokenizer(text, max_length=512, padding=True,
                                truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=216,
            length_penalty=0,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
        kb = KB()
        for seq in decoded:
            for rel in extract_relations_from_model_output(seq):
                kb.add_relation(rel)
        return kb
