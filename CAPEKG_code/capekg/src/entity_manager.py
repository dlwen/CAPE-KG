"""
Entity QID Consistency Manager.

Maintains a consistent mapping between surface entity names and QIDs across
KG construction, update, and retrieval. This consistency is required by the
multi-layer KG architecture (paper Sec. 3.1): the same surface form must
resolve to the same QID whether it is queried in the Base layer or the
per-case Overlay layer; otherwise the routing rule in Eq. 1 cannot match
the canonical subject of an edit.
"""

import hashlib


class EntityQIDManager:
    """Bidirectional cache mapping normalized entity names to consistent QIDs.

    QIDs are either (a) found in an existing KG via name matching, or
    (b) synthesized from a deterministic MD5 hash of the normalized name.
    The same surface form therefore always resolves to the same QID,
    independent of which case (Base or Overlay) it appears in.
    """

    def __init__(self):
        self.entity_to_qid_cache = {}
        self.qid_to_entity_cache = {}
        self.qid_counter = 1_000_000

    @staticmethod
    def normalize_entity(name):
        return name.lower().strip() if name else ""

    def update_from_existing_kg(self, qid2name):
        """Seed the cache from an existing qid2name mapping."""
        for qid, name in qid2name.items():
            normalized = self.normalize_entity(name)
            if normalized not in self.entity_to_qid_cache:
                self.entity_to_qid_cache[normalized] = qid
                self.qid_to_entity_cache[qid] = name

    def find_matching_qid(self, entity_name, qid2name):
        """Search an existing KG for a matching QID for ``entity_name``.

        Uses exact match first, then a heuristic containment match that
        prefers geographic entities (cities / countries) over non-geographic
        entities of the same surface form. Returns ``None`` if no good match.
        """
        normalized = self.normalize_entity(entity_name)

        # Exact match
        for qid, name in qid2name.items():
            if normalized == self.normalize_entity(name):
                return qid

        # Heuristic containment match with priority weighting
        candidates = []
        for qid, name in qid2name.items():
            n_name = self.normalize_entity(name)
            if (normalized in n_name and len(normalized) > 3) or \
               (n_name in normalized and len(n_name) > 3):
                priority = 0
                if len(n_name) < len(normalized) * 2:
                    priority += 5
                # Down-weight obviously non-entity surface forms
                if any(k in n_name for k in ["movie", "film", "album", "song", "book"]):
                    priority -= 20
                candidates.append((priority, qid))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_priority, best_qid = candidates[0]
            if best_priority > 0:
                return best_qid
        return None

    def get_consistent_qid(self, entity_name, qid2name=None):
        """Return a stable QID for ``entity_name``.

        Resolution order:
            1. Local cache hit.
            2. Match against an externally provided ``qid2name`` mapping.
            3. Synthesize a deterministic pseudo-QID from MD5(normalized name).
        """
        if not entity_name or not entity_name.strip():
            return "Q000000"

        normalized = self.normalize_entity(entity_name)
        if normalized in self.entity_to_qid_cache:
            return self.entity_to_qid_cache[normalized]

        if qid2name:
            matching = self.find_matching_qid(entity_name, qid2name)
            if matching:
                self.entity_to_qid_cache[normalized] = matching
                self.qid_to_entity_cache[matching] = entity_name
                return matching

        # Deterministic synthesized QID
        digest = hashlib.md5(normalized.encode()).hexdigest()[:8]
        qid = f"Q{int(digest, 16) % 1_000_000}"
        while qid in self.qid_to_entity_cache:
            self.qid_counter += 1
            qid = f"Q{self.qid_counter}"

        self.entity_to_qid_cache[normalized] = qid
        self.qid_to_entity_cache[qid] = entity_name
        return qid

    def generate_new_qid(self, entity_name):
        """Force-generate a fresh QID. Used to resolve subject/object collisions
        within a single edit triple."""
        normalized = self.normalize_entity(entity_name)
        self.qid_counter += 1
        qid = f"Q{self.qid_counter}"
        while qid in self.qid_to_entity_cache:
            self.qid_counter += 1
            qid = f"Q{self.qid_counter}"
        self.entity_to_qid_cache[normalized] = qid
        self.qid_to_entity_cache[qid] = entity_name
        return qid
