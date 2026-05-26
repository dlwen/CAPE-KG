"""
CAPE-KG: Consistency-Aware Parameter-Preserving Knowledge Editing
with Knowledge Graphs for Multi-Hop Question Answering.

Modules:
    entity_manager  - QID consistency manager for entity linking.
    rebel_extractor - REBEL-based triple extraction.
    detectors       - DistilBERT entity / relation detectors (paper Eq. 1 g_phi).
    knn_selector    - KNN-based dynamic few-shot example selection.
    llm_interface   - GPT / LLaMA inference calls.
    kg              - Multi-layer KG construction (paper Sec. 3.1 + 3.2):
                      base layer, per-case overlay, edit-influence sets, routing.
    retrieval       - Edit-aware progressive retrieval (paper Sec. 3.3):
                      high / low / failure confidence stages.
    utils           - Small helpers (gold path verification, formatting).
    main            - End-to-end pipeline entry point.
"""
