# Entity & Relation Detector Standalone Evaluation

Standalone evaluation of the entity detector (`g_ϕ`) and relation detector
(`g_ψ`) adopted by CAPE-KG from KEDKG (Lu et al., 2025).

## What this evaluates

CAPE-KG uses the DistilBERT-based entity & relation detectors introduced by
KEDKG. This script reports their classification performance on a held-out
80/20 split of the detector training corpora (derived from MQuAKE-CF),
stratified by label with seed 42.

Metrics: Accuracy / Precision / Recall / F1 / confusion matrix.

## Files

- `evaluate_detectors.py` — main evaluation script
- `log/` — per-run logs (auto-created)
- `result/` — per-run JSON + Markdown summaries (auto-created)

## Required external files

The script reads data and weights from `EMNLP2026/`:

| Path | Description |
|------|-------------|
| `EMNLP2026/train/datasets_entity_judge.json` | Entity-judge training corpus |
| `EMNLP2026/train/datasets_relation_judge.json` | Relation-judge training corpus |
| `EMNLP2026/train/results_entity_judge/best_model_entity_judge/` | Trained entity detector (DistilBERT) |
| `EMNLP2026/train/results/results/best_model/` | Trained relation detector (DistilBERT) |

If the detector weights are missing, download them from the KEDKG
[Google Drive](https://drive.google.com/drive/folders/14xr7ruFZdmqCJ6_thbgirmTIVeP1QWHk?usp=sharing)
and place them at the paths above. Each model directory should contain
`config.json`, `model.safetensors` (or `pytorch_model.bin`),
`tokenizer_config.json`, `vocab.txt`, and `special_tokens_map.json`.

## Requirements

- Python ≥ 3.9
- `torch`, `transformers`, `scikit-learn`, `tqdm`, `numpy`

Tested with `transformers==4.52.4`, `torch==2.7.1`, `scikit-learn==1.6.1`.

## Usage

```bash
# Evaluate both detectors (default)
python evaluate_detectors.py

# Evaluate a single detector
python evaluate_detectors.py --detector entity
python evaluate_detectors.py --detector relation

# Tune runtime knobs
python evaluate_detectors.py --seed 42 --batch_size 128 --max_length 128

# Custom repo path (if you moved EMNLP2026/ elsewhere)
python evaluate_detectors.py --repo_root /path/to/EMNLP2026
```

## Outputs

Each run writes three files (sharing a timestamp):

- `log/eval_<timestamp>.log` — full console log
- `result/results_<timestamp>.json` — machine-readable metrics
- `result/results_<timestamp>.md` — human-readable summary table
