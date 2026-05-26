#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity & Relation Detector Standalone Evaluation
=================================================
Evaluates the trained DistilBERT entity detector (g_phi) and relation
detector (g_psi) adopted by CAPE-KG on a held-out 80/20 split of their
respective training corpora (derived from MQuAKE-CF).

Reports Accuracy / Precision / Recall / F1 / confusion matrix per detector.

Usage:
    python evaluate_detectors.py
    python evaluate_detectors.py --detector entity
    python evaluate_detectors.py --detector relation
    python evaluate_detectors.py --batch_size 128 --seed 42
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

SCRIPT_DIR = Path(__file__).resolve().parent
# Script lives at: <repo>/emnlp/CAPEKG_code/entity_extractor_evaluation/
# Data & models live at: <repo>/emnlp/EMNLP2026/
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[1] / "EMNLP2026"

LOG_DIR = SCRIPT_DIR / "log"
RESULT_DIR = SCRIPT_DIR / "result"
LOG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

DETECTOR_CONFIGS = {
    "entity": {
        "data_file": "train/datasets_entity_judge.json",
        "model_dir": "train/results_entity_judge/best_model_entity_judge",
        "pair_key": "entity",
    },
    "relation": {
        "data_file": "train/datasets_relation_judge.json",
        "model_dir": "train/results/results/best_model",
        "pair_key": "relation",
    },
}


def setup_logger(timestamp: str) -> Path:
    log_path = LOG_DIR / f"eval_{timestamp}.log"
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def evaluate_detector(name: str, cfg: dict, repo_root: Path, seed: int,
                      batch_size: int, max_length: int,
                      device: torch.device) -> dict:
    log = logging.getLogger()
    log.info("=" * 70)
    log.info("Evaluating %s detector", name)
    log.info("=" * 70)

    data_path = repo_root / cfg["data_file"]
    model_path = repo_root / cfg["model_dir"]
    log.info("data : %s", data_path)
    log.info("model: %s", model_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_path}")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d samples", len(data))

    # Stratified 80/20 split with fixed seed for reproducibility
    labels = [d["label"] for d in data]
    _, test_data = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=labels
    )
    n = len(test_data)
    pos = sum(1 for d in test_data if d["label"] == 1)
    log.info("Split (seed=%d): %d train / %d test", seed, len(data) - n, n)
    log.info("Test set label dist: %d pos / %d neg", pos, n - pos)

    log.info("Loading model & tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(
        str(model_path), num_labels=2
    ).to(device)
    model.eval()

    pair_key = cfg["pair_key"]
    questions = [d["question"] for d in test_data]
    pairs = [d[pair_key] for d in test_data]
    true_labels = np.array([d["label"] for d in test_data])

    preds = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc=f"{name} eval", file=sys.stdout):
            batch_q = questions[i:i + batch_size]
            batch_p = pairs[i:i + batch_size]
            inputs = tokenizer(
                batch_q, batch_p,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)
            logits = model(**inputs).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    elapsed = time.time() - t0
    log.info("Inference: %.2fs total (%.1f samples/s)", elapsed, n / elapsed)

    preds_arr = np.array(preds)
    metrics = {
        "detector": name,
        "n_test": int(n),
        "n_pos": int(true_labels.sum()),
        "n_neg": int((1 - true_labels).sum()),
        "accuracy": float(accuracy_score(true_labels, preds_arr)),
        "precision": float(precision_score(true_labels, preds_arr, zero_division=0)),
        "recall": float(recall_score(true_labels, preds_arr, zero_division=0)),
        "f1": float(f1_score(true_labels, preds_arr, zero_division=0)),
        "confusion_matrix": confusion_matrix(true_labels, preds_arr).tolist(),
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "batch_size": batch_size,
        "max_length": max_length,
    }
    log.info("Accuracy : %.4f", metrics["accuracy"])
    log.info("Precision: %.4f", metrics["precision"])
    log.info("Recall   : %.4f", metrics["recall"])
    log.info("F1       : %.4f", metrics["f1"])
    log.info("Confusion [[TN, FP], [FN, TP]]: %s", metrics["confusion_matrix"])
    return metrics


def write_markdown(results: list, output_path: Path, seed: int) -> None:
    lines = [
        "# Entity & Relation Detector Evaluation",
        "",
        "- **Split**: 80/20 on MQuAKE-CF (stratified by label)",
        f"- **Seed**: {seed}",
        "- **Model**: DistilBERT fine-tuned, weights from KEDKG (Lu et al., 2025)",
        "",
        "| Detector | N (test) | Pos | Neg | Accuracy | Precision | Recall | F1 |",
        "|----------|---------:|----:|----:|---------:|----------:|-------:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['detector']} | {r['n_test']:,} | {r['n_pos']:,} | {r['n_neg']:,} | "
            f"{r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | "
            f"{r['recall']*100:.2f}% | {r['f1']*100:.2f}% |"
        )
    lines += ["", "## Confusion matrices  `[[TN, FP], [FN, TP]]`", ""]
    for r in results:
        lines.append(f"- **{r['detector']}**: `{r['confusion_matrix']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Detector standalone evaluation")
    parser.add_argument("--detector", choices=["entity", "relation", "both"],
                        default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--repo_root", type=str, default=str(DEFAULT_REPO_ROOT),
                        help="Path to EMNLP2026/ (containing train/...)")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logger(timestamp)
    log = logging.getLogger()
    log.info("Log file: %s", log_path)
    log.info("Args: %s", vars(args))

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    repo_root = Path(args.repo_root).resolve()
    log.info("Repo root: %s", repo_root)

    targets = ["entity", "relation"] if args.detector == "both" else [args.detector]
    results = []
    for name in targets:
        results.append(
            evaluate_detector(name, DETECTOR_CONFIGS[name], repo_root,
                              args.seed, args.batch_size, args.max_length,
                              device)
        )

    json_path = RESULT_DIR / f"results_{timestamp}.json"
    md_path = RESULT_DIR / f"results_{timestamp}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "results": results}, f, indent=2)
    write_markdown(results, md_path, args.seed)

    log.info("=" * 70)
    log.info("Wrote: %s", json_path)
    log.info("Wrote: %s", md_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
