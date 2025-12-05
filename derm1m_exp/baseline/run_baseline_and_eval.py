#!/usr/bin/env python3
"""
Run the baseline Qwen3-VL-8B model on a CSV and evaluate predictions
using the hierarchical metrics.
"""

import json
import os
from pathlib import Path
from typing import List

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = Path("/home/work/wonjun/DermAgent/dataset/Derm1M")

# Inputs
INPUT_CSV = DATA_ROOT / "Derm1M_v2_pretrain_ontology_sampled_100.csv"
IMAGE_BASE = DATA_ROOT
MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

# Outputs
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = OUTPUT_DIR / "qwen3vl8b_baseline_predictions.csv"
METRICS_JSON = OUTPUT_DIR / "qwen3vl8b_baseline_metrics.json"

# Ensure project modules on path
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DermAgent" / "eval"))

from baseline import process_csv  # noqa: E402
from model import QwenVL  # noqa: E402
from DermAgent.eval.evaluation_metrics import HierarchicalEvaluator  # noqa: E402


def evaluate_predictions(csv_path: Path) -> dict:
    """Load prediction CSV and compute hierarchical metrics."""
    evaluator = HierarchicalEvaluator()
    gt_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []

    import csv
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row.get("ground_truth_disease_label", "").strip()
            pred = row.get("predicted_disease_label", "").strip()
            if gt:
                gt_labels.append([gt])
                pred_labels.append([pred] if pred else [])

    result = evaluator.evaluate_batch(gt_labels, pred_labels)
    return {
        "exact_match": result.exact_match,
        "partial_match": result.partial_match,
        "hierarchical_precision": result.hierarchical_precision,
        "hierarchical_recall": result.hierarchical_recall,
        "hierarchical_f1": result.hierarchical_f1,
        "avg_hierarchical_distance": result.avg_hierarchical_distance,
        "total_samples": result.total_samples,
        "valid_samples": result.valid_samples,
        "skipped_samples": result.skipped_samples,
        "level_accuracy": result.level_accuracy,
    }


def main():
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Image root: {IMAGE_BASE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output CSV: {PRED_CSV}")

    # Run baseline inference
    agent = QwenVL(model_path=MODEL_PATH)
    process_csv(
        input_csv=str(INPUT_CSV),
        output_csv=str(PRED_CSV),
        image_base_folder=str(IMAGE_BASE),
        model=agent,
    )

    # Evaluate predictions
    metrics = evaluate_predictions(PRED_CSV)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"\nMetrics saved to: {METRICS_JSON}")


if __name__ == "__main__":
    main()
