#!/usr/bin/env python3
"""
Build consolidated prediction summary for outputs/no_disease_label/*.csv
and print simple accuracy stats.
"""

import csv
from pathlib import Path
import sys

# Locate DermAgent/eval for ontology imports (walk up parents)
eval_path = None
for cand in [Path.cwd(), *Path.cwd().parents]:
    maybe = cand / "DermAgent" / "eval" / "ontology_utils.py"
    if maybe.exists():
        eval_path = maybe.parent
        break

if eval_path is None:
    raise FileNotFoundError("DermAgent/eval/ontology_utils.py not found.")

sys.path.insert(0, str(eval_path))
from ontology_utils import OntologyTree  # noqa: E402

ROOT = Path("baseline/outputs/no_disease_label")
ONTO_PATH = Path("/home/work/wonjun/DermAgent/dataset/Derm1M/ontology.json")
OUT_PATH = ROOT / "all_predictions_summary.csv"

if not ONTO_PATH.exists():
    raise FileNotFoundError(f"Ontology not found: {ONTO_PATH}")

tree = OntologyTree(str(ONTO_PATH))

rows = []
csv_files = sorted(ROOT.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"No CSV files found in {ROOT}")

def read_with_encoding(path: Path):
    # Some files are ISO-8859 encoded
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                yield from reader
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Failed to decode", str(path), 0, 0, "encoding not supported")

for csv_path in csv_files:
    for row in read_with_encoding(csv_path):
        fname = row.get("filename", "").strip()
        gt_raw = (row.get("ground_truth_disease_label") or "").strip().rstrip(" .,:;")
        pred_raw = (row.get("predicted_disease_label") or "").strip().rstrip(" .,:;")
        gt_can = tree.get_canonical_name(gt_raw) if gt_raw else ""
        pred_can = tree.get_canonical_name(pred_raw) if pred_raw else ""
        rows.append({
            "source_file": csv_path.name,
            "filename": fname,
            "ground_truth_raw": gt_raw,
            "predicted_raw": pred_raw,
            "ground_truth_canonical": gt_can or "",
            "predicted_canonical": pred_can or "",
            "valid_gt": 1 if gt_can else 0,
            "exact_match": 1 if gt_can and pred_can and gt_can == pred_can else 0,
        })

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "source_file",
        "filename",
        "ground_truth_raw",
        "predicted_raw",
        "ground_truth_canonical",
        "predicted_canonical",
        "valid_gt",
        "exact_match",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

total = len(rows)
valid = sum(r["valid_gt"] for r in rows)
correct = sum(r["exact_match"] for r in rows)

print(f"Wrote summary: {OUT_PATH}")
print(f"Total rows: {total}, valid_gt: {valid}, exact_match: {correct}, exact_match_rate(valid-only): {correct / valid if valid else 0:.3f}")
