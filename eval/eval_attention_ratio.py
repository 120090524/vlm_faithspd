from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np



def load_analysis_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            labels[str(row["sample_id"])] = int(row["is_faithful"])
    return labels



def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(int(t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    tn = sum(int(t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    fp = sum(int(t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    bal_acc = (recall + specificity) / 2.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "f1": f1,
    }



def compute_auc(scores: list[float], labels: list[int]) -> float:
    # Mann–Whitney U interpretation of AUROC
    pos = [(s, l) for s, l in zip(scores, labels) if l == 1]
    neg = [(s, l) for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks = {}
    for i, (score, _) in enumerate(sorted_pairs, start=1):
        ranks.setdefault(score, []).append(i)
    avg_rank = {score: float(np.mean(rs)) for score, rs in ranks.items()}
    rank_sum_pos = sum(avg_rank[s] for s, l in zip(scores, labels) if l == 1)
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate attention-ratio against faithfulness labels")
    parser.add_argument("--ratio_file", type=str, required=True)
    parser.add_argument("--analysis_jsonl", type=str, required=True)
    parser.add_argument("--ratio_key", type=str, default="ratio_sample_mean")
    parser.add_argument("--out_file", type=str, default="outputs/attention_ratio/attention_ratio_eval.json")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.8, 1.0, 1.2, 1.5, 2.0])
    args = parser.parse_args()

    ratio_payload = json.loads(Path(args.ratio_file).read_text(encoding="utf-8"))
    labels = load_analysis_labels(Path(args.analysis_jsonl))

    scores: list[float] = []
    y_true: list[int] = []
    per_sample: list[dict[str, Any]] = []

    for row in ratio_payload["results"]:
        sample_id = str(row["sample_id"])
        if sample_id not in labels:
            continue
        if "summary" in row:
            score = float(row["summary"][args.ratio_key])
        else:
            score = float(row[args.ratio_key])
        if not np.isfinite(score):
            continue
        y = int(labels[sample_id])
        scores.append(score)
        y_true.append(y)
        per_sample.append({"sample_id": sample_id, "score": score, "label": y})

    faithful_scores = [s for s, y in zip(scores, y_true) if y == 1]
    unfaithful_scores = [s for s, y in zip(scores, y_true) if y == 0]

    threshold_results: dict[str, Any] = {}
    best = None
    for thr in args.thresholds:
        y_pred = [int(s > thr) for s in scores]
        metrics = compute_binary_metrics(y_true, y_pred)
        threshold_results[str(thr)] = metrics
        if best is None or metrics["f1"] > best[1]["f1"]:
            best = (thr, metrics)

    payload = {
        "num_samples": len(scores),
        "ratio_key": args.ratio_key,
        "auc": compute_auc(scores, y_true),
        "faithful_mean": float(np.mean(faithful_scores)) if faithful_scores else float("nan"),
        "unfaithful_mean": float(np.mean(unfaithful_scores)) if unfaithful_scores else float("nan"),
        "faithful_median": float(np.median(faithful_scores)) if faithful_scores else float("nan"),
        "unfaithful_median": float(np.median(unfaithful_scores)) if unfaithful_scores else float("nan"),
        "threshold_results": threshold_results,
        "best_threshold": None if best is None else {"threshold": best[0], **best[1]},
        "per_sample": per_sample,
    }

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
