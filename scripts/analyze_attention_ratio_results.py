from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float('nan')


def summarize_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results", [])
    if not results:
        return {"num_samples": 0}

    scalar_rows: List[Dict[str, Any]] = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        scalar_rows.append(row)
    df = pd.DataFrame(scalar_rows)

    if "num_changed_tokens_original" in df.columns and "num_other_tokens_original" in df.columns:
        denom = df["num_changed_tokens_original"] + df["num_other_tokens_original"]
        df["changed_frac"] = df["num_changed_tokens_original"] / denom.replace(0, np.nan)

    diff = df["ratio_sample_mean"] - df["ratio_random_mean"]
    try:
        paired_t = stats.ttest_rel(df["ratio_sample_mean"], df["ratio_random_mean"], nan_policy="omit")
        t_stat = safe_float(paired_t.statistic)
        t_p = safe_float(paired_t.pvalue)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")

    try:
        wilcoxon = stats.wilcoxon(diff, alternative="greater")
        w_stat = safe_float(wilcoxon.statistic)
        w_p = safe_float(wilcoxon.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    try:
        k = int(df["gt_beats_random"].sum())
        n = int(len(df))
        sign_test = stats.binomtest(k, n=n, p=0.5, alternative="greater")
        sign_p = safe_float(sign_test.pvalue)
    except Exception:
        sign_p = float("nan")

    effect_size = safe_float(diff.mean() / diff.std(ddof=1)) if len(df) > 1 and diff.std(ddof=1) > 0 else float("nan")

    summary: Dict[str, Any] = {
        "model": payload.get("model"),
        "manifest_file": payload.get("manifest_file"),
        "split": payload.get("split"),
        "question_mode": payload.get("question_mode"),
        "num_rows": payload.get("num_rows"),
        "num_samples": int(len(df)),
        "num_skipped": payload.get("num_skipped"),
        "overall": {
            "ratio_sample_mean_mean": safe_float(df["ratio_sample_mean"].mean()),
            "ratio_sample_mean_median": safe_float(df["ratio_sample_mean"].median()),
            "ratio_random_mean_mean": safe_float(df["ratio_random_mean"].mean()),
            "ratio_random_mean_median": safe_float(df["ratio_random_mean"].median()),
            "margin_mean": safe_float(df["margin_vs_random"].mean()),
            "margin_median": safe_float(df["margin_vs_random"].median()),
            "gt_beats_random_rate": safe_float(df["gt_beats_random"].mean()),
            "ratio_sample_gt1_rate": safe_float((df["ratio_sample_mean"] > 1).mean()),
            "ratio_random_gt1_rate": safe_float((df["ratio_random_mean"] > 1).mean()),
            "cross_image_mean": safe_float(df["ratio_cross_image"].mean()),
            "cross_image_median": safe_float(df["ratio_cross_image"].median()),
            "cross_image_gt1_rate": safe_float((df["ratio_cross_image"] > 1).mean()),
            "paired_t_stat": t_stat,
            "paired_t_pvalue": t_p,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_p,
            "binom_sign_pvalue": sign_p,
            "cohen_d_paired": effect_size,
        },
    }

    if "source_split" in df.columns:
        split_stats = {}
        for split, g in df.groupby("source_split"):
            split_stats[str(split)] = {
                "n": int(len(g)),
                "ratio_sample_mean_mean": safe_float(g["ratio_sample_mean"].mean()),
                "ratio_random_mean_mean": safe_float(g["ratio_random_mean"].mean()),
                "margin_mean": safe_float(g["margin_vs_random"].mean()),
                "gt_beats_random_rate": safe_float(g["gt_beats_random"].mean()),
            }
        summary["by_source_split"] = split_stats

    # Filtered subsets that often remove degenerate masks.
    if "changed_frac" in df.columns:
        subsets = {
            "changed_frac_between_0p05_and_0p6": df[(df["changed_frac"] >= 0.05) & (df["changed_frac"] <= 0.6)],
            "changed_tokens_ge_20_and_changed_frac_le_0p4": df[(df["num_changed_tokens_original"] >= 20) & (df["changed_frac"] <= 0.4)],
        }
        summary["filtered"] = {}
        for name, g in subsets.items():
            if len(g) == 0:
                continue
            summary["filtered"][name] = {
                "n": int(len(g)),
                "ratio_sample_mean_mean": safe_float(g["ratio_sample_mean"].mean()),
                "ratio_random_mean_mean": safe_float(g["ratio_random_mean"].mean()),
                "margin_mean": safe_float(g["margin_vs_random"].mean()),
                "gt_beats_random_rate": safe_float(g["gt_beats_random"].mean()),
                "cross_image_mean": safe_float(g["ratio_cross_image"].mean()),
            }

    # Aggregate layer-level means across samples.
    first_layers = results[0].get("layers", [])
    if first_layers:
        layer_rows = []
        for layer in first_layers:
            per_sample_means = []
            for r in results:
                vals = r.get("per_layer_step_ratio", {}).get(str(layer), [])
                if vals:
                    per_sample_means.append(float(np.mean(vals)))
            if per_sample_means:
                layer_rows.append({
                    "layer": int(layer),
                    "mean_ratio": safe_float(np.mean(per_sample_means)),
                    "median_ratio": safe_float(np.median(per_sample_means)),
                    "prop_gt1": safe_float(np.mean(np.array(per_sample_means) > 1.0)),
                })
        summary["layer_summary"] = layer_rows

        # Early / middle / late layer groups.
        groups = {
            "early": list(range(0, 10)),
            "mid": list(range(10, 20)),
            "late": list(range(20, 28)),
        }
        summary["layer_groups"] = {}
        for name, ls in groups.items():
            vals = []
            for r in results:
                for l in ls:
                    vals.extend(r.get("per_layer_step_ratio", {}).get(str(l), []))
            if vals:
                summary["layer_groups"][name] = {
                    "mean_ratio": safe_float(np.mean(vals)),
                    "median_ratio": safe_float(np.median(vals)),
                }

    # Step quartiles after pooling layers within each sample.
    quartiles = {0: [], 1: [], 2: [], 3: []}
    for r in results:
        layers = r.get("layers", [])
        if not layers:
            continue
        mat = []
        ok = True
        for layer in layers:
            vals = r.get("per_layer_step_ratio", {}).get(str(layer), [])
            if not vals:
                ok = False
                break
            mat.append(vals)
        if not ok:
            continue
        arr = np.array(mat, dtype=float)
        step_means = arr.mean(axis=0)
        n = len(step_means)
        for i, val in enumerate(step_means):
            q = min(3, int(i * 4 / max(n, 1)))
            quartiles[q].append(float(val))
    summary["step_quartiles"] = {
        str(q): {"mean_ratio": safe_float(np.mean(v)), "count": int(len(v))}
        for q, v in quartiles.items() if v
    }

    # Good/bad cases.
    sort_cols = ["sample_id", "ratio_sample_mean", "ratio_random_mean", "margin_vs_random", "ratio_cross_image",
                 "gt_beats_random", "num_changed_tokens_original", "num_other_tokens_original"]
    summary["lowest_margin_samples"] = df.sort_values("margin_vs_random").head(10)[sort_cols].to_dict(orient="records")
    summary["highest_margin_samples"] = df.sort_values("margin_vs_random", ascending=False).head(10)[sort_cols].to_dict(orient="records")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the attention-ratio result JSON")
    parser.add_argument("--out_dir", default=None, help="Directory for summary outputs; default is beside input")
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summarize_results(payload)

    out_dir = Path(args.out_dir) if args.out_dir else input_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
