from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty file: {path}")
    return pd.DataFrame(rows)


def label_bucket(label: str) -> str:
    if label in {"V", "T", "U"}:
        return label
    return "U"


def aggregate_token_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["token_label"] = df["token_label"].map(label_bucket)

    def layer_group(x: int) -> str:
        if x <= 9:
            return "early"
        if x <= 19:
            return "mid"
        return "late"

    df["layer_group"] = df["layer"].astype(int).map(layer_group)

    agg = (
        df.groupby(["sample_id", "token_index", "token_text", "response_text", "token_label"], as_index=False)
        .agg(
            ratio_gt_mean=("ratio_gt", "mean"),
            ratio_gt_median=("ratio_gt", "median"),
            margin_gt_mean=("margin_gt", "mean"),
            margin_vs_random_mean=("margin_vs_random", "mean"),
            gt_attn_mean=("gt_attn_mean", "mean"),
            bg_attn_mean=("bg_attn_mean", "mean"),
            random_gt_mean=("random_gt_mean", "mean"),
            cross_ratio_mean=("cross_ratio_orig_to_mod", "mean"),
            is_faithful=("is_faithful", "max"),
            type_f1=("type_f1", "max"),
            category_f1=("category_f1", "max"),
            num_recall=("num_recall", "max"),
        )
    )

    for group in ["early", "mid", "late"]:
        g = (
            df[df["layer_group"] == group]
            .groupby(["sample_id", "token_index"])["ratio_gt"]
            .mean()
            .rename(f"ratio_gt_{group}")
        )
        agg = agg.merge(g, on=["sample_id", "token_index"], how="left")
    return agg


def summarize_by_label(token_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, g in token_df.groupby("token_label"):
        out[label] = {
            "n_tokens": int(len(g)),
            "ratio_gt_mean": float(g["ratio_gt_mean"].mean()),
            "ratio_gt_median": float(g["ratio_gt_mean"].median()),
            "margin_gt_mean": float(g["margin_gt_mean"].mean()),
            "margin_vs_random_mean": float(g["margin_vs_random_mean"].mean()),
            "prop_ratio_gt_1": float((g["ratio_gt_mean"] > 1.0).mean()),
            "early_ratio_mean": float(g["ratio_gt_early"].mean()),
            "mid_ratio_mean": float(g["ratio_gt_mid"].mean()),
            "late_ratio_mean": float(g["ratio_gt_late"].mean()),
        }
    return out


def summarize_layer_by_label(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (layer, label), g in df.groupby(["layer", "token_label"]):
        rows.append({
            "layer": int(layer),
            "token_label": label,
            "mean_ratio_gt": float(g["ratio_gt"].mean()),
            "mean_margin_vs_random": float(g["margin_vs_random"].mean()),
            "count": int(len(g)),
        })
    return rows


def save_boxplot(token_df: pd.DataFrame, out_path: Path) -> None:
    labels = [x for x in ["V", "T", "U"] if x in set(token_df["token_label"]) ]
    data = [token_df[token_df["token_label"] == lb]["ratio_gt_mean"].values for lb in labels]
    plt.figure(figsize=(6, 4.5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.axhline(1.0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Token label")
    plt.ylabel("Mean GT/background ratio")
    plt.title("Visual vs Textual token attention ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_layer_curve(layer_rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(layer_rows)
    plt.figure(figsize=(8, 4.8))
    for label in ["V", "T", "U"]:
        g = df[df["token_label"] == label].sort_values("layer")
        if len(g) == 0:
            continue
        plt.plot(g["layer"], g["mean_ratio_gt"], marker="o", label=label)
    plt.axhline(1.0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Layer")
    plt.ylabel("Mean ratio_gt")
    plt.title("Per-layer ratio by token label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_sample_token_heatmap(df: pd.DataFrame, sample_id: str, out_path: Path) -> None:
    sample = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    if len(sample) == 0:
        return
    pivot = sample.pivot_table(index="layer", columns="token_index", values="ratio_gt", aggfunc="mean")
    plt.figure(figsize=(10, 5.5))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="ratio_gt")
    plt.xlabel("Token index")
    plt.ylabel("Layer")
    plt.title(f"Sample {sample_id}: token × layer ratio_gt")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual/textual token attention patterns")
    parser.add_argument("--token_labeled_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_id", type=str, default=None, help="Optional sample id for heatmap")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(Path(args.token_labeled_jsonl))
    df["token_label"] = df.get("token_label", "U")
    token_df = aggregate_token_level(df)
    label_summary = summarize_by_label(token_df)
    layer_summary = summarize_layer_by_label(df)

    payload = {
        "num_token_layer_rows": int(len(df)),
        "num_token_rows": int(len(token_df)),
        "label_summary": label_summary,
        "layer_summary": layer_summary,
    }
    (out_dir / "visual_textual_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    token_df.to_csv(out_dir / "token_level_aggregated.csv", index=False)

    save_boxplot(token_df, out_dir / "visual_vs_textual_boxplot.png")
    save_layer_curve(layer_summary, out_dir / "visual_vs_textual_layer_curve.png")
    if args.sample_id is not None:
        save_sample_token_heatmap(df, args.sample_id, out_dir / f"sample_{args.sample_id}_token_layer_heatmap.png")

    print(f"[ok] saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
