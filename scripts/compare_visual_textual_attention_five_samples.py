from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OPTIONAL_SCORE_COLS = ["is_faithful", "type_f1", "category_f1", "num_recall"]


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


def parse_sample_ids(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def label_bucket(label: str | None) -> str:
    if label in {"V", "T", "U"}:
        return str(label)
    return "U"


def aggregate_token_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["token_label"] = df.get("token_label", "U").map(label_bucket)
    df["sample_id"] = df["sample_id"].astype(str)
    df["token_index"] = df["token_index"].astype(int)
    df["layer"] = df["layer"].astype(int)

    def layer_group(x: int) -> str:
        if x <= 9:
            return "early"
        if x <= 19:
            return "mid"
        return "late"

    df["layer_group"] = df["layer"].map(layer_group)

    agg_map: dict[str, tuple[str, str]] = {
        "ratio_gt_mean": ("ratio_gt", "mean"),
        "ratio_gt_median": ("ratio_gt", "median"),
        "margin_gt_mean": ("margin_gt", "mean"),
        "margin_vs_random_mean": ("margin_vs_random", "mean"),
        "gt_attn_mean": ("gt_attn_mean", "mean"),
        "bg_attn_mean": ("bg_attn_mean", "mean"),
        "random_gt_mean": ("random_gt_mean", "mean"),
        "cross_ratio_mean": ("cross_ratio_orig_to_mod", "mean"),
    }
    for col in OPTIONAL_SCORE_COLS:
        if col in df.columns:
            agg_map[col] = (col, "max")

    token_df = (
        df.groupby(["sample_id", "token_index", "token_text", "response_text", "token_label"], as_index=False)
        .agg(**agg_map)
    )

    for group in ["early", "mid", "late"]:
        g = (
            df[df["layer_group"] == group]
            .groupby(["sample_id", "token_index"])["ratio_gt"]
            .mean()
            .rename(f"ratio_gt_{group}")
        )
        token_df = token_df.merge(g, on=["sample_id", "token_index"], how="left")

    token_df["ratio_gt_gt1"] = token_df["ratio_gt_mean"] > 1.0
    token_df["mvr_gt0"] = token_df["margin_vs_random_mean"] > 0.0
    return token_df


def summarize_by_label(token_df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, g in token_df.groupby("token_label"):
        out[label] = {
            "n_tokens": int(len(g)),
            "ratio_gt_mean": float(g["ratio_gt_mean"].mean()),
            "ratio_gt_median": float(g["ratio_gt_mean"].median()),
            "margin_gt_mean": float(g["margin_gt_mean"].mean()),
            "margin_vs_random_mean": float(g["margin_vs_random_mean"].mean()),
            "prop_ratio_gt_1": float(g["ratio_gt_gt1"].mean()),
            "prop_margin_vs_random_gt_0": float(g["mvr_gt0"].mean()),
            "early_ratio_mean": float(g["ratio_gt_early"].mean()),
            "mid_ratio_mean": float(g["ratio_gt_mid"].mean()),
            "late_ratio_mean": float(g["ratio_gt_late"].mean()),
        }
    return out


def summarize_layer_by_label(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (layer, label), g in df.groupby(["layer", "token_label"]):
        rows.append(
            {
                "layer": int(layer),
                "token_label": label,
                "mean_ratio_gt": float(g["ratio_gt"].mean()),
                "mean_margin_vs_random": float(g["margin_vs_random"].mean()),
                "count": int(len(g)),
            }
        )
    return rows


def summarize_sample_label(token_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (sample_id, label), g in token_df.groupby(["sample_id", "token_label"]):
        rows.append(
            {
                "sample_id": str(sample_id),
                "token_label": label,
                "n_tokens": int(len(g)),
                "ratio_gt_mean": float(g["ratio_gt_mean"].mean()),
                "ratio_gt_median": float(g["ratio_gt_mean"].median()),
                "margin_vs_random_mean": float(g["margin_vs_random_mean"].mean()),
                "prop_ratio_gt_1": float(g["ratio_gt_gt1"].mean()),
                "prop_margin_vs_random_gt_0": float(g["mvr_gt0"].mean()),
            }
        )
    return rows


def save_boxplot(token_df: pd.DataFrame, out_path: Path) -> None:
    labels = [x for x in ["V", "T", "U"] if x in set(token_df["token_label"])]
    data = [token_df[token_df["token_label"] == lb]["ratio_gt_mean"].values for lb in labels]
    plt.figure(figsize=(6.2, 4.6))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(1.0, linestyle="--", linewidth=1, color="black")
    plt.xlabel("Token label")
    plt.ylabel("Mean GT/background ratio")
    plt.title("Visual vs Textual token ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_layer_curve(layer_rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(layer_rows)
    plt.figure(figsize=(8.0, 4.8))
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


def save_sample_paired_plot(sample_rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(sample_rows)
    if df.empty:
        return
    pivot = df[df["token_label"].isin(["V", "T"])].pivot(index="sample_id", columns="token_label", values="ratio_gt_mean")
    pivot = pivot.dropna(subset=["V", "T"], how="any")
    if pivot.empty:
        return
    plt.figure(figsize=(7.0, 4.8))
    for sid, row in pivot.iterrows():
        plt.plot([0, 1], [row["T"], row["V"]], marker="o")
        plt.text(1.02, row["V"], str(sid), fontsize=8, va="center")
    plt.xticks([0, 1], ["T", "V"])
    plt.ylabel("Mean ratio_gt")
    plt.title("Per-sample T→V change")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_heatmap(df: pd.DataFrame, sample_id: str, out_path: Path, value_col: str = "ratio_gt") -> None:
    sample = df[df["sample_id"].astype(str) == str(sample_id)].copy()
    if sample.empty:
        return
    sample = sample.sort_values(["token_index", "layer"])
    pivot = sample.pivot_table(index="layer", columns="token_index", values=value_col, aggfunc="mean")
    token_map = (
        sample[["token_index", "token_text", "token_label"]]
        .drop_duplicates()
        .sort_values("token_index")
    )
    xticks = list(range(len(token_map)))
    xticklabels = [f"{r.token_text.strip()}\n({r.token_label})" for r in token_map.itertuples(index=False)]

    plt.figure(figsize=(max(10, len(xticks) * 0.28), 5.8))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(xticks, xticklabels, rotation=90, fontsize=7)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Token")
    plt.ylabel("Layer")
    plt.title(f"Sample {sample_id}: token × layer {value_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_top_tokens_table(token_df: pd.DataFrame, sample_id: str, out_path: Path, k: int = 12) -> None:
    sub = token_df[token_df["sample_id"].astype(str) == str(sample_id)].copy()
    if sub.empty:
        return
    cols = [
        "sample_id",
        "token_index",
        "token_text",
        "token_label",
        "ratio_gt_mean",
        "margin_vs_random_mean",
        "ratio_gt_early",
        "ratio_gt_mid",
        "ratio_gt_late",
    ]
    sub.sort_values("ratio_gt_mean", ascending=False)[cols].head(k).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual/textual token attention patterns on selected samples")
    parser.add_argument("--token_labeled_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--sample_ids",
        type=str,
        required=True,
        help="Comma-separated sample ids, e.g. 101420,102644,106881,108495,110638",
    )
    parser.add_argument("--heatmap_value", type=str, default="ratio_gt", choices=["ratio_gt", "margin_vs_random"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = parse_sample_ids(args.sample_ids)
    if not sample_ids:
        raise ValueError("--sample_ids is empty")

    df = read_jsonl(Path(args.token_labeled_jsonl))
    df["sample_id"] = df["sample_id"].astype(str)
    df["token_label"] = df.get("token_label", "U").map(label_bucket)
    df = df[df["sample_id"].isin(sample_ids)].copy()
    if df.empty:
        raise ValueError("No rows left after filtering sample_ids")

    token_df = aggregate_token_level(df)
    label_summary = summarize_by_label(token_df)
    layer_summary = summarize_layer_by_label(df)
    sample_label_summary = summarize_sample_label(token_df)

    payload = {
        "sample_ids": sample_ids,
        "num_token_layer_rows": int(len(df)),
        "num_token_rows": int(len(token_df)),
        "label_summary": label_summary,
        "sample_label_summary": sample_label_summary,
        "layer_summary": layer_summary,
    }
    (out_dir / "visual_textual_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    token_df.to_csv(out_dir / "token_level_aggregated.csv", index=False)
    pd.DataFrame(sample_label_summary).to_csv(out_dir / "sample_label_summary.csv", index=False)

    save_boxplot(token_df, out_dir / "visual_vs_textual_boxplot.png")
    save_layer_curve(layer_summary, out_dir / "visual_vs_textual_layer_curve.png")
    save_sample_paired_plot(sample_label_summary, out_dir / "sample_v_vs_t_paired_plot.png")

    for sid in sample_ids:
        save_heatmap(df, sid, out_dir / f"sample_{sid}_token_layer_{args.heatmap_value}.png", value_col=args.heatmap_value)
        save_top_tokens_table(token_df, sid, out_dir / f"sample_{sid}_top_tokens.csv")

    print(f"[ok] saved outputs to {out_dir}")
    print(f"[ok] sample_ids = {sample_ids}")


if __name__ == "__main__":
    main()
