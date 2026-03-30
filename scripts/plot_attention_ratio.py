#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot attention-ratio analysis figures from remove-only full-result JSON.

Expected input:
- original result json produced by attention_ratio_qwen25_remove_only_v2.py
  containing:
    {
      "results": [
        {
          "sample_id": "...",
          "ratio_sample_mean": ...,
          "ratio_random_mean": ...,
          "margin_vs_random": ...,
          "ratio_cross_image": ...,
          "gt_beats_random": ...,
          "per_layer_step_ratio": {
              "0": [...],
              "1": [...],
              ...
          },
          ...
        },
        ...
      ]
    }

Outputs:
- ratio_distribution_hist.png
- ratio_distribution_scatter.png
- margin_distribution.png
- heatmap_average_layer_step.png
- layer_summary_curve.png
- layer_prop_gt1_curve.png
- step_quartiles_bar.png
- heatmap_sample_<sample_id>.png   (optional)

Usage example:
python scripts/plot_attention_ratio.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/figures \
  --sample_id 108495
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attention-ratio analysis figures.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original attention-ratio result JSON.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="Optional sample_id for single-sample layer×step heatmap.",
    )
    parser.add_argument(
        "--target_steps",
        type=int,
        default=40,
        help="Normalized number of steps for averaged heatmap.",
    )
    return parser.parse_args()


def load_results(input_file: Path) -> Tuple[dict, List[dict]]:
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if "results" not in payload:
        raise ValueError(f"'results' field not found in {input_file}")
    rows = payload["results"]
    if not rows:
        raise ValueError(f"No results found in {input_file}")
    return payload, rows


def extract_basic_arrays(rows: List[dict]) -> Dict[str, np.ndarray]:
    ratio_sample = np.array([float(r["ratio_sample_mean"]) for r in rows], dtype=float)
    ratio_random = np.array([float(r["ratio_random_mean"]) for r in rows], dtype=float)
    margin = np.array([float(r["margin_vs_random"]) for r in rows], dtype=float)
    cross_image = np.array([float(r["ratio_cross_image"]) for r in rows], dtype=float)
    gt_beats_random = np.array([bool(r["gt_beats_random"]) for r in rows], dtype=bool)

    return {
        "ratio_sample": ratio_sample,
        "ratio_random": ratio_random,
        "margin": margin,
        "cross_image": cross_image,
        "gt_beats_random": gt_beats_random,
    }


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_ratio_hist(
    ratio_sample: np.ndarray,
    ratio_random: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.8))
    plt.hist(ratio_sample, bins=30, alpha=0.6, label="GT removed-region ratio")
    plt.hist(ratio_random, bins=30, alpha=0.6, label="Random-mask ratio")
    plt.axvline(ratio_sample.mean(), linestyle="--", linewidth=1)
    plt.axvline(ratio_random.mean(), linestyle="--", linewidth=1)
    plt.xlabel("Ratio")
    plt.ylabel("Count")
    plt.title("Distribution: ratio_sample_mean vs ratio_random_mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_ratio_scatter(
    ratio_sample: np.ndarray,
    ratio_random: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(ratio_random, ratio_sample, alpha=0.6)
    lo = min(ratio_random.min(), ratio_sample.min())
    hi = max(ratio_random.max(), ratio_sample.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("ratio_random_mean")
    plt.ylabel("ratio_sample_mean")
    plt.title("Per-sample GT vs Random ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_margin_hist(margin: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 4.8))
    plt.hist(margin, bins=30, alpha=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=1, color="black")
    plt.axvline(margin.mean(), linestyle="--", linewidth=1)
    plt.xlabel("margin_vs_random")
    plt.ylabel("Count")
    plt.title("Distribution of margin_vs_random")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def get_sorted_layer_keys(per_layer_step_ratio: Dict[str, List[float]]) -> List[str]:
    return sorted(per_layer_step_ratio.keys(), key=lambda x: int(x))


def get_sample_matrix(sample: dict) -> Tuple[np.ndarray, List[str]]:
    layer_keys = get_sorted_layer_keys(sample["per_layer_step_ratio"])
    mat = np.array([sample["per_layer_step_ratio"][k] for k in layer_keys], dtype=float)
    return mat, layer_keys


def resize_step_axis(mat: np.ndarray, target_steps: int) -> np.ndarray:
    L, T = mat.shape
    old_x = np.linspace(0.0, 1.0, T)
    new_x = np.linspace(0.0, 1.0, target_steps)
    resized = np.zeros((L, target_steps), dtype=float)
    for i in range(L):
        resized[i] = np.interp(new_x, old_x, mat[i])
    return resized


def save_single_heatmap(sample: dict, out_path: Path) -> None:
    mat, layer_keys = get_sample_matrix(sample)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(im, label="ratio")
    plt.yticks(range(len(layer_keys)), layer_keys)
    plt.xlabel("Generation step")
    plt.ylabel("Layer")
    plt.title(f"Layer × Step Heatmap (sample {sample['sample_id']})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_average_heatmap(rows: List[dict], target_steps: int, out_path: Path) -> None:
    mats = []
    layer_keys_ref = None

    for r in rows:
        mat, layer_keys = get_sample_matrix(r)
        resized = resize_step_axis(mat, target_steps)
        mats.append(resized)
        if layer_keys_ref is None:
            layer_keys_ref = layer_keys

    avg_mat = np.mean(np.stack(mats, axis=0), axis=0)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(avg_mat, aspect="auto", origin="lower")
    plt.colorbar(im, label="mean ratio")
    plt.yticks(range(len(layer_keys_ref)), layer_keys_ref)
    plt.xlabel("Normalized generation step")
    plt.ylabel("Layer")
    plt.title("Average Layer × Step Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def compute_layer_summary(rows: List[dict]) -> List[dict]:
    mats = []
    layer_keys_ref = None

    for r in rows:
        mat, layer_keys = get_sample_matrix(r)
        mats.append(mat)
        if layer_keys_ref is None:
            layer_keys_ref = layer_keys

    layer_values = {k: [] for k in layer_keys_ref}

    for mat in mats:
        for i, k in enumerate(layer_keys_ref):
            vals = mat[i]
            layer_values[k].extend(vals.tolist())

    summary = []
    for k in layer_keys_ref:
        vals = np.array(layer_values[k], dtype=float)
        summary.append(
            {
                "layer": int(k),
                "mean_ratio": float(vals.mean()),
                "median_ratio": float(np.median(vals)),
                "prop_gt1": float((vals > 1.0).mean()),
            }
        )
    return summary


def save_layer_summary_curves(layer_summary: List[dict], out_mean: Path, out_prop: Path) -> None:
    layers = [x["layer"] for x in layer_summary]
    means = [x["mean_ratio"] for x in layer_summary]
    props = [x["prop_gt1"] for x in layer_summary]

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, means, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Mean ratio")
    plt.title("Layer summary: mean ratio by layer")
    plt.tight_layout()
    plt.savefig(out_mean, dpi=220)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(layers, props, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Proportion > 1")
    plt.title("Layer summary: proportion > 1 by layer")
    plt.tight_layout()
    plt.savefig(out_prop, dpi=220)
    plt.close()


def compute_step_quartiles(rows: List[dict]) -> Dict[str, dict]:
    quartile_values = {0: [], 1: [], 2: [], 3: []}

    for r in rows:
        mat, _ = get_sample_matrix(r)  # [L, T]
        _, T = mat.shape

        # 每个 step 先对 layer 求均值，得到 step-level ratio
        step_vals = mat.mean(axis=0)

        # 按 step 分成 4 段
        idx = np.arange(T)
        bins = np.floor(4 * idx / T).astype(int)
        bins = np.clip(bins, 0, 3)

        for q in range(4):
            quartile_values[q].extend(step_vals[bins == q].tolist())

    out = {}
    for q in range(4):
        vals = np.array(quartile_values[q], dtype=float)
        out[str(q)] = {
            "mean_ratio": float(vals.mean()) if len(vals) else float("nan"),
            "count": int(len(vals)),
        }
    return out


def save_step_quartiles_bar(step_quartiles: Dict[str, dict], out_path: Path) -> None:
    quartiles = sorted(step_quartiles.keys(), key=int)
    means = [step_quartiles[q]["mean_ratio"] for q in quartiles]

    plt.figure(figsize=(6, 4.5))
    plt.bar(quartiles, means)
    plt.xlabel("Step quartile")
    plt.ylabel("Mean ratio")
    plt.title("Step quartile summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def print_console_summary(arrs: Dict[str, np.ndarray]) -> None:
    ratio_sample = arrs["ratio_sample"]
    ratio_random = arrs["ratio_random"]
    margin = arrs["margin"]
    cross_image = arrs["cross_image"]
    gt_beats_random = arrs["gt_beats_random"]

    print("=" * 60)
    print("Basic summary")
    print("=" * 60)
    print(f"ratio_sample_mean: mean={ratio_sample.mean():.6f}, median={np.median(ratio_sample):.6f}")
    print(f"ratio_random_mean: mean={ratio_random.mean():.6f}, median={np.median(ratio_random):.6f}")
    print(f"margin_vs_random: mean={margin.mean():.6f}, median={np.median(margin):.6f}")
    print(f"cross_image_mean: mean={cross_image.mean():.6f}, median={np.median(cross_image):.6f}")
    print(f"gt_beats_random_rate: {gt_beats_random.mean():.6f}")
    print(f"positive_margin_rate: {(margin > 0).mean():.6f}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    input_file = Path(args.input)
    out_dir = Path(args.out_dir)

    ensure_out_dir(out_dir)

    payload, rows = load_results(input_file)
    arrs = extract_basic_arrays(rows)

    # 1) ratio_sample_mean vs ratio_random_mean
    save_ratio_hist(
        ratio_sample=arrs["ratio_sample"],
        ratio_random=arrs["ratio_random"],
        out_path=out_dir / "ratio_distribution_hist.png",
    )

    save_ratio_scatter(
        ratio_sample=arrs["ratio_sample"],
        ratio_random=arrs["ratio_random"],
        out_path=out_dir / "ratio_distribution_scatter.png",
    )

    # 2) margin_vs_random
    save_margin_hist(
        margin=arrs["margin"],
        out_path=out_dir / "margin_distribution.png",
    )

    # 3) average layer × step heatmap
    save_average_heatmap(
        rows=rows,
        target_steps=args.target_steps,
        out_path=out_dir / "heatmap_average_layer_step.png",
    )

    # 4) single sample heatmap (optional)
    if args.sample_id is not None:
        matched = None
        for r in rows:
            if str(r["sample_id"]) == str(args.sample_id):
                matched = r
                break
        if matched is None:
            raise ValueError(f"sample_id={args.sample_id} not found in input JSON")
        save_single_heatmap(
            sample=matched,
            out_path=out_dir / f"heatmap_sample_{args.sample_id}.png",
        )

    # 5) layer summary curves
    layer_summary = compute_layer_summary(rows)
    save_layer_summary_curves(
        layer_summary=layer_summary,
        out_mean=out_dir / "layer_summary_curve.png",
        out_prop=out_dir / "layer_prop_gt1_curve.png",
    )

    # 6) step quartiles
    step_quartiles = compute_step_quartiles(rows)
    save_step_quartiles_bar(
        step_quartiles=step_quartiles,
        out_path=out_dir / "step_quartiles_bar.png",
    )

    # 7) save derived summaries
    derived = {
        "model": payload.get("model"),
        "manifest_file": payload.get("manifest_file"),
        "split": payload.get("split"),
        "question_mode": payload.get("question_mode"),
        "num_samples": len(rows),
        "basic_summary": {
            "ratio_sample_mean_mean": float(arrs["ratio_sample"].mean()),
            "ratio_sample_mean_median": float(np.median(arrs["ratio_sample"])),
            "ratio_random_mean_mean": float(arrs["ratio_random"].mean()),
            "ratio_random_mean_median": float(np.median(arrs["ratio_random"])),
            "margin_mean": float(arrs["margin"].mean()),
            "margin_median": float(np.median(arrs["margin"])),
            "cross_image_mean": float(arrs["cross_image"].mean()),
            "cross_image_median": float(np.median(arrs["cross_image"])),
            "gt_beats_random_rate": float(arrs["gt_beats_random"].mean()),
            "positive_margin_rate": float((arrs["margin"] > 0).mean()),
        },
        "layer_summary": layer_summary,
        "step_quartiles": step_quartiles,
    }

    (out_dir / "plot_summary.json").write_text(
        json.dumps(derived, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_console_summary(arrs)
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()