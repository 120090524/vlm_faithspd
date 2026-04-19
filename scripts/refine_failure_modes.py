from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def aggregate_sample_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["token_label"] = df.get("token_label", "U").fillna("U")

    sample_rows = []
    for sample_id, g in df.groupby("sample_id"):
        v = g[g["token_label"] == "V"]
        t = g[g["token_label"] == "T"]
        u = g[g["token_label"] == "U"]

        def safe_mean(frame: pd.DataFrame, col: str) -> float:
            return float(frame[col].mean()) if len(frame) else float("nan")

        row: dict[str, Any] = {
            "sample_id": str(sample_id),
            "n_rows": int(len(g)),
            "n_visual_rows": int(len(v)),
            "n_text_rows": int(len(t)),
            "n_uncertain_rows": int(len(u)),
            "visual_ratio_mean": safe_mean(v, "ratio_gt"),
            "text_ratio_mean": safe_mean(t, "ratio_gt"),
            "overall_ratio_mean": float(g["ratio_gt"].mean()),
            "visual_margin_random_mean": safe_mean(v, "margin_vs_random"),
            "text_margin_random_mean": safe_mean(t, "margin_vs_random"),
            "overall_margin_random_mean": float(g["margin_vs_random"].mean()),
            "is_faithful": int(g["is_faithful"].max()) if "is_faithful" in g else -1,
            "num_recall": float(g["num_recall"].max()) if "num_recall" in g else float("nan"),
            "type_f1": float(g["type_f1"].max()) if "type_f1" in g else float("nan"),
            "category_f1": float(g["category_f1"].max()) if "category_f1" in g else float("nan"),
            "response_text": g.iloc[0].get("response_text", ""),
        }
        sample_rows.append(row)
    return pd.DataFrame(sample_rows)


def classify_sample(row: pd.Series, blind_thr: float, seen_thr: float, rand_thr: float) -> str:
    v_ratio = row.get("visual_ratio_mean")
    t_ratio = row.get("text_ratio_mean")
    o_ratio = row.get("overall_ratio_mean")
    v_rand = row.get("visual_margin_random_mean")
    faithful = int(row.get("is_faithful", -1))
    n_visual = int(row.get("n_visual_rows", 0))

    if n_visual == 0:
        return "insufficient_visual_labels"

    if faithful == 1 and pd.notna(v_ratio) and v_ratio >= seen_thr:
        return "success_visual_grounded"

    if faithful == 0 and pd.notna(v_ratio) and v_ratio <= blind_thr and pd.notna(o_ratio) and o_ratio <= blind_thr:
        return "A_visual_token_blindness"

    if faithful == 0 and pd.notna(o_ratio) and o_ratio > seen_thr and pd.notna(v_ratio) and v_ratio <= blind_thr and pd.notna(t_ratio) and t_ratio > v_ratio:
        return "B_average_ratio_not_visual"

    if faithful == 0 and pd.notna(v_ratio) and v_ratio > seen_thr and pd.notna(v_rand) and v_rand > rand_thr:
        return "C_saw_region_but_answer_failed"

    if faithful == 0 and pd.notna(v_ratio) and v_ratio > seen_thr and pd.notna(v_rand) and v_rand <= rand_thr:
        return "D_spurious_or_non_gt_saliency"

    if faithful == 1 and pd.notna(v_ratio) and v_ratio <= blind_thr:
        return "success_but_low_visual_ratio"

    return "other_needs_manual_review"


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine SPD failure modes using visual/textual token labels")
    parser.add_argument("--token_labeled_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--blind_thr", type=float, default=1.0)
    parser.add_argument("--seen_thr", type=float, default=1.2)
    parser.add_argument("--rand_thr", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(Path(args.token_labeled_jsonl))
    sample_df = aggregate_sample_view(df)
    sample_df["failure_mode"] = sample_df.apply(
        classify_sample,
        axis=1,
        blind_thr=args.blind_thr,
        seen_thr=args.seen_thr,
        rand_thr=args.rand_thr,
    )

    summary = (
        sample_df.groupby("failure_mode").size().reset_index(name="count").sort_values("count", ascending=False)
    )

    payload = {
        "num_samples": int(len(sample_df)),
        "failure_mode_counts": summary.to_dict(orient="records"),
        "thresholds": {
            "blind_thr": args.blind_thr,
            "seen_thr": args.seen_thr,
            "rand_thr": args.rand_thr,
        },
    }

    (out_dir / "failure_mode_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_df.to_csv(out_dir / "failure_mode_per_sample.csv", index=False)

    # Also export a compact casebook-like JSON for easy manual reading.
    casebook = {}
    for mode, g in sample_df.groupby("failure_mode"):
        casebook[mode] = g.sort_values("visual_ratio_mean", ascending=False).head(5).to_dict(orient="records")
    (out_dir / "failure_mode_casebook.json").write_text(json.dumps(casebook, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
