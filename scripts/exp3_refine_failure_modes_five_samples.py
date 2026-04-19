from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

NO_DIFF_PATTERNS = [
    "no discernible change",
    "identical",
    "not visually apparent",
    "no visible difference",
    "appear to be identical",
]

APPEARANCE_WORDS = {
    "blue", "light", "orange", "hoodie", "jeans", "sneakers", "shirt", "pants",
    "outfit", "color", "wearing", "visible", "same", "different", "reflection",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def maybe_read_analysis(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return load_jsonl(path)


def summarize_sample(group: pd.DataFrame) -> pd.Series:
    sid = str(group["sample_id"].iloc[0])
    response_text = str(group["response_text"].iloc[0])

    def sub(label: str) -> pd.DataFrame:
        return group[group["token_label"] == label].copy()

    V = sub("V")
    T = sub("T")
    U = sub("U")

    def mean_or_nan(df: pd.DataFrame, col: str):
        return float(df[col].mean()) if len(df) else float("nan")

    def prop_true(df: pd.DataFrame, col: str):
        if len(df) == 0:
            return float("nan")
        return float(df[col].mean())

    top_v = []
    if len(V):
        top_v = V.sort_values("ratio_gt_mean", ascending=False)[["token_text", "ratio_gt_mean"]].head(6).to_dict("records")

    top_v_tokens = [str(x["token_text"]).strip() for x in top_v]
    top_v_joined = ", ".join(top_v_tokens)
    appearance_hits = sum(tok.lower() in APPEARANCE_WORDS for tok in top_v_tokens)
    no_diff_lang = any(p in response_text.lower() for p in NO_DIFF_PATTERNS)

    return pd.Series({
        "sample_id": sid,
        "response_text": response_text,
        "n_tokens": int(len(group)),
        "n_V": int(len(V)),
        "n_T": int(len(T)),
        "n_U": int(len(U)),
        "ratio_V": mean_or_nan(V, "ratio_gt_mean"),
        "ratio_T": mean_or_nan(T, "ratio_gt_mean"),
        "ratio_U": mean_or_nan(U, "ratio_gt_mean"),
        "ratio_V_late": mean_or_nan(V, "ratio_gt_late"),
        "ratio_T_late": mean_or_nan(T, "ratio_gt_late"),
        "margin_V": mean_or_nan(V, "margin_vs_random_mean"),
        "margin_T": mean_or_nan(T, "margin_vs_random_mean"),
        "prop_V_gt1": prop_true(V, "ratio_gt_gt1"),
        "prop_T_gt1": prop_true(T, "ratio_gt_gt1"),
        "prop_V_margin_pos": prop_true(V, "mvr_gt0"),
        "prop_T_margin_pos": prop_true(T, "mvr_gt0"),
        "V_minus_T": mean_or_nan(V, "ratio_gt_mean") - mean_or_nan(T, "ratio_gt_mean"),
        "top_v_tokens": top_v_joined,
        "appearance_hits": appearance_hits,
        "no_diff_language": no_diff_lang,
    })


def classify(row: pd.Series) -> tuple[str, str]:
    ratio_diff = row["V_minus_T"]
    ratio_v = row["ratio_V"]
    late_v = row["ratio_V_late"]
    prop_v_gt1 = row["prop_V_gt1"]
    margin_v = row["margin_V"]
    prop_v_margin_pos = row["prop_V_margin_pos"]
    faithful = row.get("is_faithful", None)

    # 1) obvious suspicious / spurious case
    if bool(row["no_diff_language"]) and ratio_v > 1.0:
        return (
            "spurious_or_false_negative_candidate",
            "Response claims no visible difference, but V-token attention to GT region is high; likely false negative, spurious saliency, or data issue.",
        )

    if row["appearance_hits"] >= 2 and ratio_diff > 0.08 and (pd.isna(margin_v) or margin_v <= 0):
        return (
            "spurious_saliency_candidate",
            "Top V tokens are dominated by appearance/color words and margin vs random is weak or negative.",
        )

    # 2) strong attended case
    if ratio_diff > 0.08 and late_v > 1.0 and prop_v_gt1 >= 0.30:
        if faithful == 1:
            return (
                "success_visual_grounded",
                "V tokens clearly exceed T tokens, especially in late layers, and the answer is faithful.",
            )
        if faithful == 0:
            return (
                "saw_region_but_answer_failed",
                "V tokens clearly exceed T tokens and late-layer grounding is strong, but the answer is unfaithful.",
            )
        return (
            "attended_visual_tokens_needs_answer_check",
            "V tokens clearly exceed T tokens and late-layer grounding is strong; answer correctness still needs verification.",
        )

    # 3) average ratio maybe high, but not visually driven
    if ratio_v > 0.9 and ratio_diff <= 0.08:
        return (
            "average_ratio_not_visual",
            "Sample-level/overall ratio may look high, but V tokens do not separate strongly from T tokens.",
        )

    # 4) weak visual grounding
    if ratio_diff <= 0.08 or late_v < 0.9:
        if faithful == 0:
            return (
                "visual_token_blindness",
                "V tokens do not separate clearly from T tokens, consistent with missing the critical visual evidence.",
            )
        return (
            "weak_visual_grounding",
            "There is only weak separation between V and T tokens, so grounding is not convincing yet.",
        )

    return ("manual_review", "Does not match a clean rule; inspect images and token heatmap manually.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_csv", required=True, help="token_level_aggregated.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_ids", default=None, help="Comma-separated sample ids")
    parser.add_argument("--analysis_jsonl", default=None, help="Optional analysis jsonl with is_faithful etc.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.token_csv)
    df["sample_id"] = df["sample_id"].astype(str)
    if args.sample_ids:
        sids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
        df = df[df["sample_id"].isin(sids)].copy()

    sample_df = df.groupby("sample_id", as_index=False).apply(summarize_sample).reset_index(drop=True)

    if args.analysis_jsonl:
        adf = load_jsonl(Path(args.analysis_jsonl))
        adf["sample_id"] = adf["sample_id"].astype(str)
        keep = [c for c in ["sample_id", "is_faithful", "num_recall", "type_f1", "category_f1"] if c in adf.columns]
        adf = adf[keep].drop_duplicates("sample_id")
        sample_df = sample_df.merge(adf, on="sample_id", how="left")

    cats = sample_df.apply(classify, axis=1, result_type="expand")
    sample_df["failure_mode"] = cats[0]
    sample_df["failure_mode_note"] = cats[1]

    sample_df.to_csv(out_dir / "exp3_failure_mode_candidates.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "exp3_casebook.json", "w", encoding="utf-8") as f:
        json.dump(sample_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"saved: {out_dir / 'exp3_failure_mode_candidates.csv'}")
    print(f"saved: {out_dir / 'exp3_casebook.json'}")


if __name__ == "__main__":
    main()
