from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def calc_type_f1(metric_row: dict[str, Any]) -> float:
    gt = 0
    pred = 0
    matched = 0

    for t in ["color", "remove", "position"]:
        d = metric_row["type_level"].get(t, {})
        gt += int(d.get("gt_count", 0))
        pred += int(d.get("pred_count", 0))
        matched += int(d.get("matched", 0))

    if gt + pred == 0:
        return 0.0

    return 2.0 * matched / (gt + pred)


def judge_faithful(metric_row: dict[str, Any]) -> int:
    """
    用 soft rule 判定 faithful。
    这个阈值适合你现在的 smoke test / 小样本 analysis 分组。
    """
    num_recall = float(metric_row.get("num_recall", 0.0))
    category_f1 = float(metric_row.get("category_level", {}).get("f1", 0.0))
    type_f1 = calc_type_f1(metric_row)

    # 推荐阈值：
    # - num_recall 至少要过 0.4，说明数量上不是完全跑偏
    # - category_f1 至少 0.25，说明类别识别不是完全错
    # - type_f1 至少 0.55，说明 color/remove/position 类型整体还行
    if num_recall >= 0.4 and category_f1 >= 0.25 and type_f1 >= 0.55:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses_file", type=str, required=True)
    parser.add_argument("--metrics_file", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--split", type=str, default="multi_diff")
    parser.add_argument("--out_file", type=str, default="work/analysis_multi_diff.jsonl")
    args = parser.parse_args()

    responses_file = Path(args.responses_file)
    metrics_file = Path(args.metrics_file)
    data_root = Path(args.data_root)
    split_root = data_root / args.split
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 读取 baseline responses
    responses_data = json.loads(responses_file.read_text(encoding="utf-8"))
    responses = responses_data["responses"]

    # 读取 metrics
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics_rows = metrics_data["per_sample_results"]
    metrics_map = {str(r["sample_id"]): r for r in metrics_rows}

    faithful_cnt = 0
    unfaithful_cnt = 0
    missing_metric_cnt = 0
    missing_image_cnt = 0

    with out_file.open("w", encoding="utf-8") as fout:
        for row in responses:
            sample_id = str(row["sample_id"])

            sample_dir = split_root / sample_id
            merged_path = sample_dir / "merged.jpg"
            original_path = sample_dir / f"{sample_id}_original.jpg"
            modified_path = sample_dir / f"{sample_id}_modified_final.jpg"

            if not merged_path.exists():
                print(f"[Skip] merged image not found: {merged_path}")
                missing_image_cnt += 1
                continue

            metric_row = metrics_map.get(sample_id)
            if metric_row is None:
                print(f"[Skip] metric not found for sample_id={sample_id}")
                missing_metric_cnt += 1
                continue

            gt = row.get("ground_truth", {})
            parsed = row.get("parsed_response", {})

            num_recall = float(metric_row.get("num_recall", 0.0))
            category_f1 = float(metric_row.get("category_level", {}).get("f1", 0.0))
            type_f1 = calc_type_f1(metric_row)

            is_faithful = judge_faithful(metric_row)

            if is_faithful == 1:
                faithful_cnt += 1
            else:
                unfaithful_cnt += 1

            item = {
                "qid": sample_id,
                "sample_id": sample_id,
                "image": f"{sample_id}/merged.jpg",
                "image_path": str(merged_path.resolve()),
                "original_path": str(original_path.resolve()),
                "modified_path": str(modified_path.resolve()),
                "question_same": "Are the two pictures the same?",
                "question_open": "Find all differences between the two pictures.",
                "ground_truth": gt,
                "predicted_num": parsed.get("predicted_num", None),
                "structured_output": parsed.get("structured_output", []),
                "num_recall": num_recall,
                "type_f1": type_f1,
                "category_f1": category_f1,
                "is_faithful": is_faithful,
            }

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Saved: {out_file}")
    print(f"[OK] faithful={faithful_cnt}, unfaithful={unfaithful_cnt}")
    print(f"[Info] missing_metric={missing_metric_cnt}, missing_image={missing_image_cnt}")


if __name__ == "__main__":
    main()