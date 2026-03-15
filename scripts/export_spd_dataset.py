from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image


def merge_side_by_side(img1: Image.Image, img2: Image.Image) -> Image.Image:
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")


    target_h = max(img1.height, img2.height)

    def resize_keep_aspect(img: Image.Image, target_height: int) -> Image.Image:
        if img.height == target_height:
            return img
        new_w = int(img.width * target_height / img.height)
        return img.resize((new_w, target_height))

    img1 = resize_keep_aspect(img1, target_h)
    img2 = resize_keep_aspect(img2, target_h)

    merged = Image.new("RGB", (img1.width + img2.width, target_h))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))
    return merged


def export_split(split_name: str, out_root: Path, limit: int | None = None) -> None:
    ds = load_dataset("Jackson-Lv/SPD-Faith-Bench", split=split_name)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    split_root = out_root / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / f"{split_name}.jsonl"

    with manifest_path.open("w", encoding="utf-8") as fout:
        for ex in ds:
            sample_id = str(ex["image_id"])
            sample_dir = split_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            img1 = ex["image1"]
            img2 = ex["image2"]

            original_path = sample_dir / f"{sample_id}_original.jpg"
            modified_path = sample_dir / f"{sample_id}_modified_final.jpg"
            merged_path = sample_dir / "merged.jpg"
            metadata_path = sample_dir / "metadata.json"

            img1.save(original_path)
            img2.save(modified_path)
            merge_side_by_side(img1, img2).save(merged_path)

            ground_truth = {
                "num_differences": int(ex["num_differences"]),
                "modifications": ex["differences"],
            }

            metadata = {
                "sample_id": sample_id,
                "split": split_name,
                "original_path": str(original_path.resolve()),
                "modified_path": str(modified_path.resolve()),
                "merged_path": str(merged_path.resolve()),
                "ground_truth": ground_truth,
                "question_same": "Are the two pictures the same?",
                "question_different": "Are the two pictures different?",
                "question_open": "Find all differences between the two pictures.",
                "is_faithful": 0,  # 先占位，供作者 analysis 脚本使用
            }

            with metadata_path.open("w", encoding="utf-8") as mf:
                json.dump(metadata, mf, ensure_ascii=False, indent=2)

            # 供 analysis/analyze_layer_changes.py 使用
            manifest_item = {
                "sample_id": sample_id,
                "image_path": str(merged_path.resolve()),
                "question_same": metadata["question_same"],
                "question_open": metadata["question_open"],
                "ground_truth": ground_truth,
                "is_faithful": 0,
            }
            fout.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")

    print(f"[OK] exported split={split_name} to {split_root}")
    print(f"[OK] manifest saved to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["multi_diff"],
        help="例如: easy medium hard multi_diff",
    )
    parser.add_argument("--out_root", type=str, default="work/spd_local")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        export_split(split_name, out_root, args.limit)


if __name__ == "__main__":
    main()