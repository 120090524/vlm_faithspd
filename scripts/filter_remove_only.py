from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def norm_type(x: Any) -> str:
    return str(x).strip().lower()


def is_remove_only(mods: list[dict[str, Any]]) -> bool:
    if not mods:
        return False
    types = [norm_type(m.get("type", "")) for m in mods if isinstance(m, dict)]
    return bool(types) and all(t == "remove" for t in types)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SPD-Faith exported data into a remove-only split")
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--src_split", type=str, default="multi_diff")
    parser.add_argument("--dst_split", type=str, default="remove_only")
    parser.add_argument("--single_only", action="store_true", help="Keep only samples with exactly one remove difference")
    parser.add_argument("--max_num_differences", type=int, default=None, help="Optional upper bound on num_differences")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination split if it already exists")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    src_manifest = data_root / f"{args.src_split}.jsonl"
    src_split_root = data_root / args.src_split
    dst_manifest = data_root / f"{args.dst_split}.jsonl"
    dst_split_root = data_root / args.dst_split

    if not src_manifest.exists():
        raise FileNotFoundError(f"Source manifest not found: {src_manifest}")
    if not src_split_root.exists():
        raise FileNotFoundError(f"Source split dir not found: {src_split_root}")

    if dst_split_root.exists() and args.overwrite:
        shutil.rmtree(dst_split_root)
    dst_split_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    skipped_missing_meta = 0

    with src_manifest.open("r", encoding="utf-8") as fin, dst_manifest.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            src_sample_dir = src_split_root / sample_id
            meta_path = src_sample_dir / "metadata.json"
            if not meta_path.exists():
                skipped_missing_meta += 1
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            gt = meta.get("ground_truth", {})
            mods = gt.get("modifications", [])
            num_diff = int(gt.get("num_differences", len(mods) if isinstance(mods, list) else 0))

            if not isinstance(mods, list) or not is_remove_only(mods):
                continue
            if args.single_only and len(mods) != 1:
                continue
            if args.max_num_differences is not None and num_diff > args.max_num_differences:
                continue

            dst_sample_dir = dst_split_root / sample_id
            if dst_sample_dir.exists() and args.overwrite:
                shutil.rmtree(dst_sample_dir)
            if not dst_sample_dir.exists():
                shutil.copytree(src_sample_dir, dst_sample_dir)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] total={total}")
    print(f"[OK] kept={kept}")
    print(f"[OK] skipped_missing_meta={skipped_missing_meta}")
    print(f"[OK] dst_split={dst_split_root}")
    print(f"[OK] dst_manifest={dst_manifest}")


if __name__ == "__main__":
    main()
