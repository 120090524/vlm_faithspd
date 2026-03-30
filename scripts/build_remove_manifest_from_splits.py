from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def norm_type(x: Any) -> str:
    return str(x).strip().lower()


def iter_metadata(data_root: Path, splits: list[str]):
    for split in splits:
        split_root = data_root / split
        if not split_root.exists():
            continue
        for sample_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            yield split, sample_dir, meta_path, meta


def keep_sample(meta: dict[str, Any], *, single_only: bool, remove_only: bool, max_num_differences: int | None) -> bool:
    gt = meta.get("ground_truth", {})
    mods = gt.get("modifications", [])
    if not isinstance(mods, list) or not mods:
        return False
    if remove_only and not all(norm_type(m.get("type", "")) == "remove" for m in mods if isinstance(m, dict)):
        return False
    if single_only and len(mods) != 1:
        return False
    if max_num_differences is not None:
        num_diff = int(gt.get("num_differences", len(mods)))
        if num_diff > max_num_differences:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a combined remove-only manifest from exported SPD splits")
    ap.add_argument("--data_root", type=str, default="work/spd_local")
    ap.add_argument("--splits", nargs="+", default=["easy", "medium", "hard", "multi_diff"])
    ap.add_argument("--out_file", type=str, default="work/remove_only_all_single.jsonl")
    ap.add_argument("--single_only", action="store_true", help="Keep only samples with exactly one remove modification")
    ap.add_argument("--remove_only", action="store_true", help="Keep only samples whose all modifications are remove")
    ap.add_argument("--max_num_differences", type=int, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    for split, sample_dir, meta_path, meta in iter_metadata(data_root, args.splits):
        if not keep_sample(
            meta,
            single_only=args.single_only,
            remove_only=args.remove_only,
            max_num_differences=args.max_num_differences,
        ):
            continue
        sample_id = str(meta.get("sample_id", sample_dir.name))
        # disambiguate sample ids if user merges multiple exported roots later
        key = (split, sample_id)
        if key in seen:
            continue
        seen.add(key)
        row = {
            "sample_id": sample_id,
            "source_split": split,
            "sample_dir": str(sample_dir.resolve()),
            "metadata_path": str(meta_path.resolve()),
            "original_path": meta.get("original_path"),
            "modified_path": meta.get("modified_path"),
            "merged_path": meta.get("merged_path"),
            "ground_truth": meta.get("ground_truth", {}),
        }
        rows.append(row)

    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(rows)} samples to {out_file}")
    print(f"[OK] source splits: {', '.join(args.splits)}")


if __name__ == "__main__":
    main()
