from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data" / "data_pipeline"))

from config import COCO_ANN_FILE, COCO_IMG_DIR  # noqa: E402
import api_client  # noqa: E402


def save_side_by_side(img_a, img_b, save_path: Path):
    h = max(img_a.shape[0], img_b.shape[0])

    def resize_keep_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        new_w = int(img.shape[1] * target_h / img.shape[0])
        return cv2.resize(img, (new_w, target_h))

    a = resize_keep_h(img_a, h)
    b = resize_keep_h(img_b, h)
    merged = np.concatenate([a, b], axis=1)
    cv2.imwrite(str(save_path), merged)


def build_generator(fallback_only: bool):
    # 如果只是本地调试 object removal，可以先绕过在线 client 初始化
    if fallback_only:
        api_client.initialize_clients = lambda: (None, None)

    generator_mod = importlib.import_module("generator")
    SpotDifferenceGenerator = generator_mod.SpotDifferenceGenerator
    gen = SpotDifferenceGenerator(COCO_ANN_FILE, COCO_IMG_DIR)

    if fallback_only:
        gen._ask_llm_remove_object = lambda annotated_img, objects_info: None

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", type=str, required=True, help="COCO image id，例如 391895 或 000000391895")
    parser.add_argument("--out_dir", type=str, default="outputs/remove_debug")
    parser.add_argument("--fallback_only", action="store_true", help="跳过在线物体选择，直接走 median-area fallback")
    args = parser.parse_args()

    image_id = str(int(args.image_id))
    out_root = Path(args.out_dir) / f"{int(image_id):012d}"
    out_root.mkdir(parents=True, exist_ok=True)

    gen = build_generator(args.fallback_only)

    original_img, img_info, anns = gen._get_image_and_annotations(image_id)
    result = gen._remove_object(image_id)

    removed_img = result["image"]
    log = result["log"]

    original_path = out_root / f"{int(image_id):012d}_original.jpg"
    removed_path = out_root / f"{int(image_id):012d}_removed.jpg"
    compare_path = out_root / "compare.jpg"
    log_path = out_root / "log.json"

    cv2.imwrite(str(original_path), original_img)
    cv2.imwrite(str(removed_path), removed_img)
    save_side_by_side(original_img, removed_img, compare_path)

    payload = {
        "image_id": int(image_id),
        "file_name": img_info["file_name"],
        "num_annotations": len(anns),
        "difference": log,
        "mode": "fallback_only" if args.fallback_only else "planner_plus_inpaint",
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to {out_root}")
    print(f"[OK] selection log: {log}")


if __name__ == "__main__":
    main()
