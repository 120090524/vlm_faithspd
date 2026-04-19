#!/usr/bin/env python3
"""Utilities to build CSE559 progress-report artifacts from the user's
object-removal outputs.

What it can do:
1) Scan outputs/remove_debug/*/log.json and write a summary CSV.
2) Reconstruct the selected COCO instance mask using object_index from log.json.
3) Save mask/overlay images for report figures.
4) Make a qualitative grid (original / mask overlay / removed).

Example:
    python make_progress_artifacts.py \
        --output-root outputs/remove_debug \
        --coco-ann data/raw/coco/annotations/instances_val2017.json \
        --grid-ids 139 285 632 724 \
        --fig-out outputs/progress_figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_mask_from_segmentation(img_shape: tuple[int, int, int], segmentation) -> np.ndarray:
    """Reimplementation of the repo utility for convenience."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, list):
        # polygon format
        polygons = segmentation
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
    elif isinstance(segmentation, dict):
        # RLE format
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects([rle], h, w)[0]
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        mask = (decoded > 0).astype(np.uint8) * 255
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    return mask


def alpha_overlay_bgr(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    # red overlay in BGR
    color[:, :, 2] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)[mask_bool]
    return overlay


def format_image_id(image_id: int | str) -> str:
    return f"{int(image_id):012d}"


def scan_output_cases(output_root: Path) -> list[dict]:
    cases: list[dict] = []
    for case_dir in sorted(output_root.iterdir()):
        if not case_dir.is_dir():
            continue
        log_path = case_dir / "log.json"
        original_path = case_dir / f"{case_dir.name}_original.jpg"
        removed_path = case_dir / f"{case_dir.name}_removed.jpg"
        if not log_path.exists() or not original_path.exists() or not removed_path.exists():
            continue
        payload = load_json(log_path)
        diff = payload.get("difference", {})
        bbox = diff.get("bbox", [None, None, None, None])
        area = diff.get("area")
        img_area = None
        if original_path.exists():
            img = cv2.imread(str(original_path))
            if img is not None:
                img_area = img.shape[0] * img.shape[1]
        area_ratio = None
        if area is not None and img_area:
            area_ratio = float(area) / float(img_area)
        cases.append(
            {
                "image_id": int(payload["image_id"]),
                "folder": case_dir.name,
                "file_name": payload.get("file_name"),
                "num_annotations": payload.get("num_annotations"),
                "mode": payload.get("mode"),
                "category": diff.get("category"),
                "category_id": diff.get("category_id"),
                "object_index": diff.get("object_index"),
                "area": area,
                "area_ratio": area_ratio,
                "bbox_x": bbox[0] if len(bbox) > 0 else None,
                "bbox_y": bbox[1] if len(bbox) > 1 else None,
                "bbox_w": bbox[2] if len(bbox) > 2 else None,
                "bbox_h": bbox[3] if len(bbox) > 3 else None,
                "selection_reason": diff.get("selection_reason"),
                "original_path": str(original_path),
                "removed_path": str(removed_path),
                "log_path": str(log_path),
            }
        )
    return cases


def write_summary_csv(cases: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "folder",
        "file_name",
        "num_annotations",
        "mode",
        "category",
        "category_id",
        "object_index",
        "area",
        "area_ratio",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "selection_reason",
        "original_path",
        "removed_path",
        "log_path",
        "manual_quality",
        "manual_notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cases:
            out = dict(row)
            out.setdefault("manual_quality", "")
            out.setdefault("manual_notes", "")
            writer.writerow(out)


def export_masks_and_overlays(cases: list[dict], coco_ann_file: Path, fig_out: Path, selected_ids: set[int] | None = None) -> None:
    fig_out.mkdir(parents=True, exist_ok=True)
    coco = COCO(str(coco_ann_file))
    for case in cases:
        image_id = int(case["image_id"])
        if selected_ids is not None and image_id not in selected_ids:
            continue
        object_index = case.get("object_index")
        if object_index is None:
            continue

        original = cv2.imread(case["original_path"])
        if original is None:
            print(f"[WARN] cannot load original image for {image_id}")
            continue

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        if object_index < 0 or object_index >= len(anns):
            print(f"[WARN] object_index out of range for image {image_id}: {object_index}")
            continue

        ann = anns[object_index]
        mask = create_mask_from_segmentation(original.shape, ann["segmentation"])
        overlay = alpha_overlay_bgr(original, mask)

        stem = format_image_id(image_id)
        mask_path = fig_out / f"{stem}_mask.png"
        overlay_path = fig_out / f"{stem}_overlay.png"
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), overlay)


def _rgb_for_matplotlib(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _existing_overlay_path(fig_out: Path, image_id: int) -> Path:
    return fig_out / f"{format_image_id(image_id)}_overlay.png"


def make_grid(cases: list[dict], fig_out: Path, image_ids: Iterable[int], filename: str = "qualitative_grid.png") -> Path:
    chosen = []
    case_map = {int(c["image_id"]): c for c in cases}
    for image_id in image_ids:
        if image_id in case_map:
            chosen.append(case_map[image_id])
    if not chosen:
        raise ValueError("No cases matched the requested image IDs.")

    nrows = len(chosen)
    ncols = 3  # original / overlay / removed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3.3 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, case in enumerate(chosen):
        image_id = int(case["image_id"])
        original = _rgb_for_matplotlib(case["original_path"])
        removed = _rgb_for_matplotlib(case["removed_path"])
        overlay_path = _existing_overlay_path(fig_out, image_id)
        overlay = _rgb_for_matplotlib(overlay_path) if overlay_path.exists() else original

        titles = ["Original", "Selected Mask", "Removed"]
        images = [original, overlay, removed]
        for col_idx, (img, title) in enumerate(zip(images, titles)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                cat = case.get("category", "unknown")
                ratio = case.get("area_ratio")
                ratio_str = f"{100*ratio:.2f}%" if isinstance(ratio, float) else "n/a"
                ax.set_ylabel(f"ID {image_id}\n{cat}\narea={ratio_str}", rotation=0, labelpad=45, va="center")

    plt.tight_layout()
    out_path = fig_out / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create progress-report artifacts from removal outputs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/remove_debug"))
    parser.add_argument("--coco-ann", type=Path, default=None, help="Path to instances_val2017.json")
    parser.add_argument("--fig-out", type=Path, default=Path("outputs/progress_figures"))
    parser.add_argument("--summary-name", type=str, default="summary.csv")
    parser.add_argument(
        "--grid-ids",
        nargs="*",
        type=int,
        default=None,
        help="Image IDs for the qualitative grid. If omitted, use all discovered cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = scan_output_cases(args.output_root)
    if not cases:
        raise SystemExit(f"No valid cases found under: {args.output_root}")

    args.fig_out.mkdir(parents=True, exist_ok=True)
    summary_path = args.fig_out / args.summary_name
    write_summary_csv(cases, summary_path)
    print(f"[OK] Wrote summary CSV: {summary_path}")

    selected_ids = set(args.grid_ids) if args.grid_ids else {int(c["image_id"]) for c in cases}
    if args.coco_ann is not None:
        export_masks_and_overlays(cases, args.coco_ann, args.fig_out, selected_ids=selected_ids)
        print(f"[OK] Exported masks/overlays to: {args.fig_out}")
    else:
        print("[WARN] --coco-ann not provided, so mask/overlay images were not reconstructed.")

    grid_path = make_grid(cases, args.fig_out, image_ids=selected_ids)
    print(f"[OK] Wrote qualitative grid: {grid_path}")


if __name__ == "__main__":
    main()
