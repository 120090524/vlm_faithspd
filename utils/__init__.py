from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def load_coco_image(coco_img_dir, img_info):
    img_path = Path(coco_img_dir) / img_info["file_name"]
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")
    return image


def create_mask_from_segmentation(img_shape, segmentation):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if isinstance(segmentation, list):
        # polygon format
        polygons = segmentation
        if len(polygons) > 0 and not isinstance(polygons[0], list):
            polygons = [polygons]

        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    elif isinstance(segmentation, dict):
        # RLE format
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded > 0, axis=2).astype(np.uint8)
        mask[decoded > 0] = 255

    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    return mask
