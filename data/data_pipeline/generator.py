"""
generator.py

Drop-in replacement for data/data_pipeline/generator.py.

Main change:
- Object removal no longer hard-codes SimpleLama.
- Removal and position-change cleanup now go through the pluggable inpainting backend:
  lama / opencv-telea / opencv-ns / sd15 / sdxl / flux-fill.

Expected companion file:
- data/data_pipeline/inpainting_backends.py
"""

import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from config import COLOR_MAP, PROCESSED_DATA_DIR
from utils import create_mask_from_segmentation, load_coco_image

# api_client can fail during import if API keys / SSL cert env vars are not set.
# In fallback-only ablation mode we do not need it, so keep this import soft.
try:
    from api_client import client, gemini_client
except Exception as e:  # pragma: no cover - environment-dependent
    print(f"⚠ api_client import failed; online VLM/Gemini selection will be disabled: {e}")
    client = None
    gemini_client = None

# Keep SimpleLama optional. The new backend abstraction imports it only when needed,
# but this method is retained for backward compatibility.
try:
    from simple_lama_inpainting import SimpleLama
except Exception:  # pragma: no cover - optional dependency
    SimpleLama = None

try:
    from inpainting_backends import build_inpainter
except Exception:  # pragma: no cover - package import fallback
    try:
        from .inpainting_backends import build_inpainter
    except Exception:
        build_inpainter = None


# 全局标志，确保GPU检测信息只打印一次
_gpu_info_printed = False


class SpotDifferenceGenerator:
    def __init__(
        self,
        coco_ann_file,
        coco_img_dir,
        inpaint_backend: str = "lama",
        inpaint_backend_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.coco = COCO(coco_ann_file)
        self.coco_img_dir = Path(coco_img_dir)
        self.client = client
        self.group_counter = 0

        # Old LaMa field, kept for backward compatibility.
        self.lama_model = None

        # New pluggable inpainting backend.
        self.inpaint_backend_name = inpaint_backend
        self.inpaint_backend_kwargs = inpaint_backend_kwargs or {}
        self.inpainter = None

    # ---------------------------------------------------------------------
    # Inpainting backend helpers
    # ---------------------------------------------------------------------
    def _init_lama_model(self):
        """Backward-compatible SimpleLama initializer.

        The new code path uses _init_inpainter/_run_inpaint. This method remains
        so older callers do not break.
        """
        global _gpu_info_printed

        if SimpleLama is None:
            raise ImportError(
                "simple_lama_inpainting is not installed. Install it or use a different "
                "inpaint_backend such as opencv-telea or sdxl."
            )

        if self.lama_model is None:
            try:
                import torch

                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device = torch.device("cuda")
                    try:
                        test_tensor = torch.zeros(1).to(device)
                        del test_tensor
                        torch.cuda.empty_cache()
                        if not _gpu_info_printed:
                            device_name = torch.cuda.get_device_name(0)
                            print(f"✓ 检测到GPU: {device_name}, 将使用GPU加速")
                            _gpu_info_printed = True
                    except Exception as e:
                        if not _gpu_info_printed:
                            print(f"⚠ GPU检测失败: {e}, 将使用CPU")
                            _gpu_info_printed = True
                        device = torch.device("cpu")
                else:
                    device = torch.device("cpu")
                    if not _gpu_info_printed:
                        print("⚠ 未检测到CUDA设备，将使用CPU")
                        print("  提示：如果您的系统有GPU，请确保：")
                        print(
                            "  1. 已安装CUDA版本的PyTorch: "
                            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                        )
                        print("  2. GPU驱动已正确安装")
                        print("  3. CUDA环境变量已正确设置")
                        _gpu_info_printed = True

                self.lama_model = SimpleLama(device=device)
            except Exception as e:
                if not _gpu_info_printed:
                    print(f"⚠ 初始化GPU设备失败: {e}, 使用默认设置")
                    _gpu_info_printed = True
                self.lama_model = SimpleLama()

    def _init_inpainter(self):
        if self.inpainter is None:
            if build_inpainter is None:
                raise ImportError(
                    "Cannot import build_inpainter. Make sure "
                    "data/data_pipeline/inpainting_backends.py exists."
                )
            print(
                f"[Inpaint] Initializing backend={self.inpaint_backend_name}, "
                f"kwargs={self.inpaint_backend_kwargs}"
            )
            self.inpainter = build_inpainter(
                self.inpaint_backend_name,
                **self.inpaint_backend_kwargs,
            )

    def _run_inpaint(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
    ) -> np.ndarray:
        """Run the selected inpainting backend.

        Args:
            img_bgr: OpenCV BGR image.
            mask_u8: uint8 mask, 255 = repaint / inpaint region.
            prompt: Used by diffusion/FLUX backends; ignored by LaMa/OpenCV.
            negative_prompt: Used by diffusion backends when supported.
            seed: Used by diffusion/FLUX backends.
        """
        self._init_inpainter()
        return self.inpainter(
            img_bgr,
            mask_u8,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )

    @staticmethod
    def _remove_prompt(category_name: str) -> str:
        return (
            f"Remove the {category_name} and fill the masked area with a clean, "
            f"natural, realistic background consistent with the surrounding scene. "
            f"Do not add new objects."
        )

    @staticmethod
    def _remove_negative_prompt(category_name: str) -> str:
        return (
            f"{category_name}, duplicate object, extra object, extra animal, extra person, "
            f"artifact, blurry, distorted, warped texture, unnatural edge, color shift, "
            f"inconsistent lighting"
        )

    # ---------------------------------------------------------------------
    # Image / mask utilities
    # ---------------------------------------------------------------------
    def _pad_to_multiple_of_8(self, img):
        """将图像填充到8的倍数。保留旧 LaMa 代码路径可能会用到。"""
        h, w = img.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            border_value = 0 if img.ndim == 2 else None
            if img.ndim == 2:
                img_padded = cv2.copyMakeBorder(
                    img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=border_value
                )
            else:
                img_padded = cv2.copyMakeBorder(
                    img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
                )
            return img_padded, (pad_h, pad_w)
        return img, (0, 0)

    def _crop_padding(self, img, pad_h, pad_w):
        """裁剪填充区域。"""
        if pad_h > 0 or pad_w > 0:
            h, w = img.shape[:2]
            return img[: h - pad_h if pad_h > 0 else h, : w - pad_w if pad_w > 0 else w]
        return img

    def _smooth_mask_edges(self, mask, kernel_size=3):
        """平滑 / 膨胀 mask 边缘，使修复区域过渡更自然。"""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_smooth = cv2.GaussianBlur(mask_dilated, (kernel_size, kernel_size), 0)
        _, mask_binary = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
        return mask_binary.astype(np.uint8)

    @staticmethod
    def _image_to_base64_jpeg(img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    @staticmethod
    def _safe_json_from_text(response_text: str) -> Dict[str, Any]:
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            if match:
                response_text = match.group(0)
        return json.loads(response_text)

    # ---------------------------------------------------------------------
    # Online VLM / LLM selectors
    # ---------------------------------------------------------------------
    def _ask_llm_color_change(self, img, objects_info, excluded_colors=None):
        """询问LLM选择要改变颜色的对象和目标颜色。"""
        try:
            if self.client is None:
                return None

            img_base64 = self._image_to_base64_jpeg(img)
            objects_text = ""
            for i, obj in enumerate(objects_info):
                x, y, w, h = obj["bbox"]
                area_percent = obj["area_ratio"] * 100
                objects_text += f"ID{i}: {obj['category']}\n"
                objects_text += (
                    f"  - Position: [x={x:.0f}, y={y:.0f}, "
                    f"width={w:.0f}, height={h:.0f}]\n"
                )
                objects_text += f"  - Size: {area_percent:.2f}% of image area\n\n"

            excluded_colors_text = ""
            if excluded_colors:
                excluded_colors_text = (
                    "\nIMPORTANT: The following colors have already been used in previous "
                    f"differences in this image group: {', '.join(excluded_colors)}. "
                    "Please choose a DIFFERENT color from the available colors list."
                )

            prompt = f"""You are analyzing object annotations for a "spot the difference" puzzle.

Below are the objects detected in an image:
{objects_text}

Your task:
1. Select ONE object that would be suitable for color change.
   Consider:
   - The object should have appropriate size.
   - It should commonly appear in multiple colors in real life.
   - Good examples: cars, umbrellas, clothing, bags, bicycles, trucks, buses.
   - Bad examples: grass, sky, bananas, trees.
2. Determine what color to change it to:
   - Available colors: red, orange, yellow, lime, green, cyan, blue, purple, pink, magenta.
   - Choose a target color that creates a noticeable but realistic change.
   - The target color should be different from the object's current color.
{excluded_colors_text}

IMPORTANT: Return ONLY a valid JSON object in this exact format:
{{
  "selected_object_id": 0,
  "object_name": "",
  "original_color": "",
  "target_color": ""
}}
"""
            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            response_text = message.choices[0].message.content.strip()
            return self._safe_json_from_text(response_text)
        except Exception:
            import traceback

            traceback.print_exc()
            return None

    def _ask_llm_remove_object(self, img, objects_info):
        """询问LLM选择要移除的对象；失败返回 None，调用方 fallback。"""
        try:
            if self.client is None:
                return None

            img_annotated = img.copy()
            for i, obj_info in enumerate(objects_info):
                x, y, w, h = obj_info["bbox"]
                cv2.rectangle(
                    img_annotated,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),
                    2,
                )
                label = f"ID{i}: {obj_info['category']}"
                cv2.putText(
                    img_annotated,
                    label,
                    (int(x), max(15, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            img_base64 = self._image_to_base64_jpeg(img_annotated)
            objects_text = "\n".join(
                [
                    (
                        f"ID{i}: {obj['category']} at position "
                        f"[x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, "
                        f"w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}] "
                        f"| size={obj['area_ratio'] * 100:.2f}%"
                    )
                    for i, obj in enumerate(objects_info)
                ]
            )
            area_ratios = [obj["area_ratio"] for obj in objects_info]
            median_area_pct = float(np.median(area_ratios) * 100.0) if area_ratios else 0.0

            prompt = f"""You are analyzing an image for a "spot the difference" puzzle.

The image contains the following objects marked with green boxes and IDs:
{objects_text}

Your task:
1. Select ONE object to REMOVE from the image.
2. Prefer a moderately sized, non-salient foreground object near the median area: ~{median_area_pct:.2f}% of image area.
3. Avoid background / structural surfaces such as sky, grass, ground, walls, road, large tables.
4. Avoid extremely tiny or heavily occluded objects.
5. The removal should be physically plausible and should not break scene integrity.

IMPORTANT: Return ONLY a valid JSON object in this exact format:
{{
  "selected_object_id": 0,
  "object_name": "",
  "reason": ""
}}
"""
            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            response_text = message.choices[0].message.content.strip()
            return self._safe_json_from_text(response_text)
        except Exception:
            import traceback

            traceback.print_exc()
            return None

    def _ask_vlm_position_change(self, img, objects_info):
        """询问VLM选择要移动的对象和新位置。"""
        try:
            if self.client is None:
                return None

            img_annotated = img.copy()
            for i, obj_info in enumerate(objects_info):
                x, y, w, h = obj_info["bbox"]
                cv2.rectangle(
                    img_annotated,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),
                    2,
                )
                label = f"ID{i}: {obj_info['category']}"
                cv2.putText(
                    img_annotated,
                    label,
                    (int(x), max(15, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            img_base64 = self._image_to_base64_jpeg(img_annotated)
            objects_text = "\n".join(
                [
                    (
                        f"ID{i}: {obj['category']} at position "
                        f"[x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, "
                        f"w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}]"
                    )
                    for i, obj in enumerate(objects_info)
                ]
            )

            prompt = f"""You are analyzing an image for a "spot the difference" puzzle.

The image contains these objects marked with green boxes and IDs:
{objects_text}

Your task:
1. Select ONE object suitable for a position change; prefer smaller movable objects.
2. Determine its current position.
3. Suggest a reasonable new position that is visually plausible and noticeable.

Hard constraints:
- new_bbox MUST have the SAME width and height as original_bbox. Only x and y change.
- new_bbox MUST be fully inside the image boundaries.
- new_bbox MUST NOT overlap with other object bounding boxes.
- The object must remain physically plausible: on floor/ground/road/table when appropriate; never floating.

IMPORTANT: Return ONLY a valid JSON object in this exact format:
{{
  "selected_object_id": 0,
  "object_name": "",
  "original_bbox": [x, y, w, h],
  "new_bbox": [new_x, new_y, w, h],
  "reason": ""
}}
"""
            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            response_text = message.choices[0].message.content.strip()
            return self._safe_json_from_text(response_text)
        except Exception:
            import traceback

            traceback.print_exc()
            return None

    # ---------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------
    def _get_image_and_annotations(self, image_id, difficulty=None):
        """根据 image_id 获取图像和 COCO 标注。"""
        file_name = f"{int(image_id):012d}.jpg"
        img_ids = self.coco.getImgIds()
        coco_img_id = None

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            if img_info["file_name"] == file_name:
                coco_img_id = img_id
                break

        if coco_img_id is None:
            raise ValueError(f"未找到image_id为 {image_id} 的图像")

        img_info = self.coco.loadImgs(coco_img_id)[0]
        img = load_coco_image(self.coco_img_dir, img_info)
        if img is None:
            raise ValueError(f"无法加载图像: {file_name}")

        ann_ids = self.coco.getAnnIds(imgIds=coco_img_id)
        anns = self.coco.loadAnns(ann_ids)
        return img, img_info, anns

    def test_get_image_and_annotations(self, image_id):
        img, img_info, anns = self._get_image_and_annotations(image_id)
        print("img:", img)
        print("img_info:", img_info)
        print("anns:", anns)
        return "ok"

    # ---------------------------------------------------------------------
    # Object selection helpers
    # ---------------------------------------------------------------------
    def _available_annotations(self, anns, excluded_indices=None):
        if excluded_indices is None:
            excluded_indices = []
        excluded = set(int(x) for x in excluded_indices)
        available_indices = [i for i in range(len(anns)) if i not in excluded]
        available_anns = [anns[i] for i in available_indices]
        return available_indices, available_anns

    def _objects_info_for_anns(self, img: np.ndarray, anns: Sequence[Dict[str, Any]], include_area=True):
        img_area = float(img.shape[0] * img.shape[1])
        objects_info = []
        for ann in anns:
            cat_info = self.coco.loadCats(ann["category_id"])[0]
            entry = {
                "category": cat_info["name"],
                "bbox": ann["bbox"],
            }
            if include_area:
                entry["area_ratio"] = float(ann.get("area", 0.0)) / max(img_area, 1.0)
            objects_info.append(entry)
        return objects_info

    @staticmethod
    def _select_median_area_annotation(available_indices, available_anns):
        areas = [float(ann.get("area", 0.0)) for ann in available_anns]
        median_area = float(np.median(areas)) if areas else 0.0
        selected_idx_in_available = min(
            range(len(available_anns)),
            key=lambda i: abs(float(available_anns[i].get("area", 0.0)) - median_area),
        )
        selected_original_idx = int(available_indices[selected_idx_in_available])
        selected_ann = available_anns[selected_idx_in_available]
        return selected_original_idx, selected_ann

    # ---------------------------------------------------------------------
    # Method 1: remove object
    # ---------------------------------------------------------------------
    def _remove_object(
        self,
        image_id,
        excluded_indices=None,
        force_object_index: Optional[int] = None,
        seed: int = 0,
    ):
        """移除对象。

        Args:
            image_id: 图像ID。
            excluded_indices: 已处理过的对象索引列表，避免重复处理。
            force_object_index: 强制选择 COCO annotation index，ablation 时用于公平比较。
            seed: diffusion / FLUX backend 的随机种子。
        """
        img, img_info, anns = self._get_image_and_annotations(image_id)
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")

        selected_ann = None
        selected_original_idx = None
        selection_reason = None

        if force_object_index is not None:
            force_object_index = int(force_object_index)
            if force_object_index < 0 or force_object_index >= len(anns):
                raise ValueError(
                    f"force_object_index={force_object_index} 超出范围，当前图像共有 {len(anns)} 个对象"
                )
            selected_original_idx = force_object_index
            selected_ann = anns[selected_original_idx]
            selection_reason = "forced object index"
        else:
            available_indices, available_anns = self._available_annotations(anns, excluded_indices)
            if not available_anns:
                raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")

            objects_info = self._objects_info_for_anns(img, available_anns, include_area=True)
            llm_choice = self._ask_llm_remove_object(img, objects_info)

            if llm_choice is not None:
                try:
                    idx = int(llm_choice.get("selected_object_id"))
                    if 0 <= idx < len(available_anns):
                        selected_original_idx = int(available_indices[idx])
                        selected_ann = anns[selected_original_idx]
                        selection_reason = llm_choice.get("reason")
                except Exception:
                    selected_ann = None

            if selected_ann is None:
                selected_original_idx, selected_ann = self._select_median_area_annotation(
                    available_indices, available_anns
                )
                selection_reason = "fallback: median-area selection"

        cat_info = self.coco.loadCats(selected_ann["category_id"])[0]
        category_name = cat_info["name"]

        # Create and smooth mask. 255 means repaint/inpaint.
        mask = create_mask_from_segmentation(img.shape, selected_ann["segmentation"])
        mask = self._smooth_mask_edges(mask, kernel_size=3)

        # This is the key fix: use the selected backend instead of hard-coded LaMa.
        result_bgr = self._run_inpaint(
            img,
            mask,
            prompt=self._remove_prompt(category_name),
            negative_prompt=self._remove_negative_prompt(category_name),
            seed=seed,
        )

        log = {
            "type": "remove",
            "bbox": [float(x) for x in selected_ann["bbox"]],
            "category": category_name,
            "category_id": int(selected_ann["category_id"]),
            "area": float(selected_ann.get("area", 0.0)),
            "selection_reason": selection_reason,
            "object_index": int(selected_original_idx),
            "inpaint_backend": self.inpaint_backend_name,
            "inpaint_backend_kwargs": self.inpaint_backend_kwargs,
            "seed": int(seed),
        }
        return {
            "image": result_bgr,
            "log": log,
            "processed_index": int(selected_original_idx),
        }

    # ---------------------------------------------------------------------
    # Method 2: change object color
    # ---------------------------------------------------------------------
    def _change_object_color(self, image_id, excluded_indices=None, used_colors=None):
        """改变对象颜色。"""
        img, img_info, anns = self._get_image_and_annotations(image_id)
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")

        available_indices, available_anns = self._available_annotations(anns, excluded_indices)
        if not available_anns:
            raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")

        objects_info = self._objects_info_for_anns(img, available_anns, include_area=True)
        if used_colors is None:
            used_colors = []

        llm_result = self._ask_llm_color_change(img, objects_info, excluded_colors=used_colors)
        if llm_result is None:
            raise ValueError("LLM调用失败，无法选择物体和颜色")

        try:
            selected_idx = int(llm_result["selected_object_id"])
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"LLM返回的物体ID格式错误: {e}")

        if selected_idx < 0 or selected_idx >= len(available_anns):
            raise ValueError(f"选择的物体ID {selected_idx} 超出范围 (0-{len(available_anns) - 1})")

        original_color = llm_result.get("original_color", "unknown")
        target_color = str(llm_result.get("target_color", "")).lower().strip()
        if target_color not in COLOR_MAP:
            raise ValueError(f"不支持的颜色: {target_color}. 支持的颜色: {list(COLOR_MAP.keys())}")

        original_idx = int(available_indices[selected_idx])
        selected_ann = anns[original_idx]
        cat_info = self.coco.loadCats(selected_ann["category_id"])[0]

        mask = create_mask_from_segmentation(img.shape, selected_ann["segmentation"])
        mask_bool = mask > 127

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        target_hue = COLOR_MAP[target_color]
        img_hsv[mask_bool, 0] = target_hue
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        log = {
            "type": "color",
            "bbox": [float(x) for x in selected_ann["bbox"]],
            "category": cat_info["name"],
            "category_id": int(selected_ann["category_id"]),
            "original_color": original_color,
            "target_color": target_color,
            "area": float(selected_ann.get("area", 0.0)),
            "object_index": int(original_idx),
        }
        return {
            "image": result,
            "log": log,
            "processed_index": int(original_idx),
            "used_color": target_color,
        }

    # ---------------------------------------------------------------------
    # Method 3: change object position
    # ---------------------------------------------------------------------
    def _change_object_position(self, image_id, excluded_indices=None, seed: int = 0):
        """改变对象位置。

        This method now also uses the selected backend to remove the object's
        original location and to clean paste edges.
        """
        img, img_info, anns = self._get_image_and_annotations(image_id)
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")

        available_indices, available_anns = self._available_annotations(anns, excluded_indices)
        if not available_anns:
            raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")

        objects_info = self._objects_info_for_anns(img, available_anns, include_area=False)
        vlm_result = self._ask_vlm_position_change(img, objects_info)
        if vlm_result is None:
            raise ValueError("VLM调用失败，无法选择物体和位置")

        try:
            selected_idx = int(vlm_result["selected_object_id"])
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"VLM返回的物体ID格式错误: {e}")

        if selected_idx < 0 or selected_idx >= len(available_anns):
            raise ValueError(f"选择的物体ID {selected_idx} 超出范围 (0-{len(available_anns) - 1})")

        original_idx = int(available_indices[selected_idx])
        selected_ann = anns[original_idx]
        cat_info = self.coco.loadCats(selected_ann["category_id"])[0]
        category_name = cat_info["name"]

        mask = create_mask_from_segmentation(img.shape, selected_ann["segmentation"])
        original_bbox = selected_ann["bbox"]
        new_bbox = vlm_result["new_bbox"]

        try:
            x1, y1, w, h = [int(round(v)) for v in original_bbox]
            x2, y2 = int(round(new_bbox[0])), int(round(new_bbox[1]))
            img_h, img_w = img.shape[:2]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(0, x2), max(0, y2)
            w = min(w, img_w - x1, img_w - x2)
            h = min(h, img_h - y1, img_h - y2)
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"裁剪后的物体尺寸无效: w={w}, h={h}, 原始bbox={original_bbox}, 新bbox={new_bbox}"
                )

            object_region = img[y1 : y1 + h, x1 : x1 + w].copy()
            object_mask = mask[y1 : y1 + h, x1 : x1 + w].copy()
            if object_mask.size == 0 or object_mask.shape[0] == 0 or object_mask.shape[1] == 0:
                raise ValueError(
                    f"object_mask 为空: shape={object_mask.shape}, y1={y1}, x1={x1}, h={h}, w={w}"
                )
            if np.sum(object_mask > 0) == 0:
                raise ValueError("object_mask 中没有非零像素，无法进行物体移动")

            # 1) Remove object from original position via selected backend.
            mask_smooth = self._smooth_mask_edges(mask, kernel_size=3)
            img_removed = self._run_inpaint(
                img,
                mask_smooth,
                prompt=self._remove_prompt(category_name),
                negative_prompt=self._remove_negative_prompt(category_name),
                seed=seed,
            )

            # 2) Paste object to the new position.
            result = img_removed.copy()
            object_mask_3c = cv2.cvtColor(object_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            alpha = object_mask_3c.astype(np.float32) / 255.0
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            for c in range(3):
                result[y2 : y2 + h, x2 : x2 + w, c] = (
                    alpha[:, :, c] * object_region[:, :, c]
                    + (1.0 - alpha[:, :, c]) * result[y2 : y2 + h, x2 : x2 + w, c]
                )

            # 3) Optional edge cleanup around pasted object.
            edge_mask = np.zeros_like(mask)
            edge_mask[y2 : y2 + h, x2 : x2 + w] = object_mask
            kernel = np.ones((5, 5), np.uint8)
            edge_dilated = cv2.dilate(edge_mask, kernel, iterations=2)
            edge_only = cv2.subtract(edge_dilated, edge_mask)
            if edge_only.sum() > 0:
                edge_only_smooth = self._smooth_mask_edges(edge_only, kernel_size=3)
                result = self._run_inpaint(
                    result,
                    edge_only_smooth,
                    prompt=(
                        f"Clean only the boundary around the pasted {category_name}; "
                        f"keep the object and the background natural."
                    ),
                    negative_prompt=self._remove_negative_prompt(category_name),
                    seed=seed,
                )

            log = {
                "type": "position",
                "original_bbox": [float(x) for x in original_bbox],
                "new_bbox": [float(x) for x in new_bbox],
                "category": category_name,
                "category_id": int(selected_ann["category_id"]),
                "area": float(selected_ann.get("area", 0.0)),
                "object_index": int(original_idx),
                "inpaint_backend": self.inpaint_backend_name,
                "inpaint_backend_kwargs": self.inpaint_backend_kwargs,
                "seed": int(seed),
            }
            return {
                "image": result,
                "log": log,
                "processed_index": int(original_idx),
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"image": img, "log": {"type": "position", "error": str(e)}}

    # ---------------------------------------------------------------------
    # Difference type selector and generation entry points
    # ---------------------------------------------------------------------
    def _ask_llm_difference_type(self, img, anns, num_differences):
        try:
            if self.client is None:
                # Conservative fallback: one remove for easy, otherwise a mixed set.
                base = ["remove", "color", "position", "color", "remove"]
                return {"differences": base[:num_differences]}

            img_base64 = self._image_to_base64_jpeg(img)
            img_area = img.shape[0] * img.shape[1]
            objects_info = []
            for ann in anns:
                cat_info = self.coco.loadCats(ann["category_id"])[0]
                area_ratio = ann.get("area", 0.0) / max(img_area, 1)
                objects_info.append(
                    {
                        "category": cat_info["name"],
                        "bbox": ann["bbox"],
                        "area_ratio": area_ratio,
                    }
                )

            objects_text = "\n".join(
                [
                    (
                        f"ID{i}: {obj['category']} at position "
                        f"[x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, "
                        f"w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}] "
                        f"| size={obj['area_ratio'] * 100:.2f}% of image"
                    )
                    for i, obj in enumerate(objects_info)
                ]
            )

            prompt = f"""You are analyzing an image for a "spot the difference" puzzle.

Your task is to select {num_differences} difference types that will be applied to objects in this image.

Objects:
{objects_text}

Available difference types:
1. "remove" - Remove an object using inpainting.
2. "color" - Change an object's color.
3. "position" - Move an object.

Selection rules:
- The modified image must remain natural and realistic.
- Avoid removing large objects, background elements, or structural surfaces.
- Prefer color change for large objects.
- Prefer position change for movable objects.
- All differences must be visible but not jarring.
- Select exactly {num_differences} items.

Return ONLY valid JSON:
{{
  "differences": ["remove", "color", "position"]
}}
"""
            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            response_text = message.choices[0].message.content.strip()
            result = self._safe_json_from_text(response_text)

            diffs = result.get("differences", [])
            valid = [d for d in diffs if d in {"remove", "color", "position"}]
            if len(valid) != num_differences:
                base = ["remove", "color", "position", "color", "remove"]
                valid = base[:num_differences]
            return {"differences": valid}
        except Exception:
            import traceback

            traceback.print_exc()
            base = ["remove", "color", "position", "color", "remove"]
            return {"differences": base[:num_differences]}

    def generate_single_group(self, image_id, complexity="easy"):
        """生成单个图像组，包含多个差异。"""
        if complexity.lower() not in ["easy", "medium", "hard"]:
            raise ValueError(f"不支持的复杂度: {complexity}")

        complexity_config = {"easy": 1, "medium": 3, "hard": 5}
        num_differences = complexity_config[complexity.lower()]

        img, img_info, anns = self._get_image_and_annotations(image_id, difficulty=complexity)
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")

        llm_result = self._ask_llm_difference_type(img, anns, num_differences)
        diff_types = llm_result["differences"]

        output_dir = PROCESSED_DATA_DIR / complexity.lower() / str(image_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_image_path = output_dir / f"{image_id}_original.jpg"
        cv2.imwrite(str(output_image_path), img)

        all_logs = []
        processed_indices = []
        used_colors = []

        for idx, diff_type in enumerate(diff_types, start=1):
            try:
                if diff_type == "remove":
                    result = self._remove_object(image_id, excluded_indices=processed_indices, seed=idx)
                    if "processed_index" in result:
                        processed_indices.append(result["processed_index"])
                elif diff_type == "color":
                    result = self._change_object_color(
                        image_id,
                        excluded_indices=processed_indices,
                        used_colors=used_colors,
                    )
                    if "processed_index" in result:
                        processed_indices.append(result["processed_index"])
                    if "used_color" in result:
                        used_colors.append(result["used_color"])
                elif diff_type == "position":
                    result = self._change_object_position(
                        image_id,
                        excluded_indices=processed_indices,
                        seed=idx,
                    )
                    if "processed_index" in result:
                        processed_indices.append(result["processed_index"])
                else:
                    raise ValueError(f"不支持的差异类型: {diff_type}")

                output_image_path = output_dir / f"{image_id}_modified_diff_{idx}.jpg"
                cv2.imwrite(str(output_image_path), result["image"])

                log_entry = result["log"].copy()
                log_entry["diff_index"] = idx
                all_logs.append(log_entry)
            except Exception as e:
                import traceback

                traceback.print_exc()
                all_logs.append({"diff_index": idx, "type": diff_type, "error": str(e)})

        log_data = {
            "image_id": image_id,
            "complexity": complexity.lower(),
            "num_differences": len(diff_types),
            "differences": all_logs,
            "timestamp": datetime.now().isoformat(),
            "inpaint_backend": self.inpaint_backend_name,
            "inpaint_backend_kwargs": self.inpaint_backend_kwargs,
        }
        log_path = output_dir / "log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return output_dir

    def generate_single_group_1_to_n(self, image_id, complexity="medium"):
        """生成单个图像组：原图 + Gemini 直接生成的包含多个差异的图片。"""
        if complexity.lower() not in ["medium", "hard"]:
            raise ValueError(f"不支持的复杂度: {complexity}")
        if gemini_client is None:
            raise RuntimeError("gemini_client 不可用，无法运行 generate_single_group_1_to_n")

        num_differences = 3 if complexity.lower() == "medium" else 5
        img, _, _ = self._get_image_and_annotations(image_id, difficulty=complexity)
        original_height, original_width = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        output_dir = PROCESSED_DATA_DIR / f"{complexity.lower()}-gemini" / str(image_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{image_id}_original.jpg"), img)

        prompt = f"""You are creating a "spot the difference" puzzle dataset.

Your task is to create a modified version of this image with exactly {num_differences} differences.

CRITICAL REQUIREMENTS:
1. The modified image should be identical to the original except for {num_differences} small localized changes.
2. Each difference must be confined to a small region.
3. You can only make these changes:
   - REMOVE: remove a small foreground object.
   - COLOR: change one object's color to a realistic color.
   - POSITION: move one object to a plausible new position.
4. After generating the modified image, provide JSON with exact bounding boxes.

Bounding boxes must use [x, y, width, height], where x,y are top-left pixel coordinates.

Return:
1. The modified image.
2. A JSON object with this structure:
{{
  "differences": [
    {{"type": "remove|color", "bbox": [x, y, width, height], "description": "..."}},
    {{"type": "position", "original_bbox": [x, y, width, height], "new_bbox": [x, y, width, height], "description": "..."}}
  ]
}}
"""
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, pil_img],
        )

        generated_image = None
        bbox_text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    bbox_text += part.text + "\n"
                elif getattr(part, "inline_data", None):
                    image_data = part.inline_data.data
                    generated_image = Image.open(BytesIO(image_data))

        if generated_image is None:
            raise ValueError("模型未返回生成的图片")

        generated_width, generated_height = generated_image.size
        if generated_width != original_width or generated_height != original_height:
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:  # older Pillow
                resample = Image.LANCZOS
            generated_image = generated_image.resize((original_width, original_height), resample)

        generated_image.save(str(output_dir / f"{image_id}_modified.jpg"))

        differences = []
        if bbox_text:
            try:
                bbox_data = self._safe_json_from_text(bbox_text)
                differences = bbox_data.get("differences", [])
            except Exception as e:
                print(f"解析bbox信息失败: {e}")
                differences = [{"type": "unknown", "bbox": None, "description": bbox_text}]

        log_data = {
            "image_id": image_id,
            "complexity": complexity.lower(),
            "num_differences": num_differences,
            "differences": differences,
            "timestamp": datetime.now().isoformat(),
        }
        log_path = output_dir / "log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        return output_dir
