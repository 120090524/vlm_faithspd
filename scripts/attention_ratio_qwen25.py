from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


PROMPTS = {
    "same": "Are the two pictures the same? Answer yes or no first, then briefly explain why.",
    "different": "Are the two pictures different? Answer yes or no first, then list all concrete differences you can see.",
    "drf": (
        "Look at the two images carefully.\n\n"
        "First, state the total number of differences you see.\n"
        "Then explain each difference step by step.\n\n"
        "Be explicit and grounded in the images."
    ),
    "json": (
        "You are solving a spot-the-difference task.\n\n"
        "Look at the two images carefully and output ONLY valid JSON.\n"
        "Do not output markdown fences.\n"
        "The JSON schema must be:\n\n"
        "{\n"
        '  "predicted_num": <integer>,\n'
        '  "structured_output": [\n'
        '    {"type": "color" | "remove" | "position", "category": "<object category>"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        '- "predicted_num" must equal the number of entries in "structured_output".\n'
        "- Each difference must be one of: color / remove / position.\n"
        '- "category" should be a short object noun, like "cat", "bus", "remote", "person".\n'
        "- Output JSON only."
    ),
}


@dataclass
class SampleRecord:
    sample_id: str
    original_path: Path
    modified_path: Path
    ground_truth: dict[str, Any]


@dataclass
class VisionSpan:
    image_idx: int
    token_positions: list[int]
    grid_t: int
    grid_h: int
    grid_w: int



def load_manifest(data_root: Path, split: str, limit: int | None = None) -> list[SampleRecord]:
    manifest_path = data_root / f"{split}.jsonl"
    rows: list[SampleRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            meta = json.loads(line)
            sample_id = str(meta["sample_id"])
            sample_dir = data_root / split / sample_id
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue
            full_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append(
                SampleRecord(
                    sample_id=sample_id,
                    original_path=Path(full_meta["original_path"]),
                    modified_path=Path(full_meta["modified_path"]),
                    ground_truth=full_meta["ground_truth"],
                )
            )
    return rows



def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)



def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 1
    return keep



def build_change_masks(
    original_path: Path,
    modified_path: Path,
    diff_threshold: float = 22.0,
    min_area: int = 64,
    dilate_kernel: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (diff_mask, original_rgb, modified_rgb).

    diff_mask is built from absolute RGB difference between the two aligned images.
    """
    img_a = _load_rgb(original_path)
    img_b = _load_rgb(modified_path)
    if img_a.shape[:2] != img_b.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16)).mean(axis=2)
    mask = (diff >= diff_threshold).astype(np.uint8)

    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask = _filter_small_components(mask, min_area=min_area)
    return mask.astype(bool), img_a, img_b



def downsample_mask_to_grid(mask: np.ndarray, grid_h: int, grid_w: int, positive_frac: float = 0.05) -> np.ndarray:
    """Resize a binary mask to the model's image token grid."""
    pooled = cv2.resize(mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    return pooled >= positive_frac



def save_debug_visualization(
    out_dir: Path,
    sample_id: str,
    img_a: np.ndarray,
    img_b: np.ndarray,
    mask: np.ndarray,
    patch_masks: list[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def overlay(img: np.ndarray, binary: np.ndarray) -> np.ndarray:
        over = img.copy()
        over[binary] = (0.65 * over[binary] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        return over

    vis_a = overlay(img_a, mask)
    vis_b = overlay(img_b, mask)
    Image.fromarray(np.concatenate([vis_a, vis_b], axis=1)).save(out_dir / f"{sample_id}_pixel_mask.jpg")

    if len(patch_masks) == 2:
        h, w = img_a.shape[:2]
        patch_a = cv2.resize(patch_masks[0].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        patch_b = cv2.resize(patch_masks[1].astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(np.concatenate([patch_a, patch_b], axis=1)).save(out_dir / f"{sample_id}_patch_mask.jpg")



def build_messages(original_path: Path, modified_path: Path, prompt: str) -> list[dict[str, Any]]:
    img1 = Image.open(original_path).convert("RGB")
    img2 = Image.open(modified_path).convert("RGB")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": prompt},
            ],
        }
    ]



def prepare_inputs(processor: AutoProcessor, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs



def find_vision_spans(
    input_ids: list[int],
    image_grid_thw: torch.Tensor,
    vision_start_token_id: int,
    vision_end_token_id: int,
    image_token_id: int,
) -> list[VisionSpan]:
    """
    Locate image-token spans for each image segment.

    Qwen2.5-VL uses `<|vision_start|> <|image_pad|> ... <|vision_end|>` segments.
    We find the placeholder-token positions between each start/end pair and align them
    with the per-image grid sizes from `image_grid_thw`.
    """
    spans: list[VisionSpan] = []
    cursor = 0
    image_idx = 0

    grids = image_grid_thw.tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw

    while cursor < len(input_ids):
        if input_ids[cursor] != vision_start_token_id:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(input_ids) and input_ids[end] != vision_end_token_id:
            end += 1
        if end >= len(input_ids):
            break

        token_positions = [i for i in range(cursor + 1, end) if input_ids[i] == image_token_id]
        if image_idx >= len(grids):
            break
        grid_t, grid_h, grid_w = [int(v) for v in grids[image_idx]]
        expected = grid_t * grid_h * grid_w
        if len(token_positions) != expected:
            # keep best-effort: if there are extra non-placeholder tokens, trim/pad logic here
            if len(token_positions) > expected:
                token_positions = token_positions[:expected]
            elif len(token_positions) < expected:
                raise ValueError(
                    f"Image span {image_idx}: found {len(token_positions)} image tokens, expected {expected}."
                )

        spans.append(
            VisionSpan(
                image_idx=image_idx,
                token_positions=token_positions,
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
            )
        )
        image_idx += 1
        cursor = end + 1

    return spans



def build_changed_vs_other_token_sets(
    spans: list[VisionSpan],
    diff_mask: np.ndarray,
    positive_frac: float,
) -> tuple[list[int], list[int], list[np.ndarray]]:
    """
    For a two-image prompt, build the changed-token set and the remaining-image-token set.

    Because the spot-the-difference sample is an aligned original/modified pair, we use the
    same pixel difference mask on both images, then downsample it to each image-token grid.
    """
    patch_masks: list[np.ndarray] = []
    changed_token_positions: list[int] = []
    other_token_positions: list[int] = []

    for span in spans:
        mask_grid = downsample_mask_to_grid(diff_mask, span.grid_h, span.grid_w, positive_frac=positive_frac)
        if span.grid_t > 1:
            mask_grid = np.repeat(mask_grid[None, ...], span.grid_t, axis=0)
        else:
            mask_grid = mask_grid[None, ...]

        flat_mask = mask_grid.reshape(-1)
        patch_masks.append(mask_grid[0] if span.grid_t == 1 else mask_grid.max(axis=0))

        for pos, is_changed in zip(span.token_positions, flat_mask.tolist()):
            if is_changed:
                changed_token_positions.append(pos)
            else:
                other_token_positions.append(pos)

    return changed_token_positions, other_token_positions, patch_masks



def decode_tokens(processor: AutoProcessor, token_ids: list[int]) -> list[str]:
    tokenizer = processor.tokenizer
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]



def is_content_token(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if re.fullmatch(r"[\W_]+", tok):
        return False
    return True



def compute_attention_ratio(
    attn_last_query: torch.Tensor,
    changed_positions: list[int],
    other_positions: list[int],
    ratio_mode: str = "mean",
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Compute changed-region attention ratio for a single (layer, generation-step).

    attn_last_query shape: [num_heads, kv_len]
    """
    attn = attn_last_query.float().mean(dim=0)

    if not changed_positions or not other_positions:
        return {
            "changed_score": float("nan"),
            "other_score": float("nan"),
            "ratio": float("nan"),
            "mass_ratio": float("nan"),
        }

    changed_vals = attn[changed_positions]
    other_vals = attn[other_positions]

    if ratio_mode == "mean":
        changed_score = changed_vals.mean().item()
        other_score = other_vals.mean().item()
    elif ratio_mode == "sum":
        changed_score = changed_vals.sum().item()
        other_score = other_vals.sum().item()
    else:
        raise ValueError(f"Unsupported ratio_mode: {ratio_mode}")

    mass_ratio = changed_vals.sum().item() / (other_vals.sum().item() + eps)
    ratio = changed_score / (other_score + eps)

    return {
        "changed_score": changed_score,
        "other_score": other_score,
        "ratio": ratio,
        "mass_ratio": mass_ratio,
    }



def aggregate(values: list[float], mode: str) -> float:
    values = [v for v in values if not math.isnan(v) and math.isfinite(v)]
    if not values:
        return float("nan")
    if mode == "mean":
        return float(np.mean(values))
    if mode == "max":
        return float(np.max(values))
    if mode == "median":
        return float(np.median(values))
    if mode == "last":
        return float(values[-1])
    raise ValueError(f"Unsupported aggregate mode: {mode}")



def run_one_sample(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    sample: SampleRecord,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages = build_messages(sample.original_path, sample.modified_path, prompt)
    inputs = prepare_inputs(processor, messages)
    inputs = inputs.to("cuda")

    input_ids = inputs["input_ids"][0].tolist()
    image_grid_thw = inputs["image_grid_thw"]

    config = model.config
    vision_start_token_id = int(config.vision_start_token_id)
    vision_end_token_id = int(config.vision_end_token_id)
    image_token_id = int(config.image_token_id)

    spans = find_vision_spans(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        vision_start_token_id=vision_start_token_id,
        vision_end_token_id=vision_end_token_id,
        image_token_id=image_token_id,
    )
    if len(spans) != 2:
        raise ValueError(f"Expected 2 image spans, found {len(spans)} for sample {sample.sample_id}.")

    diff_mask, img_a, img_b = build_change_masks(
        sample.original_path,
        sample.modified_path,
        diff_threshold=args.diff_threshold,
        min_area=args.min_area,
        dilate_kernel=args.dilate_kernel,
    )
    changed_positions, other_positions, patch_masks = build_changed_vs_other_token_sets(
        spans=spans,
        diff_mask=diff_mask,
        positive_frac=args.patch_positive_frac,
    )

    if args.debug_vis_dir:
        save_debug_visualization(Path(args.debug_vis_dir), sample.sample_id, img_a, img_b, diff_mask, patch_masks)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0][input_len:].tolist()
    generated_tokens = decode_tokens(processor, generated_ids)

    detailed_steps: list[dict[str, Any]] = []
    selected_layer_ids = list(range(args.layer_start, args.layer_end + 1))

    for step_idx, step_attentions in enumerate(outputs.attentions):
        if step_idx >= len(generated_tokens):
            break
        token_text = generated_tokens[step_idx]
        if args.content_only and not is_content_token(token_text):
            continue

        per_layer: dict[str, Any] = {}
        step_ratios: list[float] = []
        step_mass_ratios: list[float] = []

        for layer_idx in selected_layer_ids:
            attn_t = step_attentions[layer_idx]  # [B, H, q_len, kv_len]
            attn_last_query = attn_t[0, :, -1, :]
            stats = compute_attention_ratio(
                attn_last_query=attn_last_query,
                changed_positions=changed_positions,
                other_positions=other_positions,
                ratio_mode=args.ratio_mode,
                eps=args.eps,
            )
            per_layer[str(layer_idx)] = stats
            if math.isfinite(stats["ratio"]):
                step_ratios.append(stats["ratio"])
            if math.isfinite(stats["mass_ratio"]):
                step_mass_ratios.append(stats["mass_ratio"])

        detailed_steps.append(
            {
                "step_idx": step_idx,
                "token": token_text,
                "ratio_mean_layers": aggregate(step_ratios, "mean"),
                "ratio_max_layers": aggregate(step_ratios, "max"),
                "mass_ratio_mean_layers": aggregate(step_mass_ratios, "mean"),
                "per_layer": per_layer,
            }
        )

    step_mean_ratios = [d["ratio_mean_layers"] for d in detailed_steps if math.isfinite(d["ratio_mean_layers"])]
    step_max_ratios = [d["ratio_max_layers"] for d in detailed_steps if math.isfinite(d["ratio_max_layers"])]
    step_mass_ratios = [d["mass_ratio_mean_layers"] for d in detailed_steps if math.isfinite(d["mass_ratio_mean_layers"])]

    return {
        "sample_id": sample.sample_id,
        "original_path": str(sample.original_path),
        "modified_path": str(sample.modified_path),
        "ground_truth": sample.ground_truth,
        "prompt": prompt,
        "generated_text": processor.tokenizer.decode(generated_ids, skip_special_tokens=True),
        "generated_tokens": generated_tokens,
        "image_grid_thw": image_grid_thw.cpu().tolist(),
        "num_changed_tokens": len(changed_positions),
        "num_other_tokens": len(other_positions),
        "summary": {
            "ratio_sample_mean": aggregate(step_mean_ratios, args.step_aggregate),
            "ratio_sample_peak": aggregate(step_max_ratios, "max"),
            "ratio_sample_last": aggregate(step_mean_ratios, "last"),
            "mass_ratio_sample_mean": aggregate(step_mass_ratios, args.step_aggregate),
        },
        "steps": detailed_steps if args.save_detailed else [],
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Changed-region attention ratio experiment for Qwen2.5-VL")
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--split", type=str, default="multi_diff")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question_mode", choices=sorted(PROMPTS.keys()), default="drf")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=27)
    parser.add_argument("--ratio_mode", choices=["mean", "sum"], default="mean")
    parser.add_argument("--step_aggregate", choices=["mean", "median", "max", "last"], default="mean")
    parser.add_argument("--content_only", action="store_true", help="Only keep non-punctuation generated tokens")
    parser.add_argument("--diff_threshold", type=float, default=22.0)
    parser.add_argument("--min_area", type=int, default=64)
    parser.add_argument("--dilate_kernel", type=int, default=5)
    parser.add_argument("--patch_positive_frac", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save_detailed", action="store_true")
    parser.add_argument("--debug_vis_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="outputs/attention_ratio/qwen25_attention_ratio.json")
    args = parser.parse_args()

    if args.layer_end < args.layer_start:
        raise ValueError("layer_end must be >= layer_start")

    data_root = Path(args.data_root)
    samples = load_manifest(data_root, args.split, args.limit)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    prompt = PROMPTS[args.question_mode]
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        try:
            print(f"[{i}/{len(samples)}] sample_id={sample.sample_id}")
            result = run_one_sample(model, processor, sample, prompt, args)
            results.append(result)
        except Exception as e:  # pragma: no cover - debugging path for user-side runs
            failures.append({"sample_id": sample.sample_id, "error": repr(e)})
            print(f"[WARN] failed sample_id={sample.sample_id}: {e}")
        finally:
            torch.cuda.empty_cache()

    payload = {
        "model": Path(args.model_name).name,
        "question_mode": args.question_mode,
        "ratio_mode": args.ratio_mode,
        "step_aggregate": args.step_aggregate,
        "layer_range": [args.layer_start, args.layer_end],
        "results": results,
        "failures": failures,
    }
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
