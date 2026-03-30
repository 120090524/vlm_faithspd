from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROMPTS = {
    "different": "Are the two pictures different? Answer yes or no first, then list all concrete differences you can see.",
    "open": "Look at the two images carefully. First, state the total number of differences you see. Then explain each difference step by step. Be explicit and grounded in the images.",
    "remove": "One image may have an object removed. Compare the two images carefully and explain all removal differences step by step.",
}


def load_rows(data_root: Path | None, split: str | None, manifest_file: Path | None, limit: int | None) -> list[dict[str, Any]]:
    if manifest_file is not None:
        path = manifest_file
    else:
        assert data_root is not None and split is not None
        path = data_root / f"{split}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def modifications(meta: dict[str, Any]) -> list[dict[str, Any]]:
    mods = meta.get("ground_truth", {}).get("modifications", [])
    return [m for m in mods if isinstance(m, dict)]


def is_remove_only(meta: dict[str, Any]) -> bool:
    mods = modifications(meta)
    return bool(mods) and all(str(m.get("type", "")).strip().lower() == "remove" for m in mods)


def has_bbox(mod: dict[str, Any]) -> bool:
    keys = {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    return keys.issubset(set(mod.keys()))


def bbox_mask(mods: list[dict[str, Any]], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for mod in mods:
        if not has_bbox(mod):
            continue
        x = int(round(float(mod.get("bbox_x", 0))))
        y = int(round(float(mod.get("bbox_y", 0))))
        w = int(round(float(mod.get("bbox_w", 0))))
        h = int(round(float(mod.get("bbox_h", 0))))
        if w <= 0 or h <= 0:
            continue
        x0 = max(0, min(width - 1, x))
        y0 = max(0, min(height - 1, y))
        x1 = max(x0 + 1, min(width, x0 + w))
        y1 = max(y0 + 1, min(height, y0 + h))
        mask[y0:y1, x0:x1] = 1
    return mask


def pixel_diff_mask(img1: Image.Image, img2: Image.Image, threshold: int = 20, dilate_size: int = 5) -> np.ndarray:
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.int16)
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr1 - arr2).mean(axis=2)
    mask = (diff >= threshold).astype(np.uint8) * 255
    pil = Image.fromarray(mask, mode="L")
    if dilate_size > 1:
        pil = pil.filter(ImageFilter.MaxFilter(size=dilate_size))
    return (np.asarray(pil) > 0).astype(np.uint8)


def build_remove_masks(meta: dict[str, Any], img1: Image.Image, img2: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    mods = modifications(meta)
    m1 = bbox_mask(mods, img1.size)

    d = pixel_diff_mask(img1, img2)
    if m1.sum() == 0:
        m1 = d.copy()
    else:
        if d.shape != m1.shape:
            d = (
                np.asarray(
                    Image.fromarray((d * 255).astype(np.uint8), mode="L").resize(
                        (img1.width, img1.height), Image.Resampling.BILINEAR
                    )
                )
                > 127
            ).astype(np.uint8)
        m1 = np.maximum(m1, d.astype(np.uint8))

    if m1.sum() == 0:
        raise RuntimeError("empty remove mask after bbox/diff fusion")

    m2 = (
        np.asarray(
            Image.fromarray((m1 * 255).astype(np.uint8), mode="L").resize(
                (img2.width, img2.height), Image.Resampling.BILINEAR
            )
        )
        > 127
    ).astype(np.uint8)
    return m1.astype(np.uint8), m2.astype(np.uint8)


def get_merge_size(model) -> int:
    cfg = getattr(model, "config", None)
    vcfg = getattr(cfg, "vision_config", None)
    for name in ["spatial_merge_size", "merge_size"]:
        val = getattr(vcfg, name, None)
        if isinstance(val, int) and val > 0:
            return val
    return 2


def expected_token_counts(image_grid_thw: np.ndarray, merge_size: int) -> list[int]:
    counts: list[int] = []
    for row in image_grid_thw:
        t, h, w = [int(x) for x in row.tolist()]
        token_h = max(1, h // merge_size)
        token_w = max(1, w // merge_size)
        counts.append(max(1, t * token_h * token_w))
    return counts


def infer_token_grid(image_grid_thw_row: np.ndarray, target_len: int, image_size: tuple[int, int], merge_size: int) -> tuple[int, int]:
    t, gh, gw = [int(x) for x in image_grid_thw_row.tolist()]
    token_h = max(1, gh // merge_size)
    token_w = max(1, gw // merge_size)
    if max(1, t * token_h * token_w) == target_len:
        return token_h, token_w
    width, height = image_size
    aspect = max(width / max(height, 1), 1e-6)
    token_h = max(1, int(round(math.sqrt(target_len / aspect))))
    token_w = max(1, int(math.ceil(target_len / token_h)))
    while token_h * token_w < target_len:
        token_w += 1
    return token_h, token_w


def find_vision_spans_fallback(input_ids: list[int], tokenizer) -> list[list[int]]:
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    spans: list[list[int]] = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] == start_id:
            j = i + 1
            while j < len(input_ids) and input_ids[j] != end_id:
                j += 1
            if j < len(input_ids):
                spans.append(list(range(i + 1, j)))
            i = j + 1
        else:
            i += 1
    return spans


def get_image_token_id(model, processor) -> int | None:
    candidates: list[int] = []
    for obj in [getattr(model, "config", None), getattr(getattr(model, "config", None), "vision_config", None), processor, getattr(processor, "tokenizer", None)]:
        if obj is None:
            continue
        for name in ["image_token_id", "vision_token_id", "image_pad_token_id"]:
            val = getattr(obj, name, None)
            if isinstance(val, int) and val >= 0:
                candidates.append(val)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for token_name in ["<|image_pad|>", "<|vision_pad|>", "<image>", "<|img_pad|>"]:
            try:
                tid = tok.convert_tokens_to_ids(token_name)
            except Exception:
                continue
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                candidates.append(tid)
    return candidates[0] if candidates else None


def find_image_token_positions(input_ids: list[int], expected_counts: list[int], model, processor) -> list[np.ndarray]:
    image_token_id = get_image_token_id(model, processor)
    total_needed = int(sum(expected_counts))

    if image_token_id is not None:
        image_positions = [i for i, tid in enumerate(input_ids) if tid == image_token_id]
        if len(image_positions) >= total_needed:
            image_positions = image_positions[:total_needed]
            out: list[np.ndarray] = []
            cur = 0
            for c in expected_counts:
                chunk = np.array(image_positions[cur:cur + c], dtype=np.int64)
                out.append(chunk)
                cur += c
            return out

    # Fallback: use positions between <|vision_start|> and <|vision_end|>, trimmed to expected counts.
    raw_spans = find_vision_spans_fallback(input_ids, processor.tokenizer)
    if len(raw_spans) < len(expected_counts):
        raise RuntimeError(f"cannot find enough image spans; found={len(raw_spans)}, need={len(expected_counts)}")

    out = []
    for raw, c in zip(raw_spans, expected_counts):
        if len(raw) < c:
            raise RuntimeError(f"fallback span too short: span_len={len(raw)}, need={c}")
        out.append(np.array(raw[:c], dtype=np.int64))
    return out


def resize_mask_to_token(mask: np.ndarray, token_h: int, token_w: int, target_len: int, threshold: float = 0.12) -> np.ndarray:
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    arr = np.asarray(pil.resize((token_w, token_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    flat = arr.reshape(-1)
    if flat.size > target_len:
        flat = flat[:target_len]
    elif flat.size < target_len:
        flat = np.concatenate([flat, np.zeros(target_len - flat.size, dtype=np.float32)], axis=0)
    binary = flat >= threshold
    # Ensure both positive and negative sets are non-empty.
    if binary.sum() == 0:
        binary[int(np.argmax(flat))] = True
    if binary.sum() == binary.size:
        binary[int(np.argmin(flat))] = False
    return binary.astype(bool)


def random_same_size_mask(flat_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.zeros_like(flat_mask, dtype=bool)
    k = int(flat_mask.sum())
    if k <= 0:
        return out
    idx = list(range(flat_mask.size))
    rng.shuffle(idx)
    out[np.array(idx[:k], dtype=np.int64)] = True
    if out.sum() == out.size:
        out[idx[-1]] = False
    return out


def content_token_indices(tokens: list[str]) -> list[int]:
    idx = [i for i, tok in enumerate(tokens) if any(ch.isalnum() for ch in tok)]
    return idx if idx else list(range(len(tokens)))


def attn_ratio(attn_vec: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> float:
    eps = 1e-8
    pos = attn_vec.index_select(0, pos_idx).mean().item() if pos_idx.numel() > 0 else 0.0
    neg = attn_vec.index_select(0, neg_idx).mean().item() if neg_idx.numel() > 0 else 0.0
    return pos / max(neg, eps)


def messages_for_two_images(img1: Image.Image, img2: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "image", "image": img2},
            {"type": "text", "text": prompt},
        ],
    }]


def save_debug_overlay(img: Image.Image, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ys, xs = np.where(mask > 0)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=(255, 0, 0, 70))
    out = Image.alpha_composite(canvas, overlay)
    out.save(path)


def get_meta_from_row(row: dict[str, Any], data_root: Path | None = None, split: str | None = None) -> dict[str, Any]:
    if "metadata_path" in row:
        return json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
    if data_root is not None and split is not None and "sample_id" in row:
        meta_path = data_root / split / str(row["sample_id"]) / "metadata.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"cannot locate metadata for row: keys={list(row.keys())}")


def analyze_sample(model, processor, meta: dict[str, Any], prompt: str, layer_start: int, layer_end: int,
                   content_only: bool, n_random_masks: int, seed: int, orig_only: bool,
                   debug_vis_dir: Path | None = None) -> dict[str, Any]:
    img1 = Image.open(meta["original_path"]).convert("RGB")
    img2 = Image.open(meta["modified_path"]).convert("RGB")
    mask1, mask2 = build_remove_masks(meta, img1, img2)

    if debug_vis_dir is not None:
        sample_id = str(meta["sample_id"])
        save_debug_overlay(img1, mask1, debug_vis_dir / f"{sample_id}_orig_mask.png")
        save_debug_overlay(img2, mask2, debug_vis_dir / f"{sample_id}_mod_region.png")

    messages = messages_for_two_images(img1, img2, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    # keep on cpu for inspection, move to model device afterwards
    input_ids = inputs["input_ids"][0].tolist()
    grids = inputs["image_grid_thw"].detach().cpu().numpy()

    expected_counts = expected_token_counts(grids, get_merge_size(model))
    image_positions = find_image_token_positions(input_ids, expected_counts, model, processor)
    if len(image_positions) < 2:
        raise RuntimeError(f"expected 2 image token groups, got {len(image_positions)}")

    input_len = len(input_ids)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    gen_ids = outputs.sequences[0][input_len:].tolist()
    gen_tokens = [processor.tokenizer.decode([tid]) for tid in gen_ids]
    step_indices = content_token_indices(gen_tokens) if content_only else list(range(len(gen_tokens)))
    if not step_indices:
        raise RuntimeError("no generated content tokens")

    pos_raw1 = image_positions[0]
    pos_raw2 = image_positions[1]
    span_len1 = len(pos_raw1)
    span_len2 = len(pos_raw2)
    th1, tw1 = infer_token_grid(grids[0], span_len1, img1.size, get_merge_size(model))
    th2, tw2 = infer_token_grid(grids[1], span_len2, img2.size, get_merge_size(model))

    flat1 = resize_mask_to_token(mask1, th1, tw1, span_len1)
    flat2 = resize_mask_to_token(mask2, th2, tw2, span_len2)

    pos_orig = pos_raw1[flat1]
    neg_orig = pos_raw1[~flat1]
    pos_mod = pos_raw2[flat2]
    neg_mod = pos_raw2[~flat2]

    if len(pos_orig) == 0:
        raise RuntimeError("original image token mask empty after resize")
    if len(neg_orig) == 0:
        # force one negative by taking the lowest-activation token region surrogate
        neg_orig = np.array([int(pos_raw1[np.argmin(flat1.astype(np.int32))])], dtype=np.int64)
        pos_orig = np.array([x for x in pos_orig if x != neg_orig[0]], dtype=np.int64)
        if len(pos_orig) == 0:
            pos_orig = np.array([int(pos_raw1[0])], dtype=np.int64)
            neg_orig = np.array([int(pos_raw1[-1])], dtype=np.int64)

    if orig_only:
        pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(neg_orig, dtype=torch.long, device=model.device)
    else:
        if len(pos_mod) == 0:
            pos_mod = np.array([int(pos_raw2[0])], dtype=np.int64)
        if len(neg_mod) == 0:
            neg_mod = np.array([int(pos_raw2[-1])], dtype=np.int64)
        pos_idx = torch.tensor(np.concatenate([pos_orig, pos_mod]), dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(np.concatenate([neg_orig, neg_mod]), dtype=torch.long, device=model.device)

    orig_pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
    mod_pos_base = pos_mod if len(pos_mod) > 0 else pos_raw2[: max(1, min(len(pos_raw2), len(pos_orig)))]
    mod_pos_idx = torch.tensor(mod_pos_base, dtype=torch.long, device=model.device)

    layer_indices = list(range(layer_start, layer_end + 1))
    per_layer = {str(layer): [] for layer in layer_indices}
    pooled_vectors: list[torch.Tensor] = []

    for t in step_indices:
        # during generation, the key length grows with every generated token.
        # we only keep the original prompt prefix so that vectors have identical length.
        for layer in layer_indices:
            full_attn = outputs.attentions[t][layer][0, :, -1, :].mean(dim=0).float()
            attn = full_attn[:input_len]
            pooled_vectors.append(attn)
            per_layer[str(layer)].append(attn_ratio(attn, pos_idx, neg_idx))

    pooled = torch.stack(pooled_vectors, dim=0).mean(dim=0)
    ratio_main = attn_ratio(pooled, pos_idx, neg_idx)
    ratio_cross = attn_ratio(pooled, orig_pos_idx, mod_pos_idx)

    rng = random.Random(seed)
    rand_scores = []
    orig_flat = flat1.astype(bool)
    for _ in range(n_random_masks):
        rand_flat = random_same_size_mask(orig_flat, rng)
        rand_pos = pos_raw1[rand_flat]
        rand_neg = pos_raw1[~rand_flat]
        if len(rand_pos) == 0 or len(rand_neg) == 0:
            continue
        rand_pos_idx = torch.tensor(rand_pos, dtype=torch.long, device=model.device)
        rand_neg_idx = torch.tensor(rand_neg, dtype=torch.long, device=model.device)
        rand_scores.append(attn_ratio(pooled, rand_pos_idx, rand_neg_idx))
    if not rand_scores:
        rand_scores = [0.0]

    return {
        "sample_id": str(meta["sample_id"]),
        "ratio_sample_mean": ratio_main,
        "ratio_cross_image": ratio_cross,
        "ratio_random_mean": float(np.mean(rand_scores)),
        "ratio_random_std": float(np.std(rand_scores)),
        "margin_vs_random": float(ratio_main - float(np.mean(rand_scores))),
        "gt_beats_random": bool(ratio_main > float(np.mean(rand_scores))),
        "num_changed_tokens_original": int(len(pos_orig)),
        "num_other_tokens_original": int(len(neg_orig)),
        "step_count": len(step_indices),
        "layers": layer_indices,
        "per_layer_step_ratio": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust attention ratio for remove-only SPD benchmark on Qwen2.5-VL")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--manifest_file", type=str, default=None, help="Use a custom manifest instead of data_root/split.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question_mode", type=str, default="remove", choices=list(PROMPTS.keys()))
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=27)
    parser.add_argument("--n_random_masks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--content_only", action="store_true")
    parser.add_argument("--orig_only", action="store_true")
    parser.add_argument("--strict_remove_only", action="store_true")
    parser.add_argument("--out_file", type=str, default="outputs/attention_ratio/remove_only_attention_ratio_v2.json")
    parser.add_argument("--debug_vis_dir", type=str, default=None)
    parser.add_argument("--use_slow_processor", action="store_true")
    args = parser.parse_args()

    if args.manifest_file is None and (args.data_root is None or args.split is None):
        raise ValueError("Either --manifest_file or both --data_root and --split are required")

    prompt = PROMPTS[args.question_mode]
    data_root = Path(args.data_root) if args.data_root else None
    split = args.split
    manifest_file = Path(args.manifest_file) if args.manifest_file else None
    rows = load_rows(data_root, split, manifest_file, args.limit)
    debug_vis_dir = Path(args.debug_vis_dir) if args.debug_vis_dir else None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=not args.use_slow_processor)

    results: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        try:
            meta = get_meta_from_row(row, data_root=data_root, split=split)
        except Exception as exc:
            skipped += 1
            print(f"[Skip] metadata load failed for row={row.get('sample_id', '?')}: {exc}")
            continue

        sample_id = str(meta.get("sample_id", row.get("sample_id", "?")))
        if args.strict_remove_only and not is_remove_only(meta):
            print(f"[Skip] non-remove-only sample: {sample_id}")
            skipped += 1
            continue

        try:
            out = analyze_sample(
                model=model,
                processor=processor,
                meta=meta,
                prompt=prompt,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                content_only=args.content_only,
                n_random_masks=args.n_random_masks,
                seed=args.seed + (hash(sample_id) % 10000),
                orig_only=args.orig_only,
                debug_vis_dir=debug_vis_dir,
            )
            if "source_split" in row:
                out["source_split"] = row["source_split"]
            results.append(out)
            print(
                f"[OK] {sample_id}: ratio={out['ratio_sample_mean']:.4f}, "
                f"rand={out['ratio_random_mean']:.4f}, cross={out['ratio_cross_image']:.4f}"
            )
        except Exception as exc:
            skipped += 1
            print(f"[Skip] {sample_id}: {exc}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model_name,
        "manifest_file": str(manifest_file) if manifest_file else None,
        "split": split,
        "question_mode": args.question_mode,
        "prompt": prompt,
        "content_only": args.content_only,
        "orig_only": args.orig_only,
        "strict_remove_only": args.strict_remove_only,
        "layer_range": [args.layer_start, args.layer_end],
        "num_rows": len(rows),
        "num_samples": len(results),
        "num_skipped": skipped,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
