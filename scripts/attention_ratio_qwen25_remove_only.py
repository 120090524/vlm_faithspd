from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROMPTS = {
    "different": "Are the two pictures different? Answer yes or no first, then list all concrete differences you can see.",
    "open": "Look at the two images carefully. First, state the total number of differences you see. Then explain each difference step by step. Be explicit and grounded in the images.",
    "remove": "One image may have an object removed. Compare the two images carefully and explain all removal differences step by step.",
}


def load_manifest(data_root: Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = data_root / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def find_vision_spans(input_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(input_ids):
        if input_ids[i] == start_id:
            j = i + 1
            while j < len(input_ids) and input_ids[j] != end_id:
                j += 1
            if j < len(input_ids):
                spans.append((i + 1, j))
            i = j + 1
        else:
            i += 1
    return spans


def get_merge_size(model) -> int:
    cfg = getattr(model, "config", None)
    vcfg = getattr(cfg, "vision_config", None)
    for name in ["spatial_merge_size", "merge_size"]:
        val = getattr(vcfg, name, None)
        if isinstance(val, int) and val > 0:
            return val
    return 2


def infer_token_grid(image_grid_thw_row: np.ndarray, span_len: int, image_size: tuple[int, int], merge_size: int) -> tuple[int, int]:
    t, gh, gw = [int(x) for x in image_grid_thw_row.tolist()]
    token_h = max(1, gh // merge_size)
    token_w = max(1, gw // merge_size)
    if max(1, t * token_h * token_w) == span_len:
        return token_h, token_w
    width, height = image_size
    aspect = max(width / max(height, 1), 1e-6)
    token_h = max(1, int(round(math.sqrt(span_len / aspect))))
    token_w = max(1, int(math.ceil(span_len / token_h)))
    while token_h * token_w < span_len:
        token_w += 1
    return token_h, token_w


def modifications(meta: dict[str, Any]) -> list[dict[str, Any]]:
    mods = meta.get("ground_truth", {}).get("modifications", [])
    return [m for m in mods if isinstance(m, dict)]


def is_remove_only(meta: dict[str, Any]) -> bool:
    mods = modifications(meta)
    if not mods:
        return False
    return all(str(m.get("type", "")).strip().lower() == "remove" for m in mods)


def has_bbox(mod: dict[str, Any]) -> bool:
    keys = {"bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    return keys.issubset(set(mod.keys()))


def bbox_mask(mods: list[dict[str, Any]], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for mod in mods:
        if not has_bbox(mod):
            continue
        x = int(round(float(mod["bbox_x"])))
        y = int(round(float(mod["bbox_y"])))
        w = int(round(float(mod["bbox_w"])))
        h = int(round(float(mod["bbox_h"])))
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
    m2 = np.zeros((img2.height, img2.width), dtype=np.uint8)

    # Fallback or enlargement with pixel diff.
    d = pixel_diff_mask(img1, img2)
    if m1.sum() == 0:
        m1 = d.copy()
    else:
        if d.shape != m1.shape:
            d = (np.asarray(Image.fromarray((d * 255).astype(np.uint8), mode="L").resize((img1.width, img1.height), Image.Resampling.BILINEAR)) > 127).astype(np.uint8)
        m1 = np.maximum(m1, d.astype(np.uint8))

    # For the modified image, we only use the same image-space region as a contrast region.
    if m1.shape != m2.shape:
        m2 = (np.asarray(Image.fromarray((m1 * 255).astype(np.uint8), mode="L").resize((img2.width, img2.height), Image.Resampling.BILINEAR)) > 127).astype(np.uint8)
    else:
        m2 = m1.copy()
    return m1.astype(np.uint8), m2.astype(np.uint8)


def resize_mask_to_token(mask: np.ndarray, token_h: int, token_w: int, target_len: int, threshold: float = 0.12) -> np.ndarray:
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    arr = np.asarray(pil.resize((token_w, token_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    flat = (arr.reshape(-1) >= threshold)
    if flat.size > target_len:
        flat = flat[:target_len]
    elif flat.size < target_len:
        flat = np.concatenate([flat, np.zeros(target_len - flat.size, dtype=bool)], axis=0)
    if mask.sum() > 0 and flat.sum() == 0:
        best = int(np.argmax(arr.reshape(-1)[:target_len]))
        flat[best] = True
    return flat


def random_same_size_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    k = int(mask.sum())
    if k <= 0:
        return out
    idx = list(range(mask.size))
    rng.shuffle(idx)
    out[np.array(idx[:k], dtype=np.int64)] = True
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
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"][0].tolist()
    input_len = len(input_ids)
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

    spans = find_vision_spans(input_ids, processor.tokenizer)
    if len(spans) < 2:
        raise RuntimeError(f"expected 2 image spans, got {len(spans)}")
    grids = inputs["image_grid_thw"].detach().cpu().numpy()
    merge_size = get_merge_size(model)

    # Build token-level mask for original image.
    (s1, e1), (s2, e2) = spans[:2]
    span_len1 = e1 - s1
    span_len2 = e2 - s2
    th1, tw1 = infer_token_grid(grids[0], span_len1, img1.size, merge_size)
    th2, tw2 = infer_token_grid(grids[1], span_len2, img2.size, merge_size)

    flat1 = resize_mask_to_token(mask1, th1, tw1, span_len1)
    flat2 = resize_mask_to_token(mask2, th2, tw2, span_len2)

    pos_orig = np.arange(s1, e1, dtype=np.int64)[flat1]
    neg_orig = np.arange(s1, e1, dtype=np.int64)[~flat1]
    pos_mod = np.arange(s2, e2, dtype=np.int64)[flat2]
    neg_mod = np.arange(s2, e2, dtype=np.int64)[~flat2]

    if len(pos_orig) == 0 or len(neg_orig) == 0:
        raise RuntimeError("original image token mask empty after resize")

    if orig_only:
        pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(neg_orig, dtype=torch.long, device=model.device)
    else:
        pos_idx = torch.tensor(np.concatenate([pos_orig, pos_mod]), dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(np.concatenate([neg_orig, neg_mod]), dtype=torch.long, device=model.device)

    orig_pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
    mod_pos_idx = torch.tensor(pos_mod if len(pos_mod) > 0 else pos_orig, dtype=torch.long, device=model.device)
    cross_neg_idx = mod_pos_idx

    layer_indices = list(range(layer_start, layer_end + 1))
    per_layer = {str(layer): [] for layer in layer_indices}
    pooled_vectors: list[torch.Tensor] = []
    pooled_cross: list[float] = []

    for t in step_indices:
        for layer in layer_indices:
            attn = outputs.attentions[t][layer][0, :, -1, :].mean(dim=0).float()
            pooled_vectors.append(attn)
            per_layer[str(layer)].append(attn_ratio(attn, pos_idx, neg_idx))
            pooled_cross.append(attn_ratio(attn, orig_pos_idx, cross_neg_idx))

    pooled = torch.stack(pooled_vectors, dim=0).mean(dim=0)
    ratio_main = attn_ratio(pooled, pos_idx, neg_idx)
    ratio_cross = attn_ratio(pooled, orig_pos_idx, cross_neg_idx)

    rng = random.Random(seed)
    rand_scores = []
    orig_flat = flat1.astype(bool)
    for _ in range(n_random_masks):
        rand_flat = random_same_size_mask(orig_flat, rng)
        rand_pos = np.arange(s1, e1, dtype=np.int64)[rand_flat]
        rand_neg = np.arange(s1, e1, dtype=np.int64)[~rand_flat]
        rand_pos_idx = torch.tensor(rand_pos, dtype=torch.long, device=model.device)
        rand_neg_idx = torch.tensor(rand_neg, dtype=torch.long, device=model.device)
        rand_scores.append(attn_ratio(pooled, rand_pos_idx, rand_neg_idx))

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
    parser = argparse.ArgumentParser(description="Attention ratio for remove-only SPD benchmark on Qwen2.5-VL")
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--split", type=str, default="remove_only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question_mode", type=str, default="remove", choices=list(PROMPTS.keys()))
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=27)
    parser.add_argument("--n_random_masks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--content_only", action="store_true")
    parser.add_argument("--orig_only", action="store_true", help="Only use original-image removed region as positive set")
    parser.add_argument("--strict_remove_only", action="store_true", help="Skip samples whose modifications are not all remove")
    parser.add_argument("--out_file", type=str, default="outputs/attention_ratio/remove_only_attention_ratio.json")
    parser.add_argument("--debug_vis_dir", type=str, default=None)
    args = parser.parse_args()

    prompt = PROMPTS[args.question_mode]
    data_root = Path(args.data_root)
    samples = load_manifest(data_root, args.split, args.limit)
    debug_vis_dir = Path(args.debug_vis_dir) if args.debug_vis_dir else None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    split_root = data_root / args.split
    results: list[dict[str, Any]] = []
    for row in samples:
        sample_id = str(row["sample_id"])
        meta_path = split_root / sample_id / "metadata.json"
        if not meta_path.exists():
            print(f"[Skip] metadata not found: {meta_path}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if args.strict_remove_only and not is_remove_only(meta):
            print(f"[Skip] non-remove-only sample: {sample_id}")
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
                seed=args.seed + int(sample_id) % 10000,
                orig_only=args.orig_only,
                debug_vis_dir=debug_vis_dir,
            )
            results.append(out)
            print(f"[OK] {sample_id}: ratio={out['ratio_sample_mean']:.4f}, rand={out['ratio_random_mean']:.4f}, cross={out['ratio_cross_image']:.4f}")
        except Exception as exc:
            print(f"[Skip] {sample_id}: {exc}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model_name,
        "split": args.split,
        "question_mode": args.question_mode,
        "prompt": prompt,
        "content_only": args.content_only,
        "orig_only": args.orig_only,
        "strict_remove_only": args.strict_remove_only,
        "layer_range": [args.layer_start, args.layer_end],
        "num_samples": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
