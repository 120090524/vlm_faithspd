from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Reuse your existing attention-ratio utilities.
import attention_ratio_qwen25_remove_only_v2 as base


def load_analysis_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


def mean_over_indices(attn: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return 0.0
    return float(attn.index_select(0, idx).mean().item())


def build_random_masks_for_sample(
    pos_raw1: np.ndarray,
    flat1: np.ndarray,
    device: torch.device,
    n_random_masks: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    masks: list[tuple[torch.Tensor, torch.Tensor]] = []
    orig_flat = flat1.astype(bool)
    for _ in range(n_random_masks):
        rand_flat = base.random_same_size_mask(orig_flat, rng)
        rand_pos = pos_raw1[rand_flat]
        rand_neg = pos_raw1[~rand_flat]
        if len(rand_pos) == 0 or len(rand_neg) == 0:
            continue
        masks.append(
            (
                torch.tensor(rand_pos, dtype=torch.long, device=device),
                torch.tensor(rand_neg, dtype=torch.long, device=device),
            )
        )
    return masks


def analyze_one_sample(
    model,
    processor,
    meta: dict[str, Any],
    prompt: str,
    layer_start: int,
    layer_end: int,
    content_only: bool,
    orig_only: bool,
    n_random_masks: int,
    seed: int,
    sample_label: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    img1 = Image.open(meta["original_path"]).convert("RGB")
    img2 = Image.open(meta["modified_path"]).convert("RGB")
    mask1, mask2 = base.build_remove_masks(meta, img1, img2)

    messages = base.messages_for_two_images(img1, img2, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    input_ids = inputs["input_ids"][0].tolist()
    grids = inputs["image_grid_thw"].detach().cpu().numpy()
    expected_counts = base.expected_token_counts(grids, base.get_merge_size(model))
    image_positions = base.find_image_token_positions(input_ids, expected_counts, model, processor)
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
    response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
    step_indices = base.content_token_indices(gen_tokens) if content_only else list(range(len(gen_tokens)))
    if not step_indices:
        raise RuntimeError("no generated tokens")

    pos_raw1 = image_positions[0]
    pos_raw2 = image_positions[1]
    span_len1 = len(pos_raw1)
    span_len2 = len(pos_raw2)
    th1, tw1 = base.infer_token_grid(grids[0], span_len1, img1.size, base.get_merge_size(model))
    th2, tw2 = base.infer_token_grid(grids[1], span_len2, img2.size, base.get_merge_size(model))

    flat1 = base.resize_mask_to_token(mask1, th1, tw1, span_len1)
    flat2 = base.resize_mask_to_token(mask2, th2, tw2, span_len2)

    pos_orig = pos_raw1[flat1]
    neg_orig = pos_raw1[~flat1]
    pos_mod = pos_raw2[flat2]
    neg_mod = pos_raw2[~flat2]

    if len(pos_orig) == 0:
        raise RuntimeError("empty original positive region after mask resize")
    if len(neg_orig) == 0:
        raise RuntimeError("empty original background region after mask resize")
    if len(pos_mod) == 0:
        pos_mod = np.array([int(pos_raw2[0])], dtype=np.int64)
    if len(neg_mod) == 0:
        neg_mod = np.array([int(pos_raw2[-1])], dtype=np.int64)

    if orig_only:
        pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(neg_orig, dtype=torch.long, device=model.device)
    else:
        pos_idx = torch.tensor(np.concatenate([pos_orig, pos_mod]), dtype=torch.long, device=model.device)
        neg_idx = torch.tensor(np.concatenate([neg_orig, neg_mod]), dtype=torch.long, device=model.device)

    orig_pos_idx = torch.tensor(pos_orig, dtype=torch.long, device=model.device)
    orig_neg_idx = torch.tensor(neg_orig, dtype=torch.long, device=model.device)
    mod_pos_idx = torch.tensor(pos_mod, dtype=torch.long, device=model.device)
    mod_neg_idx = torch.tensor(neg_mod, dtype=torch.long, device=model.device)

    random_masks = build_random_masks_for_sample(
        pos_raw1=pos_raw1,
        flat1=flat1,
        device=model.device,
        n_random_masks=n_random_masks,
        seed=seed,
    )

    layer_indices = list(range(layer_start, layer_end + 1))
    rows: list[dict[str, Any]] = []

    for t in step_indices:
        token_text = gen_tokens[t]
        for layer in layer_indices:
            full_attn = outputs.attentions[t][layer][0, :, -1, :].mean(dim=0).float()
            attn = full_attn[:input_len]

            gt_mean = mean_over_indices(attn, pos_idx)
            bg_mean = mean_over_indices(attn, neg_idx)
            ratio_gt = gt_mean / max(bg_mean, 1e-8)
            margin_gt = gt_mean - bg_mean

            orig_gt_mean = mean_over_indices(attn, orig_pos_idx)
            orig_bg_mean = mean_over_indices(attn, orig_neg_idx)
            mod_gt_mean = mean_over_indices(attn, mod_pos_idx)
            mod_bg_mean = mean_over_indices(attn, mod_neg_idx)
            cross_ratio = orig_gt_mean / max(mod_gt_mean, 1e-8)

            rand_means = []
            rand_ratios = []
            for rand_pos_idx, rand_neg_idx in random_masks:
                rp = mean_over_indices(attn, rand_pos_idx)
                rn = mean_over_indices(attn, rand_neg_idx)
                rand_means.append(rp)
                rand_ratios.append(rp / max(rn, 1e-8))

            random_gt_mean = float(np.mean(rand_means)) if rand_means else 0.0
            random_ratio_mean = float(np.mean(rand_ratios)) if rand_ratios else 0.0

            row = {
                "sample_id": str(meta["sample_id"]),
                "token_index": int(t),
                "token_text": token_text,
                "response_text": response_text,
                "question_prompt": prompt,
                "layer": int(layer),
                "gt_attn_mean": gt_mean,
                "bg_attn_mean": bg_mean,
                "ratio_gt": ratio_gt,
                "margin_gt": margin_gt,
                "orig_gt_mean": orig_gt_mean,
                "orig_bg_mean": orig_bg_mean,
                "mod_gt_mean": mod_gt_mean,
                "mod_bg_mean": mod_bg_mean,
                "cross_ratio_orig_to_mod": cross_ratio,
                "random_gt_mean": random_gt_mean,
                "random_ratio_mean": random_ratio_mean,
                "margin_vs_random": gt_mean - random_gt_mean,
                "num_changed_tokens_original": int(len(pos_orig)),
                "num_other_tokens_original": int(len(neg_orig)),
                "num_changed_tokens_modified": int(len(pos_mod)),
                "num_other_tokens_modified": int(len(neg_mod)),
            }

            if sample_label is not None:
                for key in [
                    "is_faithful",
                    "num_recall",
                    "type_f1",
                    "category_f1",
                    "predicted_num",
                ]:
                    if key in sample_label:
                        row[key] = sample_label[key]
            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export token × layer × region attention for SPD remove-only experiments")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--manifest_file", type=str, default=None)
    parser.add_argument("--analysis_jsonl", type=str, default=None, help="Optional build_analysis_jsonl.py output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_ids", type=str, default=None, help="Comma separated sample ids")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--question_mode", type=str, default="remove", choices=list(base.PROMPTS.keys()))
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=27)
    parser.add_argument("--n_random_masks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--content_only", action="store_true")
    parser.add_argument("--orig_only", action="store_true")
    parser.add_argument("--strict_remove_only", action="store_true")
    parser.add_argument("--out_jsonl", type=str, default="outputs/token_region_attention/token_region_attention.jsonl")
    parser.add_argument("--use_slow_processor", action="store_true")
    args = parser.parse_args()

    if args.manifest_file is None and (args.data_root is None or args.split is None):
        raise ValueError("Either --manifest_file or both --data_root and --split are required")

    prompt = base.PROMPTS[args.question_mode]
    data_root = Path(args.data_root) if args.data_root else None
    split = args.split
    manifest_file = Path(args.manifest_file) if args.manifest_file else None
    rows = base.load_rows(data_root, split, manifest_file, args.limit)
    analysis_labels = load_analysis_labels(Path(args.analysis_jsonl)) if args.analysis_jsonl else {}

    selected_ids: set[str] | None = None
    if args.sample_ids:
        selected_ids = {x.strip() for x in args.sample_ids.split(",") if x.strip()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=not args.use_slow_processor)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            try:
                meta = base.get_meta_from_row(row, data_root=data_root, split=split)
            except Exception as exc:
                skipped += 1
                print(f"[skip] metadata load failed for {row.get('sample_id', '?')}: {exc}")
                continue

            sample_id = str(meta.get("sample_id", row.get("sample_id", "?")))
            if selected_ids is not None and sample_id not in selected_ids:
                continue
            if args.strict_remove_only and not base.is_remove_only(meta):
                skipped += 1
                print(f"[skip] non-remove-only sample: {sample_id}")
                continue

            try:
                sample_rows = analyze_one_sample(
                    model=model,
                    processor=processor,
                    meta=meta,
                    prompt=prompt,
                    layer_start=args.layer_start,
                    layer_end=args.layer_end,
                    content_only=args.content_only,
                    orig_only=args.orig_only,
                    n_random_masks=args.n_random_masks,
                    seed=args.seed + (hash(sample_id) % 10000),
                    sample_label=analysis_labels.get(sample_id),
                )
                for r in sample_rows:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                exported += 1
                print(f"[ok] exported sample {sample_id}: {len(sample_rows)} token-layer rows")
            except Exception as exc:
                skipped += 1
                print(f"[skip] {sample_id}: {exc}")

    print(f"[done] exported_samples={exported}, skipped={skipped}, out={out_path}")


if __name__ == "__main__":
    main()
