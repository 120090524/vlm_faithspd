from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


PROMPT_JSON = """You are solving a spot-the-difference task.

Look at the two images carefully and output ONLY valid JSON.
Do not output markdown fences.
The JSON schema must be:

{
  "predicted_num": <integer>,
  "structured_output": [
    {"type": "color" | "remove" | "position", "category": "<object category>"},
    ...
  ]
}

Rules:
- "predicted_num" must equal the number of entries in "structured_output".
- Each difference must be one of: color / remove / position.
- "category" should be a short object noun, like "cat", "bus", "remote", "person".
- Output JSON only.
"""

PROMPT_SAME = """Are the two pictures the same?
Answer yes or no first, then briefly explain why.
"""

PROMPT_DIFFERENT = """Are the two pictures different?
Answer yes or no first, then list all concrete differences you can see.
"""

PROMPT_DRF = """Look at the two images carefully.

First, state the total number of differences you see.
Then explain each difference step by step.

Be explicit and grounded in the images.
"""


def call_model(model, processor, img1_path: str, img2_path: str, prompt: str, max_new_tokens: int = 256) -> str:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


def try_parse_json(text: str) -> dict[str, Any]:
    text = text.strip()

    # 优先直接整段解析
    try:
        data = json.loads(text)
        return normalize_parsed(data, ok=True)
    except Exception:
        pass

    # 再尝试提取第一个 JSON 对象
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return normalize_parsed(data, ok=True)
        except Exception:
            pass

    return {
        "has_structured_output": False,
        "predicted_num": None,
        "structured_output": [],
        "raw_text": text,
    }


def normalize_parsed(data: dict[str, Any], ok: bool) -> dict[str, Any]:
    pred_num = data.get("predicted_num", None)
    structured = data.get("structured_output", [])
    if not isinstance(structured, list):
        structured = []

    cleaned = []
    for item in structured:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "type": str(item.get("type", "")).strip().lower(),
            "category": str(item.get("category", "")).strip().lower(),
        })

    if pred_num is None:
        pred_num = len(cleaned)

    return {
        "has_structured_output": ok,
        "predicted_num": pred_num,
        "structured_output": cleaned,
    }


def load_manifest(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--split", type=str, default="multi_diff")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--out_dir", type=str, default="outputs/baseline")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    manifest_path = data_root / f"{args.split}.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    samples = load_manifest(manifest_path, args.limit)
    print(f"Loaded {len(samples)} samples")

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    multi_diff_responses = []
    consistency_responses = []
    faithfulness_responses = []

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        sample_dir = data_root / args.split / sample_id
        meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))

        img1 = meta["original_path"]
        img2 = meta["modified_path"]
        gt = meta["ground_truth"]

        print(f"[{idx}/{len(samples)}] sample_id={sample_id}")

        # 1) 结构化差异输出
        raw_json_response = call_model(model, processor, img1, img2, PROMPT_JSON)
        parsed = try_parse_json(raw_json_response)

        multi_diff_responses.append({
            "sample_id": sample_id,
            "ground_truth": gt,
            "raw_response": raw_json_response,
            "parsed_response": parsed,
        })

        # 2) CR 所需的两种问法
        response_same = call_model(model, processor, img1, img2, PROMPT_SAME)
        response_different = call_model(model, processor, img1, img2, PROMPT_DIFFERENT)

        consistency_responses.append({
            "sample_id": sample_id,
            "ground_truth": gt,
            "response_same": response_same,
            "response_different": response_different,
        })

        # 3) DRF 所需的“数量 + 解释”响应
        response_drf = call_model(model, processor, img1, img2, PROMPT_DRF)

        faithfulness_responses.append({
            "sample_id": sample_id,
            "ground_truth": gt,
            "model_response": response_drf,
        })

    model_short_name = Path(args.model_name).name.replace("/", "_")

    with (out_dir / f"{model_short_name}_multi_diff_responses.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_short_name, "responses": multi_diff_responses}, f, ensure_ascii=False, indent=2)

    with (out_dir / f"{model_short_name}_consistency_responses.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_short_name, "responses": consistency_responses}, f, ensure_ascii=False, indent=2)

    with (out_dir / f"{model_short_name}_faithfulness_responses.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_short_name, "responses": faithfulness_responses}, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()