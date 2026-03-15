from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_first_sample(data_root: Path, split: str) -> dict:
    split_dir = data_root / split
    sample_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample dirs found in {split_dir}")
    meta_path = sample_dirs[0] / "metadata.json"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def ask_model(model, processor, img1_path: str, img2_path: str, question: str, max_new_tokens: int = 128) -> str:
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": question},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="work/spd_local")
    parser.add_argument("--split", type=str, default="multi_diff")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    sample = load_first_sample(Path(args.data_root), args.split)
    print(f"Sample: {sample['sample_id']}")

    answer = ask_model(
        model,
        processor,
        sample["original_path"],
        sample["modified_path"],
        "Are the two pictures different? Briefly explain.",
    )
    print("\n=== MODEL OUTPUT ===")
    print(answer)


if __name__ == "__main__":
    main()