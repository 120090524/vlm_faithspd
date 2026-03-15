from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 强制把 repo 根目录加入 Python 搜索路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from analysis.tam import TAM


def tam_demo(image_path: str, prompt_text: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    special_ids = {
        "img_id": [151652, 151653],
        "prompt_id": [151653, [151645, 198, 151644, 77091]],
        "answer_id": [[198, 151644, 77091, 198], -1],
    }

    vision_shape = (
        inputs["image_grid_thw"][0, 1] // 2,
        inputs["image_grid_thw"][0, 2] // 2,
    )
    vis_inputs = image_inputs

    raw_map_records = []
    for i in range(len(logits)):
        TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(save_dir, f"{i}.jpg"),
            i,
            raw_map_records,
            False,
        )

    print(f"[OK] TAM visualizations saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--question", type=str, default="Are the two pictures the same?")
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    tam_demo(args.image_path, args.question, args.save_dir)


if __name__ == "__main__":
    main()