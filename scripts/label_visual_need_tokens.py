from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "and", "or", "but", "with", "without",
    "there", "here", "this", "that", "these", "those", "it", "they", "he", "she",
    "as", "by", "from", "than", "then", "so", "if", "not", "yes", "no",
}


def read_token_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_unique_tokens(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    response_text_map: dict[str, str] = {}
    prompt_map: dict[str, str] = {}
    for r in rows:
        sid = str(r["sample_id"])
        tidx = int(r["token_index"])
        grouped[sid].setdefault(tidx, {"token_index": tidx, "token_text": r["token_text"]})
        response_text_map[sid] = r.get("response_text", "")
        prompt_map[sid] = r.get("question_prompt", "")

    out = []
    for sid, token_map in grouped.items():
        toks = [token_map[k] for k in sorted(token_map.keys())]
        out.append({
            "sample_id": sid,
            "response_text": response_text_map.get(sid, ""),
            "question_prompt": prompt_map.get(sid, ""),
            "tokens": toks,
        })
    return out


def heuristic_label(token_text: str) -> tuple[str, str]:
    t = token_text.strip()
    low = t.lower().strip()
    if not any(ch.isalnum() for ch in low):
        return "U", "punctuation or formatting token"
    if low in STOPWORDS:
        return "T", "function word; usually does not require image evidence"
    if low.isdigit():
        return "V", "number likely encodes count or quantity difference"
    return "V", "content token likely tied to visual grounding"


def build_llm_prompt(sample: dict[str, Any]) -> str:
    lines = []
    lines.append("You are labeling answer tokens in a spot-the-difference task.")
    lines.append("Label each token with exactly one of:")
    lines.append("V = this token needs image evidence to decide the answer/difference.")
    lines.append("T = this token does not need image evidence; mostly functional/textual.")
    lines.append("U = punctuation, subword fragment, or genuinely uncertain.")
    lines.append("Important rules:")
    lines.append("1. Keep token_index unchanged.")
    lines.append("2. If contiguous token fragments form one semantic word, assign the same label to all fragments.")
    lines.append("3. Prefer V for nouns, attributes, counts, positions, colors, objects, and visual difference words.")
    lines.append("4. Prefer T for articles, auxiliaries, prepositions, conjunctions, and filler words.")
    lines.append("Return JSON only: {\"sample_id\": ..., \"labels\": [{\"token_index\": ..., \"label\": \"V/T/U\", \"reason\": ...}, ...]}")
    lines.append("")
    lines.append(f"sample_id: {sample['sample_id']}")
    lines.append(f"prompt: {sample['question_prompt']}")
    lines.append(f"response_text: {sample['response_text']}")
    lines.append("tokens:")
    for tok in sample["tokens"]:
        lines.append(f"- token_index={tok['token_index']} token_text={json.dumps(tok['token_text'], ensure_ascii=False)}")
    return "\n".join(lines)


def prepare_mode(rows: list[dict[str, Any]], out_path: Path, max_samples: int | None) -> None:
    grouped = group_unique_tokens(rows)
    if max_samples is not None:
        grouped = grouped[:max_samples]
    with out_path.open("w", encoding="utf-8") as f:
        for sample in grouped:
            item = {
                "sample_id": sample["sample_id"],
                "response_text": sample["response_text"],
                "tokens": sample["tokens"],
                "llm_prompt": build_llm_prompt(sample),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[ok] wrote label tasks to {out_path}")


def heuristic_mode(rows: list[dict[str, Any]], out_path: Path, max_samples: int | None) -> None:
    grouped = group_unique_tokens(rows)
    if max_samples is not None:
        grouped = grouped[:max_samples]
    with out_path.open("w", encoding="utf-8") as f:
        for sample in grouped:
            item = {
                "sample_id": sample["sample_id"],
                "labels": [],
            }
            for tok in sample["tokens"]:
                label, reason = heuristic_label(tok["token_text"])
                item["labels"].append({
                    "token_index": tok["token_index"],
                    "label": label,
                    "reason": reason,
                })
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[ok] wrote heuristic labels to {out_path}")


def read_label_file(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    mapping: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = str(item["sample_id"])
            for lab in item.get("labels", []):
                key = (sid, int(lab["token_index"]))
                mapping[key] = {
                    "label": lab.get("label", "U"),
                    "reason": lab.get("reason", ""),
                }
    return mapping


def merge_mode(token_rows: list[dict[str, Any]], label_path: Path, out_path: Path) -> None:
    mapping = read_label_file(label_path)
    merged = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in token_rows:
            key = (str(row["sample_id"]), int(row["token_index"]))
            lab = mapping.get(key, {"label": "U", "reason": "missing label"})
            row = dict(row)
            row["token_label"] = lab["label"]
            row["token_label_reason"] = lab["reason"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if key in mapping:
                merged += 1
    print(f"[ok] merged labels into {out_path}; matched_token_rows={merged}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare / heuristically label / merge visual-textual token labels")
    parser.add_argument("--mode", choices=["prepare", "heuristic", "merge"], required=True)
    parser.add_argument("--token_jsonl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--label_jsonl", type=str, default=None, help="Required in merge mode")
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()

    token_rows = read_token_rows(Path(args.token_jsonl))
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare":
        prepare_mode(token_rows, out_path, args.max_samples)
    elif args.mode == "heuristic":
        heuristic_mode(token_rows, out_path, args.max_samples)
    else:
        if args.label_jsonl is None:
            raise ValueError("--label_jsonl is required for merge mode")
        merge_mode(token_rows, Path(args.label_jsonl), out_path)


if __name__ == "__main__":
    main()
