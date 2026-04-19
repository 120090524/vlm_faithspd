from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

LABEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token_index": {"type": "integer", "minimum": 0},
                    "label": {"type": "string", "enum": ["V", "T", "U"]},
                    "reason": {"type": "string"},
                },
                "required": ["token_index", "label", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sample_id", "labels"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are annotating answer tokens for a multimodal difference-finding task.
For each token in the answer, assign exactly one label:
- V: this token needs image information to be justified
- T: this token does not need image information
- U: punctuation, formatting token, broken subword, or genuinely uncertain

Important rules:
1. Label based on WHETHER THE TOKEN NEEDS IMAGE EVIDENCE, not whether it is a noun.
2. Words referring to visual entities, attributes, counts, locations, colors, removed objects, and differences are usually V.
3. Function words, punctuation, auxiliaries, articles, many discourse connectives are usually T or U.
4. Be conservative: when a token is only meaningful because of the image difference, prefer V.
5. Return labels for EVERY token_index exactly once.
6. Do not omit tokens.
"""


def build_user_prompt(task: Dict[str, Any]) -> str:
    response_text = task.get("response_text", "")
    tokens = task.get("tokens", [])
    token_lines = []
    for t in tokens:
        idx = t["token_index"] if isinstance(t, dict) else t[0]
        tok = t["token_text"] if isinstance(t, dict) else t[1]
        token_lines.append(f"{idx}\t{tok}")

    return (
        f"sample_id: {task['sample_id']}\n"
        f"response_text: {response_text}\n\n"
        "tokens (tab-separated: token_index, token_text):\n"
        + "\n".join(token_lines)
        + "\n\nReturn JSON only."
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = str(row.get("sample_id", ""))
                if sid:
                    done.add(sid)
            except Exception:
                continue
    return done


def validate_label_count(task: Dict[str, Any], result: Dict[str, Any]) -> None:
    expected = len(task.get("tokens", []))
    labels = result.get("labels", [])
    if len(labels) != expected:
        raise ValueError(f"sample {task['sample_id']}: expected {expected} labels, got {len(labels)}")

    seen = set()
    for item in labels:
        idx = item["token_index"]
        if idx in seen:
            raise ValueError(f"sample {task['sample_id']}: duplicate token_index {idx}")
        seen.add(idx)
        if item["label"] not in {"V", "T", "U"}:
            raise ValueError(f"sample {task['sample_id']}: invalid label {item['label']}")

    expected_indices = {t["token_index"] if isinstance(t, dict) else t[0] for t in task.get("tokens", [])}
    if seen != expected_indices:
        raise ValueError(
            f"sample {task['sample_id']}: token indices mismatch. expected={sorted(expected_indices)} got={sorted(seen)}"
        )


def call_openai(client: OpenAI, model: str, task: Dict[str, Any], store: bool) -> Dict[str, Any]:
    prompt = build_user_prompt(task)
    resp = client.responses.create(
        model=model,
        store=store,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "token_visual_need_labels",
                "strict": True,
                "schema": LABEL_SCHEMA,
            }
        },
    )

    raw = resp.output_text
    data = json.loads(raw)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Label token visual-need with OpenAI Responses API.")
    parser.add_argument("--tasks_jsonl", required=True, help="Path to token_label_tasks.jsonl")
    parser.add_argument("--out_jsonl", required=True, help="Where to save token_labels_openai.jsonl")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4"), help="OpenAI model name")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--store", action="store_true", help="Store responses on OpenAI servers (default: false)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    tasks_path = Path(args.tasks_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(tasks_path)
    done = already_done(out_path)
    client = OpenAI(api_key=api_key)

    count = 0
    for task in tasks:
        sample_id = str(task["sample_id"])
        if sample_id in done:
            continue
        if args.max_samples is not None and count >= args.max_samples:
            break

        last_err: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                result = call_openai(client, args.model, task, store=args.store)
                validate_label_count(task, result)
                append_jsonl(out_path, result)
                print(f"[OK] sample_id={sample_id}")
                count += 1
                time.sleep(args.sleep)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[WARN] sample_id={sample_id} attempt={attempt}/{args.retries} error={e}", file=sys.stderr)
                time.sleep(min(2.0 * attempt, 5.0))

        if last_err is not None:
            print(f"[FAIL] sample_id={sample_id}: {last_err}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
