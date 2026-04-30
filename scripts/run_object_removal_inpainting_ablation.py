from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "data" / "data_pipeline"))

from config import COCO_ANN_FILE, COCO_IMG_DIR  # noqa: E402



def save_grid(images: list[np.ndarray], labels: list[str], save_path: Path, target_h: int = 360) -> None:
    rendered = []
    for img, label in zip(images, labels):
        if img.shape[0] != target_h:
            new_w = max(1, int(img.shape[1] * target_h / img.shape[0]))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        canvas = img.copy()
        cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 420), 32), (255, 255, 255), -1)
        cv2.putText(canvas, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        rendered.append(canvas)
    merged = np.concatenate(rendered, axis=1)
    cv2.imwrite(str(save_path), merged)


def parse_image_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.image_ids:
        ids.extend(args.image_ids)
    if args.image_ids_file:
        for line in Path(args.image_ids_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    # Normalize COCO id formatting, dedupe while preserving order.
    out, seen = [], set()
    for x in ids:
        norm = str(int(x))
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def backend_kwargs(backend: str, args: argparse.Namespace) -> dict:
    backend = backend.lower()
    common = {
        "preserve_unmasked": not args.no_preserve_unmasked,
        "feather": args.feather,
    }
    if backend in {"sd15", "sd-1.5", "stable-diffusion", "sdxl", "sdxl-inpaint"}:
        common.update(
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            padding_mask_crop=args.padding_mask_crop,
            cpu_offload=not args.no_cpu_offload,
        )
    elif backend in {"flux", "flux-fill", "flux1-fill"}:
        common.update(
            num_inference_steps=args.flux_steps if args.flux_steps is not None else args.steps,
            guidance_scale=args.flux_guidance,
            cpu_offload=not args.no_cpu_offload,
        )
    return common


# def build_generator(backend: str, args: argparse.Namespace):
#     if args.fallback_only:
#         api_client.initialize_clients = lambda: (None, None)
#     generator_mod = importlib.import_module("generator")
#     SpotDifferenceGenerator = generator_mod.SpotDifferenceGenerator
#     gen = SpotDifferenceGenerator(
#         COCO_ANN_FILE,
#         COCO_IMG_DIR,
#         inpaint_backend=backend,
#         inpaint_backend_kwargs=backend_kwargs(backend, args),
#     )
#     if args.fallback_only:
#         gen._ask_llm_remove_object = lambda annotated_img, objects_info: None
#     return gen
def install_fake_api_client_for_fallback() -> None:
    """
    In --fallback_only mode, we do not need Gemini/OpenAI at all.
    The original generator.py imports api_client at the top level.
    This fake module prevents api_client.py from initializing OpenAI/httpx.
    """
    import types
    import sys

    fake = types.ModuleType("api_client")
    fake.client = None
    fake.gemini_client = None
    sys.modules["api_client"] = fake


def build_generator(backend: str, args: argparse.Namespace):
    if args.fallback_only:
        install_fake_api_client_for_fallback()
    else:
        import api_client  # real API mode only

    generator_mod = importlib.import_module("generator")
    SpotDifferenceGenerator = generator_mod.SpotDifferenceGenerator

    gen = SpotDifferenceGenerator(
        COCO_ANN_FILE,
        COCO_IMG_DIR,
        inpaint_backend=backend,
        inpaint_backend_kwargs=backend_kwargs(backend, args),
    )

    if args.fallback_only:
        gen._ask_llm_remove_object = lambda annotated_img, objects_info: None

    return gen


def main() -> None:
    ap = argparse.ArgumentParser(description="Run object-removal inpainting ablation with multiple backends")
    ap.add_argument("--image_ids", nargs="*", default=[])
    ap.add_argument("--image_ids_file", type=str, default=None)
    ap.add_argument("--backends", nargs="+", default=["lama", "opencv-telea", "sd15", "sdxl"])
    ap.add_argument("--out_dir", type=str, default="outputs/inpaint_ablation")
    ap.add_argument("--fallback_only", action="store_true", help="skip online object selector and use median-area fallback")
    ap.add_argument("--force_object_index", type=int, default=None, help="force a COCO annotation index for every backend")
    ap.add_argument("--seed", type=int, default=0)

    # Diffusion settings.
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--strength", type=float, default=0.98)
    ap.add_argument("--padding_mask_crop", type=int, default=32)
    ap.add_argument("--flux_steps", type=int, default=None)
    ap.add_argument("--flux_guidance", type=float, default=30.0)
    ap.add_argument("--no_cpu_offload", action="store_true")

    # Preservation settings.
    ap.add_argument("--no_preserve_unmasked", action="store_true")
    ap.add_argument("--feather", type=int, default=5)

    args = ap.parse_args()
    image_ids = parse_image_ids(args)
    if not image_ids:
        raise SystemExit("Provide --image_ids or --image_ids_file")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        sample_dir = out_root / f"{int(image_id):012d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        first_gen = build_generator(args.backends[0], args)
        original_img, img_info, anns = first_gen._get_image_and_annotations(image_id)
        cv2.imwrite(str(sample_dir / f"{int(image_id):012d}_original.jpg"), original_img)

        fixed_idx = args.force_object_index
        logs = []
        images = [original_img]
        labels = ["original"]

        for b_i, backend in enumerate(args.backends):
            gen = first_gen if b_i == 0 else build_generator(backend, args)
            result = gen._remove_object(
                image_id,
                force_object_index=fixed_idx,
                seed=args.seed,
            )
            # Reuse the first backend's selected object for all subsequent backends.
            if fixed_idx is None and "processed_index" in result:
                fixed_idx = int(result["processed_index"])

            out_img = result["image"]
            safe_backend = backend.replace("/", "_")
            cv2.imwrite(str(sample_dir / f"removed_{safe_backend}.jpg"), out_img)
            images.append(out_img)
            labels.append(backend)

            log = dict(result.get("log", {}))
            log.update(
                image_id=int(image_id),
                file_name=img_info.get("file_name"),
                backend=backend,
                backend_kwargs=backend_kwargs(backend, args),
                seed=args.seed,
                forced_or_reused_object_index=fixed_idx,
            )
            logs.append(log)

        save_grid(images, labels, sample_dir / "compare_grid.jpg")
        (sample_dir / "logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {image_id}: saved {len(args.backends)} backend results to {sample_dir}")


if __name__ == "__main__":
    main()
