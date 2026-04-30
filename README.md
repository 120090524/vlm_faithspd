# CSE559: Spot-the-Difference Faithfulness Benchmark

This repository is the CSE559 project workspace for studying **faithfulness in vision-language models (VLMs)** on spot-the-difference tasks. The project combines three directions:

1. **Benchmark reproduction**: export and evaluate SPD-Faith-Bench samples with Qwen2.5-VL.
2. **Mechanistic analysis**: measure whether generated answers attend to the changed visual regions through an attention-ratio metric.
3. **Data-generation improvement**: replace the original LaMa-only object-removal pipeline with a pluggable inpainting backend, including SD1.5, SDXL, OpenCV, and optional FLUX Fill.

The working branch used for the CSE559 experiments is:

```bash
git checkout remove_attention_5
```

---

## 1. Project Overview

The benchmark asks a model to compare an original image and a modified image, then identify concrete differences such as:

- `color`: an object's color changes.
- `remove`: an object disappears.
- `position`: an object moves.

The core research question is not only whether a VLM gives the correct answer, but whether its answer is **visually grounded** in the actual changed region. The repository therefore includes both evaluation scripts and attention-based diagnostic scripts.

---

## 2. Repository Structure

```text
vlm_faithspd/
├── analysis/                              # Additional analysis utilities
├── data/
│   ├── data_pipeline/
│   │   ├── api_client.py                  # Gemini/OpenAI client setup for data generation
│   │   ├── config.py                      # COCO paths and output paths
│   │   ├── generator.py                   # Spot-difference data generator
│   │   ├── inpainting_backends.py         # Pluggable inpainting backends
│   │   └── main.py                        # COCO-based data generation entry point
│   └── examples/                          # Example SPD-Faith images, if exported locally
├── eval/
│   ├── eval_multi_diff_metrics.py         # DQR, TF1, CF1 evaluation
│   ├── eval_cot_faithfulness.py           # CoT faithfulness / DRF-style evaluation
│   ├── eval_consistency_rate.py           # Consistency-rate evaluation
│   └── eval_cosine.py                     # Hidden-state / cosine analysis
├── fig/                                   # Paper/report figures
├── outputs/                               # Local experiment outputs, usually gitignored
├── scripts/
│   ├── export_spd_dataset.py              # Export Hugging Face SPD-Faith-Bench locally
│   ├── run_baseline_qwen25vl.py           # Qwen2.5-VL baseline prediction script
│   ├── build_remove_manifest_from_splits.py
│   ├── filter_remove_only.py
│   ├── attention_ratio_qwen25.py
│   ├── attention_ratio_qwen25_remove_only.py
│   ├── attention_ratio_qwen25_remove_only_v2.py
│   ├── analyze_attention_ratio_results.py
│   ├── plot_attention_ratio.py
│   ├── reproduce_object_removal.py
│   └── run_object_removal_inpainting_ablation.py
├── utils/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 3. Environment Setup

### 3.1 Create Python environment

```bash
conda create -n spd-faith python=3.10 -y
conda activate spd-faith
```

Install the repository requirements:

```bash
pip install -r requirements.txt
pip install qwen-vl-utils
```

Recommended additional packages for the CSE559 experiments:

```bash
pip install -U datasets pycocotools opencv-python pillow matplotlib pandas scipy tqdm
pip install -U simple-lama-inpainting
pip install -U diffusers transformers accelerate safetensors huggingface_hub sentencepiece protobuf certifi
```

For CUDA machines, install a CUDA-compatible PyTorch build, for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Adjust the CUDA wheel according to your local CUDA version.

### 3.2 Hugging Face login and model downloads

The benchmark export and diffusion inpainting models use Hugging Face.

```bash
hf auth login
hf auth whoami
```

Download the recommended SDXL inpainting model:

```bash
hf download diffusers/stable-diffusion-xl-1.0-inpainting-0.1
```

Optional SD1.5 inpainting model:

```bash
hf download stable-diffusion-v1-5/stable-diffusion-inpainting
```

Optional FLUX Fill model, if you want to test the `flux-fill` backend:

```bash
hf download black-forest-labs/FLUX.1-Fill-dev
```

FLUX Fill may require accepting model access terms on Hugging Face before download.

### 3.3 Recommended cache location

To avoid filling the C drive on Windows:

```bash
export HF_HOME=/e/huggingface_cache
export HF_HUB_CACHE=/e/huggingface_cache/hub
```

PowerShell equivalent:

```powershell
$env:HF_HOME="E:\huggingface_cache"
$env:HF_HUB_CACHE="E:\huggingface_cache\hub"
```

### 3.4 Windows SSL note

If Hugging Face or OpenAI/Gemini clients fail with `SSL_CERT_FILE` errors, clear broken certificate variables:

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
```

If SSL still fails:

```bash
pip install -U certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

---

## 4. Dataset Setup

There are two ways to obtain data.

### 4.1 Export the released dataset

The easiest way to reproduce benchmark experiments is to export samples from Hugging Face:

```bash
python scripts/export_spd_dataset.py \
  --splits easy medium hard multi_diff \
  --out_root work/spd_local
```

For a quick smoke test:

```bash
python scripts/export_spd_dataset.py \
  --splits multi_diff \
  --out_root work/spd_local \
  --limit 20
```

This creates:

```text
work/spd_local/
├── easy.jsonl
├── medium.jsonl
├── hard.jsonl
├── multi_diff.jsonl
└── multi_diff/<sample_id>/
    ├── <sample_id>_original.jpg
    ├── <sample_id>_modified_final.jpg
    ├── merged.jpg
    └── metadata.json
```

### 4.2 Generate new COCO-based spot-difference data

For custom data generation, configure COCO paths in:

```text
data/data_pipeline/config.py
```

Typical required paths are:

```text
COCO_ANN_FILE = path/to/instances_train2017.json
COCO_IMG_DIR  = path/to/train2017
```

Then run the data-generation entry point:

```bash
python data/data_pipeline/main.py
```

The generator supports `remove`, `color`, and `position` modifications. For object removal, the CSE559 branch now supports multiple inpainting backends through `data/data_pipeline/inpainting_backends.py`.

---

## 5. Reproduce Baseline VLM Evaluation

### 5.1 Run Qwen2.5-VL baseline

```bash
python scripts/run_baseline_qwen25vl.py \
  --data_root work/spd_local \
  --split multi_diff \
  --limit 50 \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --out_dir outputs/baseline_qwen25
```

This produces three response files:

```text
outputs/baseline_qwen25/
├── Qwen2.5-VL-7B-Instruct_multi_diff_responses.json
├── Qwen2.5-VL-7B-Instruct_consistency_responses.json
└── Qwen2.5-VL-7B-Instruct_faithfulness_responses.json
```

### 5.2 Evaluate difference-detection metrics

```bash
python eval/eval_multi_diff_metrics.py \
  --input_dir outputs/baseline_qwen25 \
  --output_dir outputs/baseline_qwen25_metrics
```

Main metrics:

- **DQR**: Difference Quantity Recall. Measures whether the predicted number of differences matches the ground truth count.
- **TF1**: Type-Level F1 over `color`, `remove`, and `position` differences.
- **CF1**: Category-Level F1 over object categories.

### 5.3 Optional reasoning faithfulness and consistency

```bash
python eval/eval_consistency_rate.py \
  --mode responses \
  --response_files outputs/baseline_qwen25/Qwen2.5-VL-7B-Instruct_consistency_responses.json \
  --data_dir work/spd_local/multi_diff \
  --output_dir outputs/consistency
```

```bash
python eval/eval_cot_faithfulness.py \
  --responses_file outputs/baseline_qwen25/Qwen2.5-VL-7B-Instruct_faithfulness_responses.json \
  --data_dir work/spd_local/multi_diff \
  --output_file outputs/cot_faithfulness/qwen25_drf.json
```

---

## 6. Attention Ratio Analysis

The attention-ratio experiment asks whether Qwen2.5-VL places more generation-time attention on the visual tokens corresponding to the actual changed region.

For each sample, define:

```text
attention_ratio = mean_attention(changed_region_visual_tokens)
                  / mean_attention(other_visual_tokens)
```

The script also compares this ratio to random masks of the same size.

Key fields in output JSON:

```text
ratio_sample_mean       # attention ratio on the GT changed region
ratio_random_mean       # attention ratio on random same-size regions
margin_vs_random        # ratio_sample_mean - ratio_random_mean
gt_beats_random         # whether GT ratio is higher than random baseline
ratio_cross_image       # attention contrast between original and modified image regions
per_layer_step_ratio    # layer × generation-step ratio values
```

### 6.1 Build remove-only manifest

For the cleanest attention analysis, use samples where all differences are object removals:

```bash
python scripts/build_remove_manifest_from_splits.py \
  --data_root work/spd_local \
  --splits easy medium hard multi_diff \
  --remove_only \
  --single_only \
  --out_file work/remove_only_all_single.jsonl
```

Alternative: create a local `remove_only` split from `multi_diff`:

```bash
python scripts/filter_remove_only.py \
  --data_root work/spd_local \
  --src_split multi_diff \
  --dst_split remove_only \
  --single_only \
  --overwrite
```

### 6.2 Run robust remove-only attention ratio

```bash
python scripts/attention_ratio_qwen25_remove_only_v2.py \
  --manifest_file work/remove_only_all_single.jsonl \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --question_mode remove \
  --layer_start 0 \
  --layer_end 27 \
  --n_random_masks 8 \
  --content_only \
  --strict_remove_only \
  --out_file outputs/attention_ratio/remove_only_all_single.json \
  --debug_vis_dir outputs/attention_ratio/debug_vis
```

Notes:

- `--question_mode remove` uses a prompt targeted to removal differences.
- `--content_only` pools only generated content tokens instead of all generated tokens.
- `--strict_remove_only` skips samples that are not remove-only.
- `--debug_vis_dir` saves visual overlays of estimated changed regions.

### 6.3 Summarize attention-ratio statistics

```bash
python scripts/analyze_attention_ratio_results.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/summary
```

This writes:

```text
outputs/attention_ratio/summary/summary.json
```

The summary includes overall means, filtered subsets, layer summaries, early/mid/late layer groups, step quartiles, and highest/lowest margin samples.

### 6.4 Plot attention-ratio figures

```bash
python scripts/plot_attention_ratio.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/figures \
  --target_steps 40
```

Optional single-sample heatmap:

```bash
python scripts/plot_attention_ratio.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/figures \
  --sample_id 108495
```

Expected figures:

```text
ratio_distribution_hist.png
ratio_distribution_scatter.png
margin_distribution.png
heatmap_average_layer_step.png
layer_summary_curve.png
layer_prop_gt1_curve.png
step_quartiles_bar.png
```

---

## 7. New Inpainting Methods for Object Removal

The original data generator used LaMa directly for object removal. The CSE559 branch introduces a pluggable inpainting interface:

```text
data/data_pipeline/inpainting_backends.py
```

Supported backends:

| Backend | Name in script | Use case |
|---|---|---|
| LaMa | `lama` | Strong classical object-removal baseline; usually stable for texture completion. |
| OpenCV Telea | `opencv-telea` | Fast sanity baseline; works only for small/simple masks. |
| OpenCV Navier-Stokes | `opencv-ns` | Alternative classical baseline. |
| Stable Diffusion 1.5 Inpainting | `sd15` | Generative baseline; may hallucinate if prompt/mask are poor. |
| SDXL Inpainting | `sdxl` | Higher-quality diffusion inpainting; recommended modern baseline. |
| FLUX.1 Fill | `flux-fill` | Optional high-quality generative backend; heavy and may require gated access. |

Important design choices for benchmark cleanliness:

1. **Preserve unmasked pixels**: non-mask pixels are copied back from the original image to avoid accidental global changes.
2. **Background-only prompt**: the positive prompt describes the background to be filled, not the object to remove.
3. **Negative prompt blocks object regeneration**: object category, faces, fur, people, animals, and duplicate objects are discouraged.
4. **Expanded masks**: removal masks are dilated/closed so leftover object edges do not cause the model to reconstruct the removed object.
5. **Avoid huge foreground subjects**: main subjects are poor removal candidates because the model must hallucinate an unknown background.

Recommended removal candidate area:

```text
0.005 <= object_area_ratio <= 0.15
```

Large subjects, such as a bear occupying most of the image, should generally be skipped for the benchmark because the true background is unobservable.

---

## 8. Reproduce Object Removal and Inpainting Ablation

### 8.1 Single-image object-removal reproduction

Use this to debug one COCO image and inspect the chosen object, mask, and output:

```bash
python scripts/reproduce_object_removal.py \
  --image_id 391895 \
  --fallback_only \
  --out_dir outputs/reproduce_object_removal
```

If the script version uses `--image_ids` rather than `--image_id`, use:

```bash
python scripts/reproduce_object_removal.py \
  --image_ids 391895 \
  --fallback_only \
  --out_dir outputs/reproduce_object_removal
```

### 8.2 Inpainting backend ablation

Run LaMa, OpenCV, SD1.5, and SDXL on the same object-removal target:

```bash
python scripts/run_object_removal_inpainting_ablation.py \
  --image_ids 391895 \
  --backends lama opencv-telea sd15 sdxl \
  --fallback_only \
  --seed 0 \
  --steps 50 \
  --guidance_scale 3.5 \
  --strength 0.85 \
  --padding_mask_crop 0 \
  --out_dir outputs/inpaint_ablation_test
```

Outputs:

```text
outputs/inpaint_ablation_test/000000391895/
├── 000000391895_original.jpg
├── removed_lama.jpg
├── removed_opencv-telea.jpg
├── removed_sd15.jpg
├── removed_sdxl.jpg
├── compare_grid.jpg
└── logs.json
```

### 8.3 Fair comparison with fixed object index

To make backends remove exactly the same COCO annotation:

```bash
python scripts/run_object_removal_inpainting_ablation.py \
  --image_ids 391895 \
  --backends lama opencv-telea sd15 sdxl \
  --fallback_only \
  --force_object_index 3 \
  --seed 0 \
  --steps 50 \
  --guidance_scale 3.5 \
  --strength 0.85 \
  --padding_mask_crop 0 \
  --out_dir outputs/inpaint_ablation_fixed_object
```

### 8.4 Practical inpainting recommendations

For SPD-Faith benchmark generation, visual quality alone is not enough. The goal is to create a clean, single difference.

Recommended settings:

```bash
--backends lama sdxl
--steps 50
--guidance_scale 3.5
--strength 0.85
--padding_mask_crop 0
```

Avoid:

- very large foreground objects;
- people or animals occupying most of the frame;
- masks that cut through faces, fur, hands, or fine structures;
- disabling unmasked-pixel preservation unless explicitly studying model artifacts.

If diffusion regenerates the object, inspect `logs.json` and check:

- mask area ratio;
- positive prompt;
- negative prompt;
- whether object edges remain outside the mask;
- whether the selected object is too large.

---

## 9. Full CSE559 Reproduction Pipeline

A minimal end-to-end reproduction run:

```bash
# 1. Create environment
conda create -n spd-faith python=3.10 -y
conda activate spd-faith
pip install -r requirements.txt
pip install qwen-vl-utils datasets pycocotools opencv-python pillow matplotlib pandas scipy tqdm
pip install simple-lama-inpainting diffusers transformers accelerate safetensors huggingface_hub sentencepiece protobuf certifi

# 2. Export benchmark data
python scripts/export_spd_dataset.py \
  --splits multi_diff \
  --out_root work/spd_local \
  --limit 50

# 3. Run Qwen2.5-VL baseline
python scripts/run_baseline_qwen25vl.py \
  --data_root work/spd_local \
  --split multi_diff \
  --limit 50 \
  --out_dir outputs/baseline_qwen25

# 4. Evaluate DQR / TF1 / CF1
python eval/eval_multi_diff_metrics.py \
  --input_dir outputs/baseline_qwen25 \
  --output_dir outputs/baseline_qwen25_metrics

# 5. Build remove-only manifest
python scripts/build_remove_manifest_from_splits.py \
  --data_root work/spd_local \
  --splits multi_diff \
  --remove_only \
  --single_only \
  --out_file work/remove_only_all_single.jsonl

# 6. Run attention-ratio analysis
python scripts/attention_ratio_qwen25_remove_only_v2.py \
  --manifest_file work/remove_only_all_single.jsonl \
  --question_mode remove \
  --content_only \
  --strict_remove_only \
  --limit 20 \
  --out_file outputs/attention_ratio/remove_only_all_single.json

# 7. Summarize and plot attention-ratio results
python scripts/analyze_attention_ratio_results.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/summary

python scripts/plot_attention_ratio.py \
  --input outputs/attention_ratio/remove_only_all_single.json \
  --out_dir outputs/attention_ratio/figures

# 8. Run inpainting backend ablation on a COCO image
python scripts/run_object_removal_inpainting_ablation.py \
  --image_ids 391895 \
  --backends lama opencv-telea sd15 sdxl \
  --fallback_only \
  --seed 0 \
  --steps 50 \
  --guidance_scale 3.5 \
  --strength 0.85 \
  --padding_mask_crop 0 \
  --out_dir outputs/inpaint_ablation_test
```

---

## 10. Expected Output Artifacts

Typical CSE559 outputs:

```text
outputs/
├── baseline_qwen25/
│   ├── *_multi_diff_responses.json
│   ├── *_consistency_responses.json
│   └── *_faithfulness_responses.json
├── baseline_qwen25_metrics/
│   ├── *_metrics.json
│   └── all_models_comparison.json
├── attention_ratio/
│   ├── remove_only_all_single.json
│   ├── summary/summary.json
│   ├── figures/*.png
│   └── debug_vis/*.png
└── inpaint_ablation_test/
    └── 000000391895/
        ├── compare_grid.jpg
        ├── logs.json
        └── removed_*.jpg
```

---

## 11. Troubleshooting

### 11.1 Git Bash line continuation

In Git Bash, a line-continuation backslash must be the last character on the line. This is valid:

```bash
python script.py \
  --arg value
```

This is invalid because there is a space after `\`:

```bash
python script.py \ 
  --arg value
```

If you see an error like:

```text
ValueError: invalid literal for int() with base 10: ' '
bash: --backends: command not found
```

remove trailing spaces after `\`.

### 11.2 SDXL generates a new object instead of background

This usually means one of the following:

- the positive prompt mentions the removed object;
- the mask is too tight and object edges remain visible;
- the object is the main subject and too large;
- the guidance scale is too high;
- `preserve_unmasked` leaves object remnants outside the mask.

Recommended debug command:

```bash
python scripts/run_object_removal_inpainting_ablation.py \
  --image_ids 391895 \
  --backends lama sdxl \
  --fallback_only \
  --seed 0 \
  --steps 50 \
  --guidance_scale 3.5 \
  --strength 0.85 \
  --padding_mask_crop 0 \
  --out_dir outputs/inpaint_debug
```

Then inspect:

```text
outputs/inpaint_debug/<image_id>/compare_grid.jpg
outputs/inpaint_debug/<image_id>/logs.json
```

### 11.3 `hf auth whoami` fails with SSL certificate error

```bash
unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE
unset CURL_CA_BUNDLE
pip install -U certifi
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
hf auth whoami
```

### 11.4 `api_client.py` crashes although `--fallback_only` is used

The inpainting ablation script installs a fake `api_client` in fallback mode so object selection does not require Gemini/OpenAI. If it still crashes, verify that `scripts/run_object_removal_inpainting_ablation.py` contains the fallback fake-client logic and that you are running the script from the current branch.

---

## 12. Notes for CSE559 Report

Recommended report structure:

1. **Task and motivation**: faithfulness in spot-the-difference VLM reasoning.
2. **Benchmark reproduction**: Qwen2.5-VL results with DQR, TF1, CF1.
3. **Mechanistic analysis**: attention ratio vs random-mask baseline.
4. **Data-generation improvement**: LaMa vs SDXL/SD1.5/OpenCV inpainting.
5. **Failure analysis**: large-object removal, diffusion hallucination, mask leakage, and prompt sensitivity.
6. **Conclusion**: whether correct VLM answers are visually grounded and how data quality affects faithfulness evaluation.

Recommended tables:

| Experiment | Output file | Main metric |
|---|---|---|
| Qwen baseline | `outputs/baseline_qwen25_metrics/*.json` | DQR / TF1 / CF1 |
| Attention ratio | `outputs/attention_ratio/summary/summary.json` | ratio, margin, gt_beats_random |
| Inpainting ablation | `outputs/inpaint_ablation_test/*/logs.json` | visual cleanliness / object regeneration |

---

## 13. Citation / Attribution

This repository uses SPD-Faith-Bench and Qwen2.5-VL-based evaluation scripts for CSE559 research experiments. Please cite the original dataset/model papers or project pages when using released datasets or pretrained model outputs in a report.

