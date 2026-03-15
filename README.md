# SPD-Faith: Spot-the-Difference Faithfulness Benchmark

A comprehensive benchmark for evaluating vision-language models' faithfulness in spot-the-difference task.

## Dataset

The SPD-Faith-Bench dataset is available on Hugging Face:

[<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" width="24"/> **SPD-Faith-Bench**](https://huggingface.co/datasets/Jackson-Lv/SPD-Faith-Bench)

## Project Structure

```
SPD-Faith-Clean/
├── analysis/                         # Mechanism analysis modules
│   ├── analyze_layer_changes.py     # MHA/FFN layer change analysis
│   ├── layer_analysis_generation.py  # Generation-phase attention analysis
│   ├── neuron_activation.py          # Neuron activation analysis
│   ├── tam.py                        # Token Activation Map
│   └── demo.py                       # Quick demo script
├── data/                             # Data and pipeline
│   ├── data_pipeline/                # Data generation pipeline
│   │   ├── api_client.py
│   │   ├── config.py
│   │   ├── generator.py
│   │   └── main.py
│   └── examples/                    # Sample images (easy / medium / hard / multi_diff)
│       ├── easy/
│       ├── medium/
│       ├── hard/
│       └── multi_diff/
├── eval/                             # Evaluation modules
│   ├── eval_multi_diff_metrics.py    # Difference detection metrics (DQR, DS, TF1, CF1)
│   ├── eval_cot_faithfulness.py      # CoT faithfulness evaluation (DRF)
│   └── eval_consistency_rate.py      # Consistency rate evaluation (CR)
├── fig/                              # Figures
│   ├── fig1.png
│   ├── fig2.png
│   ├── fig3.png
│   └── fig4.png
├── utils/                            # Utility modules
│   ├── model_utils.py                # Model loading utilities
│   └── visualization.py              # Visualization utilities
├── requirements.txt                  # Python dependencies
├── LICENSE
└── README.md                         # This file
```

## Quick Start

### Installation

```bash
conda create -n spd-faith python=3.10
conda activate spd-faith

pip install -r requirements.txt
pip install qwen-vl-utils
```

### Run Demo

```bash
python mechanism_analysis/demo.py --image_path path/to/image.jpg --question "Are the two pictures same?"
```

## Evaluation Modules

### 1. Difference Detection Metrics (eval_multi_diff_metrics.py)

Evaluates model performance on identifying image differences:

**Global Perception:**
- **DQR (Difference Quantity Recall)**: Whether predicted count equals GT count
- **DS (Difference Sensitivity)**: Sensitivity to difference count deviations

**Faithful Perception:**
- **TF1 (Type-Level F1)**: Accuracy on difference types (color/remove/position)
- **CF1 (Category-Level F1)**: Accuracy on object categories

```bash
python evaluation/eval_multi_diff_metrics.py \
    --input_dir results/predictions/ \
    --output_dir results/metrics/
```

### 2. CoT Faithfulness Evaluation (eval_cot_faithfulness.py)

**Faithful Reasoning:**
- **DRF (Difference Reasoning Faithfulness)**: Detects unfaithful shortcut reasoning using GPT-4o

```bash
python evaluation/eval_cot_faithfulness.py \
    --responses_file results/responses.json \
    --data_dir data/multi_diff/ \
    --output_file results/faithfulness.json
```

### 3. Consistency Rate Evaluation (eval_consistency_rate.py)

**Faithful Reasoning:**
- **CR (Consistency Ratio)**: Consistency between answers to complementary questions

```bash
python evaluation/eval_consistency_rate.py \
    --mode responses \
    --response_files results/responses.json \
    --data_dir data/multi_diff/ \
    --output_dir results/consistency/
```

### 4. Hidden State Cosine Analysis (eval_cosine.py)

Analyzes hidden state similarities and ICR scores for contradiction detection.

```bash
python evaluation/eval_cosine.py \
    --model qwen \
    --comparison_result results/comparison.json \
    --image_dir data/images/ \
    --output_dir results/faithfulness/
```

## Mechanism Analysis Modules

### 1. MHA/FFN Layer Change Analysis

Analyzes representation changes in Multi-Head Attention and Feed-Forward Network sublayers.

**Metrics:**
- L2 norm changes
- Cosine similarity
- KL divergence between MHA and FFN distributions

```bash
python mechanism_analysis/analyze_layer_changes.py \
    --input_file data/samples.jsonl \
    --output_file results/layer_changes.json \
    --output_pdf results/layer_changes.pdf \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --image_dir data/images/ \
    --max_samples 200
```

### 2. Generation-Phase Attention Analysis

Tracks token-wise attention allocation during generation.

```bash
python mechanism_analysis/layer_analysis_generation.py \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --multi-diff-dir data/multi_diff/ \
    --output-dir results/generation_plots/ \
    --layer-start 0 --layer-end 27
```

### 3. Neuron Activation Analysis

Tracks binary activation states of FFN intermediate neurons.

```bash
python mechanism_analysis/neuron_activation.py \
    --in_file_path data/samples.jsonl \
    --visualize_path results/neuron_activation.pdf \
    --pretrained_model_path Qwen/Qwen2-VL-7B-Instruct \
    --image_dir data/images/ \
    --max_samples 100
```

### 4. Token Activation Map (TAM)

Generates visual explanations for VLM predictions.

```bash
python mechanism_analysis/demo.py \
    --image_path data/test_image.jpg \
    --question "Are the two pictures same?"
```

## Data Format

### Input Sample (JSONL)
```json
{
  "qid": "100238_color",
  "image": "merged_images/100238_color.jpg",
  "modifications": [{"type": "color", "category": "chair"}],
  "is_faithful": 1,
  "question": "Are the two pictures same?"
}
```

### Model Output (JSON)
```json
{
  "qid": "100238_color",
  "predicted_num": 1,
  "structured_output": [{"type": "color", "category": "chair"}],
  "response_text": "There is one difference: the chair color.",
  "is_faithful": 1
}
```

## Supported Models

- **Qwen2-VL-7B-Instruct**: Default model for most analyses
- **InternVL2.5-8B**: Alternative VLM
- **LLaVA-1.5-7B**: Alternative VLM

## Benchmark Results

| Model | DRF | TF1 | CF1 | DQR | DS | CR | Overall |
|-------|-----|-----|-----|-----|-----|-----|---------|
| GLM-4.5V | 58.3 | 69.4 | 69.6 | 67.0 | 92.5 | 87.7 | 74.1 |
| Gemini-2.5-Pro | 40.4 | 66.1 | 62.8 | 57.3 | 79.7 | 78.2 | 64.1 |
| Claude-4.5-Haiku | 38.2 | 54.7 | 51.4 | 54.2 | 64.5 | 65.4 | 54.7 |
| GPT-4o | 39.3 | 56.8 | 59.7 | 62.2 | 82.0 | 74.8 | 62.5 |
| Qwen3-VL-235B-A22B | 44.8 | 59.5 | 60.9 | 62.8 | 90.1 | 82.3 | 66.7 |
| Qwen2.5-VL-72B | 31.7 | 48.1 | 56.6 | 61.6 | 44.4 | 56.8 | 49.9 |

## Requirements

- Python >= 3.8
- CUDA >= 11.8
- GPU Memory >= 24GB (recommended)

## 补充本地复现新增脚本说明

下面这部分是基于本地复现流程新增的 `scripts/` 目录说明，主要用于 **Qwen2.5-VL-7B**、benchmark 导出、baseline 生成、analysis 适配和反事实图像构造。它们属于**本地复现辅助脚本**，不是原始仓库官方发布内容。

- `export_spd_dataset.py`：从 Hugging Face 导出 SPD-Faith-Bench 到本地目录（如 `work/spd_local/`），同时保存图片、`metadata.json` 和 `jsonl` 清单。
- `check_qwen25vl.py`：最小连通性测试脚本，用来确认 Qwen2.5-VL-7B 可以正常加载，并且能读取双图输入完成简单问答。
- `run_baseline_qwen25vl.py`：运行 Qwen2.5-VL-7B baseline，生成三类输出：`multi_diff`、`consistency`、`faithfulness`，供后续评测使用。
- `build_analysis_jsonl.py`：把 baseline 输出和 metrics 整理成 analysis 所需的 `jsonl`，并基于 soft rule 给样本打上 `faithful / unfaithful` 标签。
- `patch_analysis_for_qwen25.py`：复制并修补官方 analysis 脚本，使其兼容 Qwen2.5-VL-7B（例如替换模型类名和默认模型路径）。
- `layer_analysis_generation_qwen25.py`：Qwen2.5 版本的 generation attention 分析脚本，用来观察生成阶段对 `system / visual / user` 三类 token 的注意力变化。
- `analyze_layer_changes_qwen25.py`：Qwen2.5 版本的层变化分析脚本，用来比较 MHA / FFN 的表示变化、cosine 和 KL divergence。
- `neuron_activation_qwen25.py`：Qwen2.5 版本的神经元激活分析脚本，用来比较 faithful / unfaithful 两组样本在 FFN 中间层的激活差异。
- `run_tam_one_sample.py`：单样本 TAM 可视化脚本，用来对某个 faithful 或 unfaithful 样本生成 token-level 热图。
- `reproduce_object_removal.py`：用于复现论文中的“物体消失”反事实图像构造流程，支持 `planner` 和 `fallback_only` 两种模式。

### 建议使用顺序

1. 先运行 `export_spd_dataset.py` 导出数据。  
2. 再运行 `check_qwen25vl.py` 确认模型加载和双图输入正常。  
3. 使用 `run_baseline_qwen25vl.py` 生成 baseline 输出。  
4. 跑官方 `eval/` 脚本得到 metrics。  
5. 用 `build_analysis_jsonl.py` 生成 faithful / unfaithful 分组文件。  
6. 用 `patch_analysis_for_qwen25.py` 和 `*_qwen25.py` 脚本完成 generation attention、layer changes、neuron activation 等分析。  
7. 如需可视化单样本证据区域，可运行 `run_tam_one_sample.py`。  
8. 如需复现 benchmark 构造中的反事实图像编辑，可运行 `reproduce_object_removal.py`。  


## Acknowledgements

We thank the authors of the TAM (Token Activation Map) paper for their inspiring work and for providing ideas that motivated parts of our mechanism analysis. We are grateful for their contributions to interpretability research in vision-language models.  
Project page: https://github.com/xmed-lab/TAM

## Acknowledgements

We thank the authors of the TAM (Token Activation Map) paper for their inspiring work and for providing ideas that motivated parts of our mechanism analysis. We are grateful for their contributions to interpretability research in vision-language models.  
Project page: https://github.com/xmed-lab/TAM
