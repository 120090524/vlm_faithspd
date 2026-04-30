# Patch notes for `remove_attention_5`: pluggable inpainting backend

Copy `inpainting_backends.py` into:

```text
data/data_pipeline/inpainting_backends.py
```

Copy `run_object_removal_inpainting_ablation.py` into:

```text
scripts/run_object_removal_inpainting_ablation.py
```

Then patch `data/data_pipeline/generator.py` as follows.

## 1. Make LaMa import optional and import backend factory

Replace:

```python
from simple_lama_inpainting import SimpleLama
```

with:

```python
try:
    from simple_lama_inpainting import SimpleLama
except Exception:
    SimpleLama = None

from inpainting_backends import build_inpainter
```

## 2. Extend `SpotDifferenceGenerator.__init__`

Replace:

```python
def __init__(self, coco_ann_file, coco_img_dir):
    self.coco = COCO(coco_ann_file)
    self.coco_img_dir = Path(coco_img_dir)
    self.client = client
    self.group_counter = 0
    self.lama_model = None
```

with:

```python
def __init__(self, coco_ann_file, coco_img_dir, inpaint_backend="lama", inpaint_backend_kwargs=None):
    self.coco = COCO(coco_ann_file)
    self.coco_img_dir = Path(coco_img_dir)
    self.client = client
    self.group_counter = 0
    self.lama_model = None  # kept for backward compatibility
    self.inpaint_backend_name = inpaint_backend
    self.inpaint_backend_kwargs = inpaint_backend_kwargs or {}
    self.inpainter = None
```

## 3. Add helper methods inside the class

Add these methods near `_init_lama_model`:

```python
def _init_inpainter(self):
    if self.inpainter is None:
        self.inpainter = build_inpainter(self.inpaint_backend_name, **self.inpaint_backend_kwargs)

def _run_inpaint(self, img_bgr, mask_u8, prompt="", negative_prompt="", seed=0):
    self._init_inpainter()
    return self.inpainter(
        img_bgr,
        mask_u8,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
    )

def _remove_prompt(self, category_name):
    return (
        "realistic photograph of the same scene after removing the "
        f"{category_name}; fill the masked region with only natural background, "
        "consistent texture, lighting, perspective, and no new salient object"
    )

def _remove_negative_prompt(self, category_name):
    return (
        f"{category_name}, removed object, duplicate object, new object, text, watermark, "
        "cartoon, painting, unrealistic, blurry, artifacts, distorted edges"
    )
```

## 4. Extend `_remove_object` signature and support a fixed object index

Change:

```python
def _remove_object(self, image_id, excluded_indices=None):
```

into:

```python
def _remove_object(self, image_id, excluded_indices=None, force_object_index=None, seed=0):
```

After `objects_info` is constructed and before `llm_choice = ...`, insert:

```python
selected_ann = None
selection_reason = None

if force_object_index is not None:
    force_object_index = int(force_object_index)
    if force_object_index not in available_indices:
        raise ValueError(
            f"force_object_index={force_object_index} is not available; "
            f"available_indices={available_indices}"
        )
    selected_original_idx = force_object_index
    selected_ann = anns[selected_original_idx]
    selection_reason = f"forced object_index={force_object_index}"
```

Then wrap the existing LLM/fallback selection block with:

```python
if selected_ann is None:
    # existing llm_choice + fallback selection code goes here
```

## 5. Replace the LaMa call in `_remove_object`

Replace the block from:

```python
self._init_lama_model()
# 将图像和 mask 填充到8的倍数...
...
result_bgr = self._crop_padding(result_bgr, pad_h, pad_w)
```

with:

```python
result_bgr = self._run_inpaint(
    img,
    mask,
    prompt=self._remove_prompt(cat_info["name"]),
    negative_prompt=self._remove_negative_prompt(cat_info["name"]),
    seed=seed,
)
```

## 6. Replace LaMa calls in `_change_object_position`

Replace the first LaMa removal call with:

```python
img_removed = self._run_inpaint(
    img.copy(),
    mask_smooth,
    prompt=self._remove_prompt(cat_info["name"]),
    negative_prompt=self._remove_negative_prompt(cat_info["name"]),
    seed=0,
)
```

Replace the edge-repair LaMa call with:

```python
result = self._run_inpaint(
    result,
    edge_only_smooth,
    prompt="realistic photo, seamless clean boundary, consistent local texture and lighting",
    negative_prompt="ghosting, duplicated object, distorted edge, blur, artifact",
    seed=1,
)
```

## 7. Dependencies

Add at least:

```text
diffusers>=0.35.0
safetensors>=0.4.0
sentencepiece>=0.2.0
protobuf>=4.25.0
simple-lama-inpainting>=0.1.2
```

For FLUX.1 Fill, run `huggingface-cli login` and accept the FLUX model terms on Hugging Face before the first run.
