from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def ensure_mask_u8(mask: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask where 255 means repaint / inpaint."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(ensure_mask_u8(mask)).convert("L")


def pad_image_and_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    multiple: int = 8,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pad bottom/right so image size is divisible by `multiple`."""
    h, w = img_bgr.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return img_bgr, mask_u8, (0, 0)
    img_pad = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    mask_pad = cv2.copyMakeBorder(mask_u8, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return img_pad, mask_pad, (pad_h, pad_w)


def crop_padding(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    if pad_h == 0 and pad_w == 0:
        return img
    h, w = img.shape[:2]
    return img[: h - pad_h if pad_h else h, : w - pad_w if pad_w else w]


def blend_keep_unmasked(
    original_bgr: np.ndarray,
    generated_bgr: np.ndarray,
    mask_u8: np.ndarray,
    feather: int = 5,
) -> np.ndarray:
    """Preserve non-masked area exactly; optionally feather only the mask boundary."""
    mask_u8 = ensure_mask_u8(mask_u8)
    if generated_bgr.shape[:2] != original_bgr.shape[:2]:
        generated_bgr = cv2.resize(
            generated_bgr,
            (original_bgr.shape[1], original_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    if feather and feather > 0:
        k = int(feather) * 2 + 1
        alpha = cv2.GaussianBlur(mask_u8.astype(np.float32) / 255.0, (k, k), 0)
        alpha = alpha[:, :, None]
        out = generated_bgr.astype(np.float32) * alpha + original_bgr.astype(np.float32) * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)

    out = original_bgr.copy()
    out[mask_u8 > 0] = generated_bgr[mask_u8 > 0]
    return out


@dataclass
class BaseInpainter:
    preserve_unmasked: bool = True
    feather: int = 5
    pad_multiple: int = 8

    def __call__(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        *,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
    ) -> np.ndarray:
        mask_u8 = ensure_mask_u8(mask)
        img_pad, mask_pad, (pad_h, pad_w) = pad_image_and_mask(img_bgr, mask_u8, self.pad_multiple)
        gen_pad = self._inpaint(
            img_pad,
            mask_pad,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        if self.preserve_unmasked:
            gen_pad = blend_keep_unmasked(img_pad, gen_pad, mask_pad, feather=self.feather)
        return crop_padding(gen_pad, pad_h, pad_w)

    def _inpaint(
        self,
        img_bgr: np.ndarray,
        mask_u8: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class OpenCVInpainter(BaseInpainter):
    method: str = "telea"  # "telea" or "ns"
    radius: int = 3

    def _inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray, *, prompt: str, negative_prompt: str, seed: int) -> np.ndarray:
        flag = cv2.INPAINT_TELEA if self.method.lower() == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(img_bgr, mask_u8, self.radius, flag)


class LamaInpainter(BaseInpainter):
    def __init__(self, device: Optional[str] = None, preserve_unmasked: bool = True, feather: int = 5):
        super().__init__(preserve_unmasked=preserve_unmasked, feather=feather, pad_multiple=8)
        import torch
        from simple_lama_inpainting import SimpleLama

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = SimpleLama(device=self.device)

    def _inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray, *, prompt: str, negative_prompt: str, seed: int) -> np.ndarray:
        out = self.model(bgr_to_pil(img_bgr), mask_to_pil(mask_u8))
        return pil_to_bgr(out)


class DiffusersInpainter(BaseInpainter):
    """Stable Diffusion / SDXL inpainting through diffusers.AutoPipelineForInpainting."""

    def __init__(
        self,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        variant: Optional[str] = "fp16",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.98,
        padding_mask_crop: Optional[int] = 32,
        cpu_offload: bool = True,
        preserve_unmasked: bool = True,
        feather: int = 5,
    ):
        super().__init__(preserve_unmasked=preserve_unmasked, feather=feather, pad_multiple=8)
        import torch
        from diffusers import AutoPipelineForInpainting

        self.model_id = model_id
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.strength = float(strength)
        self.padding_mask_crop = padding_mask_crop
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        kwargs = {"torch_dtype": dtype}
        if variant:
            kwargs["variant"] = variant
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)
        except TypeError:
            # Some repos do not define an fp16 variant.
            kwargs.pop("variant", None)
            self.pipe = AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)

        if cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        # Safe optional optimization. Ignore if unavailable or already handled by PyTorch SDPA.
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def _inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray, *, prompt: str, negative_prompt: str, seed: int) -> np.ndarray:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        kwargs = dict(
            prompt=prompt,
            image=bgr_to_pil(img_bgr),
            mask_image=mask_to_pil(mask_u8),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if self.padding_mask_crop is not None and self.padding_mask_crop > 0:
            kwargs["padding_mask_crop"] = int(self.padding_mask_crop)
        out = self.pipe(**kwargs).images[0]
        return pil_to_bgr(out)


class FluxFillInpainter(BaseInpainter):
    """FLUX.1 Fill through diffusers.FluxFillPipeline. Requires HF access acceptance for FLUX.1-Fill-dev."""

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        device: Optional[str] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 30.0,
        max_sequence_length: int = 512,
        cpu_offload: bool = True,
        preserve_unmasked: bool = True,
        feather: int = 5,
    ):
        super().__init__(preserve_unmasked=preserve_unmasked, feather=feather, pad_multiple=16)
        import torch
        from diffusers import FluxFillPipeline

        self.model_id = model_id
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.max_sequence_length = int(max_sequence_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=dtype)
        if cpu_offload and self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

    def _inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray, *, prompt: str, negative_prompt: str, seed: int) -> np.ndarray:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            image=bgr_to_pil(img_bgr),
            mask_image=mask_to_pil(mask_u8),
            height=img_bgr.shape[0],
            width=img_bgr.shape[1],
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
        ).images[0]
        return pil_to_bgr(out)


def build_inpainter(name: str, **kwargs) -> BaseInpainter:
    name = name.lower().strip()
    if name in {"opencv", "opencv-telea", "telea"}:
        return OpenCVInpainter(method="telea", **kwargs)
    if name in {"opencv-ns", "ns"}:
        return OpenCVInpainter(method="ns", **kwargs)
    if name in {"lama", "simple-lama"}:
        return LamaInpainter(**kwargs)
    if name in {"sd15", "sd-1.5", "stable-diffusion"}:
        kwargs.setdefault("model_id", "stable-diffusion-v1-5/stable-diffusion-inpainting")
        return DiffusersInpainter(**kwargs)
    if name in {"sdxl", "sdxl-inpaint"}:
        kwargs.setdefault("model_id", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
        return DiffusersInpainter(**kwargs)
    if name in {"flux", "flux-fill", "flux1-fill"}:
        return FluxFillInpainter(**kwargs)
    raise ValueError(f"Unknown inpainting backend: {name}")
