import json
import os

import torch

import comfy.model_management as mm

from .nodes_shared import (
    MODEL_REPO_ID_APPEARANCE,
    MODEL_REPO_ID_LIGHTING,
    _cleanup_hf_cache,
    _ensure_base_runtime_files,
    _ensure_model_files,
    _marigold_zero_empty_conditioning,
    _require_module,
    _select_torch_dtype,
)


class _MarigoldBaseLoader:
    def __init__(self, repo_id: str, kind: str):
        self.repo_id = str(repo_id)
        self.kind = str(kind)
        self._pipe = None
        self._config = None

    def loadmodel(self, model_variant: str, precision: str):
        device = mm.get_torch_device()
        if not isinstance(device, torch.device):
            device = torch.device(device)

        dtype = _select_torch_dtype(precision, device)
        if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            # Diffusers pipelines cannot execute fp16/bf16 reliably on CPU.
            dtype = torch.float32

        config = {
            "model_variant": str(model_variant),
            "dtype": str(dtype),
            "device": str(device),
        }

        if self._pipe is None or self._config != config:
            _require_module("diffusers")

            if model_variant not in ("fp16", "fp32"):
                raise ValueError(f"Invalid model_variant={model_variant!r}")

            from diffusers import AutoencoderKL, DDIMScheduler, MarigoldIntrinsicsPipeline, UNet2DConditionModel

            base_dir = _ensure_base_runtime_files(model_variant, self.repo_id)
            model_dir = _ensure_model_files(self.repo_id, self.kind, model_variant)

            with open(os.path.join(model_dir, "model_index.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)

            unet = UNet2DConditionModel.from_pretrained(
                model_dir,
                subfolder="unet",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            vae = AutoencoderKL.from_pretrained(
                base_dir,
                subfolder="vae",
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
            )
            scheduler = DDIMScheduler.from_pretrained(
                base_dir,
                subfolder="scheduler",
                local_files_only=True,
            )

            pipe = MarigoldIntrinsicsPipeline(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                text_encoder=None,
                tokenizer=None,
                prediction_type=cfg.get("prediction_type"),
                target_properties=cfg.get("target_properties"),
                default_denoising_steps=cfg.get("default_denoising_steps"),
                default_processing_resolution=cfg.get("default_processing_resolution"),
            )
            pipe.empty_text_embedding = _marigold_zero_empty_conditioning(unet=unet, dtype=dtype, device=device)

            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[Marigold IID] xFormers not enabled: {e}")

            pipe = pipe.to(device)

            self._pipe = pipe
            self._config = config
            _cleanup_hf_cache()

        return (
            {
                "pipe": self._pipe,
                "dtype": dtype,
                "kind": self.kind,
                "repo_id": self.repo_id,
                "keep_on_gpu": True,
            },
        )


_APPEARANCE_LOADER = _MarigoldBaseLoader(MODEL_REPO_ID_APPEARANCE, "appearance")
_LIGHTING_LOADER = _MarigoldBaseLoader(MODEL_REPO_ID_LIGHTING, "lighting")


def load_marigold_iid_appearance_base_model(model_variant: str, precision: str):
    return _APPEARANCE_LOADER.loadmodel(model_variant=model_variant, precision=precision)


def load_marigold_iid_lighting_base_model(model_variant: str, precision: str):
    return _LIGHTING_LOADER.loadmodel(model_variant=model_variant, precision=precision)
