import gc
import os
import glob
import re

import torch

import comfy.model_management as mm
import folder_paths

from .nodes_loaders import (
    load_marigold_iid_appearance_base_model,
    load_marigold_iid_lighting_base_model,
)
from .nodes_pager import (
    PAGER_MODEL_IDS,
    _pager_taesd_choices,
    _ensure_pager_model_files,
    load_pager_base_model,
)
from .nodes_shared import (
    MODEL_REPO_ID_APPEARANCE,
    MODEL_REPO_ID_LIGHTING,
    STORAGE_DIRNAME,
    _ensure_model_files,
    _marigold_zero_empty_conditioning,
)
from .nodes_tensorrt import TensorRTUNetWrapper

_ENGINE_AUTO = "__AUTO__"
_ENGINE_EXTS = (".engine", ".trt")


def _normalize_engine_path(engine_path: str) -> str:
    text = str(engine_path or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    return text


def _engine_storage_roots() -> list[str]:
    roots: list[str] = []

    models_dir = getattr(folder_paths, "models_dir", None)
    if isinstance(models_dir, str) and models_dir.strip():
        roots.append(os.path.join(models_dir, STORAGE_DIRNAME))

    # StabilityMatrix/portable fallback: resolve from this custom node location.
    local_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    roots.append(os.path.join(local_models, STORAGE_DIRNAME))

    deduped = []
    seen = set()
    for root in roots:
        full = os.path.abspath(root)
        key = full.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(full)
    return deduped


def _list_engine_files(glob_patterns: list[str]) -> list[str]:
    found = []
    for pattern in glob_patterns:
        found.extend(glob.glob(pattern, recursive=True))
    uniq = sorted(
        {
            os.path.abspath(path)
            for path in found
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in _ENGINE_EXTS
        }
    )
    return uniq


def _engine_globs(kind: str) -> list[str]:
    patterns: list[str] = []
    roots = _engine_storage_roots()
    if kind == "appearance":
        for root in roots:
            patterns.extend(
                [
                    os.path.join(root, "appearance", "*", "trt", "*.engine"),
                    os.path.join(root, "appearance", "*", "trt", "*.trt"),
                ]
            )
        return patterns
    if kind == "lighting":
        for root in roots:
            patterns.extend(
                [
                    os.path.join(root, "lighting", "*", "trt", "*.engine"),
                    os.path.join(root, "lighting", "*", "trt", "*.trt"),
                ]
            )
        return patterns
    if kind == "pager":
        for root in roots:
            patterns.extend(
                [
                    os.path.join(root, "**", "trt", "pager_*_unet_*.engine"),
                    os.path.join(root, "**", "trt", "pager_*_unet_*.trt"),
                ]
            )
        return patterns
    for root in roots:
        patterns.extend(
            [
                os.path.join(root, "**", "trt", "*.engine"),
                os.path.join(root, "**", "trt", "*.trt"),
            ]
        )
    return patterns


def _engine_dropdown(kind: str) -> list[str]:
    choices = _list_engine_files(_engine_globs(kind))
    if choices:
        return choices + [_ENGINE_AUTO]
    return [_ENGINE_AUTO]


def _infer_model_variant_from_engine(engine_path: str, default: str = "fp16") -> str:
    parts = [p.lower() for p in os.path.normpath(engine_path).split(os.sep)]
    if "trt" in parts:
        idx = parts.index("trt")
        if idx > 0 and parts[idx - 1] in ("fp16", "fp32"):
            return parts[idx - 1]
    name = os.path.basename(engine_path).lower()
    m = re.search(r"(fp16|fp32)", name)
    if m is not None:
        return m.group(1)
    return str(default)


def _infer_pager_model_id_from_engine(engine_path: str) -> str:
    norm_parts = [p.lower() for p in os.path.normpath(engine_path).split(os.sep)]
    file_name = os.path.basename(engine_path).lower()

    # Preferred: resolve from storage folder name: .../pager/<repo_id with "__">/trt/<engine>.
    for repo_id in PAGER_MODEL_IDS:
        key = repo_id.replace("/", "__").lower()
        if key in norm_parts:
            return repo_id

    # Fallback: resolve from engine filename emitted by builder.
    for repo_id in PAGER_MODEL_IDS:
        key1 = f"pager_{repo_id.replace('/', '_').lower()}_unet_"
        key2 = f"pager_{repo_id.replace('/', '__').lower()}_unet_"
        if key1 in file_name or key2 in file_name:
            return repo_id

    raise RuntimeError(
        "Could not infer PaGeR model_id from selected engine path. "
        "Use an engine built by `Load UNet -> TensorRT Engine` for a PaGeR model."
    )


def _resolve_engine_path(
    engine_file: str,
    fallback_globs: list[str] | None = None,
) -> str:
    text = _normalize_engine_path(engine_file)
    if text and text != _ENGINE_AUTO:
        path = os.path.abspath(text)
        if not os.path.isfile(path):
            raise RuntimeError(f"TensorRT engine not found at '{path}'.")
        if os.path.splitext(path)[1].lower() not in _ENGINE_EXTS:
            raise RuntimeError(f"Engine must be a TensorRT .engine or .trt file, got '{path}'.")
        return path

    matches = _list_engine_files(fallback_globs or [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "Multiple TensorRT engines found. Select one from the engine_file dropdown."
        )
    raise RuntimeError("No TensorRT .engine/.trt file found. Build one first.")


def _current_device() -> torch.device:
    device = mm.get_torch_device()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    return device


def _inject_marigold_tensorrt_unet(iid_model: dict, model_dir: str, engine_path: str) -> dict:
    pipe = iid_model.get("pipe", None) if isinstance(iid_model, dict) else None
    if pipe is None:
        raise RuntimeError("Invalid IIDMODEL produced by base loader: missing 'pipe'.")

    old_unet = getattr(pipe, "unet", None)
    trt_unet = TensorRTUNetWrapper.from_model_dir(
        engine_path=engine_path,
        model_dir=model_dir,
        default_text_tokens=2,
    )
    pipe.unet = trt_unet

    dtype = iid_model.get("dtype", torch.float16)
    device = _current_device()
    pipe.empty_text_embedding = _marigold_zero_empty_conditioning(unet=trt_unet, dtype=dtype, device=device)
    pipe = pipe.to(device)

    if old_unet is not None:
        del old_unet
    gc.collect()
    mm.soft_empty_cache()

    out = dict(iid_model)
    out["pipe"] = pipe
    out["backend"] = "tensorrt"
    out["engine_path"] = engine_path
    out["keep_on_gpu"] = True
    return out


class DownloadAndLoadMarigoldIIDAppearanceModelRT:
    @classmethod
    def INPUT_TYPES(cls):
        engine_choices = _engine_dropdown("appearance")
        default_engine = engine_choices[0]
        return {
            "required": {
                "engine_file": (engine_choices, {"default": default_engine}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IIDMODEL",)
    RETURN_NAMES = ("iid_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "TensorRT/Marigold IID Appearance"
    DESCRIPTION = "Loads Marigold IID Appearance with TensorRT UNet backend."

    def loadmodel(self, engine_file: str, precision: str):
        engine_path = _resolve_engine_path(
            engine_file=engine_file,
            fallback_globs=_engine_globs("appearance"),
        )
        model_variant = _infer_model_variant_from_engine(engine_path, default="fp16")
        model_dir = _ensure_model_files(MODEL_REPO_ID_APPEARANCE, "appearance", model_variant)

        config = {
            "model_variant": str(model_variant),
            "precision": str(precision),
            "engine_file": str(engine_file),
            "engine_path": engine_path,
        }
        if getattr(self, "_cache", None) is not None and self._cache.get("config") == config:
            return self._cache["value"]

        base_value = load_marigold_iid_appearance_base_model(model_variant=model_variant, precision=precision)
        base_model = base_value[0]

        trt_model = _inject_marigold_tensorrt_unet(base_model, model_dir, engine_path)

        result = (trt_model,)
        self._cache = {"config": config, "value": result}
        return result


class DownloadAndLoadMarigoldIIDLightingModelRT:
    @classmethod
    def INPUT_TYPES(cls):
        engine_choices = _engine_dropdown("lighting")
        default_engine = engine_choices[0]
        return {
            "required": {
                "engine_file": (engine_choices, {"default": default_engine}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IIDMODEL",)
    RETURN_NAMES = ("iid_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "TensorRT/Marigold IID Lighting"
    DESCRIPTION = "Loads Marigold IID Lighting with TensorRT UNet backend."

    def loadmodel(self, engine_file: str, precision: str):
        engine_path = _resolve_engine_path(
            engine_file=engine_file,
            fallback_globs=_engine_globs("lighting"),
        )
        model_variant = _infer_model_variant_from_engine(engine_path, default="fp16")
        model_dir = _ensure_model_files(MODEL_REPO_ID_LIGHTING, "lighting", model_variant)

        config = {
            "model_variant": str(model_variant),
            "precision": str(precision),
            "engine_file": str(engine_file),
            "engine_path": engine_path,
        }
        if getattr(self, "_cache", None) is not None and self._cache.get("config") == config:
            return self._cache["value"]

        base_value = load_marigold_iid_lighting_base_model(model_variant=model_variant, precision=precision)
        base_model = base_value[0]

        trt_model = _inject_marigold_tensorrt_unet(base_model, model_dir, engine_path)

        result = (trt_model,)
        self._cache = {"config": config, "value": result}
        return result


class DownloadAndLoadPaGeRModelRT:
    @classmethod
    def INPUT_TYPES(cls):
        taesd_choices = _pager_taesd_choices()
        default_taesd = "taesd" if "taesd" in taesd_choices else taesd_choices[0]
        engine_choices = _engine_dropdown("pager")
        default_engine = engine_choices[0]
        return {
            "required": {
                "engine_file": (engine_choices, {"default": default_engine}),
                "precision": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "keep_on_gpu": ("BOOLEAN", {"default": True}),
                "vae_tiling": ("BOOLEAN", {"default": False, "advanced": True}),
                "use_comfy_taesd_vae": ("BOOLEAN", {"default": False}),
                "taesd_vae_name": (taesd_choices, {"default": default_taesd, "advanced": True}),
                "force_gpu": ("BOOLEAN", {"default": True, "advanced": True}),
            },
        }

    RETURN_TYPES = ("PAGERMODEL",)
    RETURN_NAMES = ("pager_model",)
    FUNCTION = "load"
    CATEGORY = "TensorRT/PaGeR"
    DESCRIPTION = "Loads PaGeR with TensorRT UNet backend."

    def load(
        self,
        engine_file: str,
        precision: str,
        keep_on_gpu: bool = True,
        vae_tiling: bool = False,
        use_comfy_taesd_vae: bool = False,
        taesd_vae_name: str = "taesd",
        force_gpu: bool = True,
    ):
        engine_path = _resolve_engine_path(
            engine_file=engine_file,
            fallback_globs=_engine_globs("pager"),
        )
        model_id = _infer_pager_model_id_from_engine(engine_path)
        model_dir = _ensure_pager_model_files(model_id)
        config = {
            "model_id": str(model_id),
            "precision": str(precision),
            "keep_on_gpu": bool(keep_on_gpu),
            "vae_tiling": bool(vae_tiling),
            "use_comfy_taesd_vae": bool(use_comfy_taesd_vae),
            "taesd_vae_name": str(taesd_vae_name),
            "force_gpu": bool(force_gpu),
            "engine_file": str(engine_file),
            "engine_path": engine_path,
        }
        if getattr(self, "_cache", None) is not None and self._cache.get("config") == config:
            return self._cache["value"]

        base_value = load_pager_base_model(
            model_id=model_id,
            precision=precision,
            keep_on_gpu=keep_on_gpu,
            vae_tiling=vae_tiling,
            use_comfy_taesd_vae=use_comfy_taesd_vae,
            taesd_vae_name=taesd_vae_name,
            force_gpu=force_gpu,
        )
        base_model = base_value[0]

        pipe = base_model.get("pipe", None) if isinstance(base_model, dict) else None
        if pipe is None:
            raise RuntimeError("Invalid PAGERMODEL produced by base loader: missing 'pipe'.")

        modality = str(base_model.get("modality", ""))
        if modality not in ("depth", "normal"):
            raise RuntimeError(f"Unsupported PaGeR modality: {modality!r}")
        if not hasattr(pipe, "unet") or modality not in pipe.unet:
            raise RuntimeError(f"PaGeR pipeline missing UNet for modality '{modality}'.")

        tokens = 77
        if hasattr(pipe, "_expected_text_token_length"):
            try:
                tokens = int(pipe._expected_text_token_length())
            except Exception:
                tokens = 77

        old_unet = pipe.unet.get(modality, None)
        pipe.unet[modality] = TensorRTUNetWrapper.from_model_dir(
            engine_path=engine_path,
            model_dir=model_dir,
            default_text_tokens=tokens,
        )
        if hasattr(pipe, "prepare_empty_encoding"):
            pipe.prepare_empty_encoding()

        if old_unet is not None:
            del old_unet
        gc.collect()
        mm.soft_empty_cache()

        out = dict(base_model)
        out["pipe"] = pipe
        out["backend"] = "tensorrt"
        out["engine_path"] = engine_path
        print(f"[PaGeR TensorRT] Using engine: {engine_path}")

        result = (out,)
        self._cache = {"config": config, "value": result}
        return result
