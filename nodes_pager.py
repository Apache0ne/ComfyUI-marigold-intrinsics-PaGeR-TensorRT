import gc
import math
import os
import shutil
import sys
from pathlib import Path

import torch

import comfy.model_management as mm
import folder_paths

from .nodes_shared import STORAGE_DIRNAME, _require_module, _select_torch_dtype


PAGER_MODEL_IDS = [
    "prs-eth/PaGeR-depth",
    "prs-eth/PaGeR-metric-depth",
    "prs-eth/PaGeR-depth-indoor",
    "prs-eth/PaGeR-metric-depth-indoor",
    "prs-eth/PaGeR-normals-Structured3D",
]
PAGER_UNET_WEIGHT_PATH = "unet/diffusion_pytorch_model.safetensors"

DEFAULT_MIN_DEPTH = math.log(1e-2)
DEFAULT_DEPTH_RANGE = math.log(75.0)

_BASE_REQUIRED_FILES = [
    "scheduler/scheduler_config.json",
    "vae/config.json",
]

_VAE_WEIGHT_CANDIDATES = [
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.bin",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.bin",
]

_PAGER_TAESD_PREFERRED = ["taesd", "taesdxl"]


def _pager_taesd_choices() -> list[str]:
    # Reuse split-loader discovery so choices match ComfyUI vae_approx availability.
    try:
        from .splitloader import _vae_list

        available = set(_vae_list([]))
        choices = [name for name in _PAGER_TAESD_PREFERRED if name in available]
        if choices:
            return choices
    except Exception:
        pass
    return ["taesd"]


def _comfy_vae_scaling_factor(vae) -> float:
    first_stage = getattr(vae, "first_stage_model", None)
    if first_stage is not None and hasattr(first_stage, "vae_scale"):
        try:
            scale = first_stage.vae_scale
            if isinstance(scale, torch.Tensor):
                return float(scale.detach().cpu().item())
            return float(scale)
        except Exception:
            pass
    return 0.18215


class _PaGeRComfyVAEAdapter(torch.nn.Module):
    def __init__(self, comfy_vae):
        super().__init__()
        self._vae = comfy_vae
        self.use_tiling = False
        self.use_slicing = False
        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        pixels = ((x / 2.0) + 0.5).clamp(0.0, 1.0).movedim(1, -1)
        if self.use_tiling and hasattr(self._vae, "encode_tiled"):
            latents = self._vae.encode_tiled(pixels)
        else:
            latents = self._vae.encode(pixels)
        if latents.device != x.device:
            latents = latents.to(x.device)
        return latents

    def decode(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        if self.use_tiling and hasattr(self._vae, "decode_tiled"):
            pixels = self._vae.decode_tiled(z)
        else:
            pixels = self._vae.decode(z)
        sample = pixels.movedim(-1, 1) * 2.0 - 1.0
        if sample.device != z.device:
            sample = sample.to(z.device)
        return sample

    def enable_tiling(self, enabled: bool = True):
        self.use_tiling = bool(enabled)
        return self

    def disable_tiling(self):
        self.use_tiling = False
        return self

    def enable_slicing(self):
        self.use_slicing = True
        return self

    def disable_slicing(self):
        self.use_slicing = False
        return self

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        del non_blocking
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def requires_grad_(self, requires_grad: bool):
        del requires_grad
        return self

    def eval(self):
        return self


def _ensure_pager_paths() -> Path:
    repo_root = Path(__file__).resolve().parent
    src_root = repo_root / "src"
    marigold_root = repo_root / "Marigold"
    if not src_root.exists():
        raise RuntimeError(f"PaGeR 'src' folder not found at '{src_root}'.")
    if not marigold_root.exists():
        raise RuntimeError(f"PaGeR 'Marigold' folder not found at '{marigold_root}'.")

    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)

    src_mod = sys.modules.get("src")
    if src_mod is not None:
        src_file = getattr(src_mod, "__file__", None)
        src_paths = getattr(src_mod, "__path__", None)
        owns_src = False
        if isinstance(src_file, str) and str(src_root) in src_file:
            owns_src = True
        if src_paths is not None:
            try:
                owns_src = any(str(src_root) in str(p) for p in list(src_paths))
            except Exception:
                pass
        if not owns_src:
            for key in list(sys.modules.keys()):
                if key == "src" or key.startswith("src."):
                    del sys.modules[key]

    marigold_mod = sys.modules.get("Marigold")
    if marigold_mod is not None:
        marigold_file = getattr(marigold_mod, "__file__", None)
        marigold_paths = getattr(marigold_mod, "__path__", None)
        owns_marigold = False
        if isinstance(marigold_file, str) and str(marigold_root) in marigold_file:
            owns_marigold = True
        if marigold_paths is not None:
            try:
                owns_marigold = any(str(marigold_root) in str(p) for p in list(marigold_paths))
            except Exception:
                pass
        if not owns_marigold:
            for key in list(sys.modules.keys()):
                if key == "Marigold" or key.startswith("Marigold."):
                    del sys.modules[key]

    return repo_root


def _pager_storage_root() -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME, "pager")


def _pager_repo_dir(repo_id: str) -> str:
    return os.path.join(_pager_storage_root(), repo_id.replace("/", "__"))


def _shared_base_dir(variant: str) -> str:
    return os.path.join(folder_paths.models_dir, STORAGE_DIRNAME, "base", variant)


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _ensure_pager_file(repo_id: str, remote_path: str, local_path: str) -> None:
    if os.path.exists(local_path):
        return

    _require_module("huggingface_hub")
    from huggingface_hub import hf_hub_download

    src_path = hf_hub_download(repo_id=repo_id, filename=remote_path)
    _link_or_copy(src_path, local_path)


def _has_any_file(root: str, candidates: list[str]) -> bool:
    for rel in candidates:
        if os.path.exists(os.path.join(root, rel)):
            return True
    return False


def _is_valid_shared_base(root: str) -> bool:
    for rel in _BASE_REQUIRED_FILES:
        if not os.path.exists(os.path.join(root, rel)):
            return False
    if not _has_any_file(root, _VAE_WEIGHT_CANDIDATES):
        return False
    return True


def _resolve_shared_base(precision: str) -> str | None:
    pref = (precision or "auto").lower()
    if pref == "fp32":
        order = ["fp32", "fp16"]
    else:
        order = ["fp16", "fp32"]

    for variant in order:
        base_dir = _shared_base_dir(variant)
        if _is_valid_shared_base(base_dir):
            print(f"[PaGeR Loader] using shared base variant '{variant}': {base_dir}")
            return base_dir

    return None


def _preferred_base_variant(precision: str, dtype: torch.dtype) -> str:
    pref = (precision or "auto").lower()
    if pref == "fp32":
        return "fp32"
    if pref in ("fp16", "bf16"):
        return "fp16"
    return "fp32" if dtype == torch.float32 else "fp16"


def _ensure_pager_base_files(model_variant: str, repo_id_source: str) -> str:
    if model_variant not in ("fp16", "fp32"):
        raise ValueError(f"Invalid model_variant={model_variant!r}")

    base = _shared_base_dir(model_variant)
    files = [
        ("scheduler/scheduler_config.json", "scheduler/scheduler_config.json"),
        ("vae/config.json", "vae/config.json"),
    ]
    if model_variant == "fp16":
        files.append(("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"))
    else:
        files.append(("vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"))

    for remote_path, rel_dst in files:
        _ensure_pager_file(repo_id_source, remote_path, os.path.join(base, rel_dst))

    return base


def _ensure_pager_model_files(model_repo: str) -> str:
    model_dir = _pager_repo_dir(model_repo)
    files = [
        ("config.yaml", "config.yaml"),
        ("unet/config.json", "unet/config.json"),
        (PAGER_UNET_WEIGHT_PATH, PAGER_UNET_WEIGHT_PATH),
    ]

    for remote_path, rel_dst in files:
        _ensure_pager_file(model_repo, remote_path, os.path.join(model_dir, rel_dst))

    return model_dir


def _load_pager_config(local_repo_dir: str):
    cfg_path = os.path.join(local_repo_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise RuntimeError(f"Missing config.yaml in '{local_repo_dir}'.")

    _require_module("omegaconf")
    from omegaconf import OmegaConf

    return OmegaConf.load(cfg_path)


def _infer_modality(cfg, repo_id: str) -> str:
    modality = ""
    try:
        modality = str(cfg.model.modality).lower()
    except Exception:
        modality = ""

    if modality.startswith("depth"):
        return "depth"
    if "normal" in modality:
        return "normal"
    if "normal" in repo_id.lower():
        return "normal"
    if "depth" in repo_id.lower():
        return "depth"

    raise RuntimeError(f"Could not infer modality from config or repo_id='{repo_id}'.")


def _pager_to(pager, device: torch.device, dtype: torch.dtype) -> None:
    pager.device = device
    pager.weight_dtype = dtype
    if hasattr(pager, "vae"):
        pager.vae.to(device=device, dtype=dtype)
    if hasattr(pager, "unet"):
        for unet in pager.unet.values():
            unet.to(device=device, dtype=dtype)
    if hasattr(pager, "empty_encoding"):
        pager.empty_encoding = pager.empty_encoding.to(device=device, dtype=dtype)
    if hasattr(pager, "alpha_prod"):
        pager.alpha_prod = pager.alpha_prod.to(device=device, dtype=dtype)
    if hasattr(pager, "beta_prod"):
        pager.beta_prod = pager.beta_prod.to(device=device, dtype=dtype)
    if hasattr(pager, "PE_cubemap"):
        pager.PE_cubemap = pager.PE_cubemap.to(device=device, dtype=dtype)


class _PaGeRBaseLoader:
    def __init__(self):
        self._cache = None

    def load(
        self,
        model_id: str,
        precision: str,
        keep_on_gpu: bool = True,
        vae_tiling: bool = False,
        use_comfy_taesd_vae: bool = False,
        taesd_vae_name: str = "taesd",
        force_gpu: bool = True,
    ):
        _require_module("diffusers")
        _require_module("omegaconf")
        _require_module("pytorch360convert")
        _require_module("huggingface_hub")
        _require_module("einops")

        _ensure_pager_paths()
        from src.pager import Pager

        device = mm.get_torch_device()
        if force_gpu and torch.cuda.is_available():
            device = torch.device("cuda")

        dtype = _select_torch_dtype(precision, device)

        config = {
            "model_id": str(model_id),
            "precision": str(precision),
            "dtype": str(dtype),
            "keep_on_gpu": bool(keep_on_gpu),
            "vae_tiling": bool(vae_tiling),
            "use_comfy_taesd_vae": bool(use_comfy_taesd_vae),
            "taesd_vae_name": str(taesd_vae_name),
            "force_gpu": bool(force_gpu),
            "device": str(device),
        }
        if self._cache is not None and self._cache.get("config") == config:
            return self._cache["value"]

        # Free/offload previous cached model when switching configs.
        if self._cache is not None:
            old_value = self._cache.get("value", None)
            try:
                if isinstance(old_value, tuple) and len(old_value) > 0 and isinstance(old_value[0], dict):
                    old_pager_model = old_value[0]
                    old_pipe = old_pager_model.get("pipe", None)
                    old_dtype = old_pager_model.get("dtype", torch.float32)
                    if old_pipe is not None:
                        _pager_to(old_pipe, mm.unet_offload_device(), old_dtype)
            except Exception as e:
                print(f"[PaGeR Loader] Warning: failed to offload previous cached model: {e}")
            self._cache = None
            gc.collect()
            mm.soft_empty_cache()

        local_model_dir = _ensure_pager_model_files(model_id)
        cfg = _load_pager_config(local_model_dir)

        pretrained_repo = str(cfg.model.pretrained_path)
        shared_base_dir = _resolve_shared_base(precision)
        if shared_base_dir is not None:
            local_pretrained_dir = shared_base_dir
        else:
            target_variant = _preferred_base_variant(precision, dtype)
            print(
                f"[PaGeR Loader] shared base missing; downloading into shared base/{target_variant} from {pretrained_repo}"
            )
            try:
                local_pretrained_dir = _ensure_pager_base_files(target_variant, pretrained_repo)
            except Exception as e:
                if target_variant == "fp16":
                    print(
                        f"[PaGeR Loader] fp16 base download failed ({e}); retrying shared base/fp32 from {pretrained_repo}"
                    )
                    local_pretrained_dir = _ensure_pager_base_files("fp32", pretrained_repo)
                else:
                    raise

        modality = _infer_modality(cfg, model_id)
        model_configs = {
            modality: {
                "path": local_model_dir,
                "mode": "trained",
                "config": cfg.model,
            }
        }

        pager = Pager(
            model_configs=model_configs,
            pretrained_path=local_pretrained_dir,
            device=device,
            weight_dtype=dtype,
        )

        if use_comfy_taesd_vae:
            try:
                from .splitloader import _load_vae_by_name

                comfy_vae = _load_vae_by_name(str(taesd_vae_name), [])
                latent_channels = int(getattr(comfy_vae, "latent_channels", 4))
                if latent_channels != 4:
                    raise RuntimeError(
                        f"Selected TAESD '{taesd_vae_name}' has latent_channels={latent_channels}, expected 4."
                    )
                pager.vae = _PaGeRComfyVAEAdapter(comfy_vae)
                vae_scale = _comfy_vae_scaling_factor(comfy_vae)
                pager.rgb_latent_scale_factor = float(vae_scale)
                pager.depth_latent_scale_factor = float(vae_scale)
                print(
                    f"[PaGeR Loader] Using ComfyUI TAESD VAE '{taesd_vae_name}' "
                    f"(latent_scale={vae_scale:.5f})."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load ComfyUI TAESD VAE '{taesd_vae_name}': {e}") from e

        _pager_to(pager, device, dtype)

        if hasattr(pager, "vae"):
            try:
                # PaGeR uses custom cubemap padding that requires face-batch=6.
                pager.vae.disable_slicing()
                print("[PaGeR Loader] VAE slicing: disabled (required by PaGeR)")
            except Exception as e:
                print(f"[PaGeR Loader] Warning: could not disable VAE slicing: {e}")
            try:
                if vae_tiling:
                    pager.vae.enable_tiling(True)
                else:
                    pager.vae.disable_tiling()
                print(f"[PaGeR Loader] VAE tiling: {'enabled' if vae_tiling else 'disabled'}")
            except Exception as e:
                print(f"[PaGeR Loader] Warning: could not set VAE tiling={vae_tiling}: {e}")

        if modality in pager.unet:
            pager.unet[modality].eval()

        pager_model = {
            "pipe": pager,
            "dtype": dtype,
            "autocast_dtype": dtype,
            "modality": modality,
            "log_scale": bool(getattr(cfg.model, "log_scale", False)),
            "metric_depth": bool(getattr(cfg.model, "metric_depth", False)),
            "repo_id": str(model_id),
            "model_dir": local_model_dir,
            "pretrained_path": pretrained_repo,
            "keep_on_gpu": bool(keep_on_gpu),
            "vae_tiling": bool(vae_tiling),
            "use_comfy_taesd_vae": bool(use_comfy_taesd_vae),
            "taesd_vae_name": str(taesd_vae_name),
            "force_gpu": bool(force_gpu),
            "min_depth": DEFAULT_MIN_DEPTH,
            "depth_range": DEFAULT_DEPTH_RANGE,
        }

        result = (pager_model,)
        self._cache = {"config": config, "value": result}
        return result


_PAGER_LOADER = _PaGeRBaseLoader()


def load_pager_base_model(
    model_id: str,
    precision: str,
    keep_on_gpu: bool = True,
    vae_tiling: bool = False,
    use_comfy_taesd_vae: bool = False,
    taesd_vae_name: str = "taesd",
    force_gpu: bool = True,
):
    return _PAGER_LOADER.load(
        model_id=model_id,
        precision=precision,
        keep_on_gpu=keep_on_gpu,
        vae_tiling=vae_tiling,
        use_comfy_taesd_vae=use_comfy_taesd_vae,
        taesd_vae_name=taesd_vae_name,
        force_gpu=force_gpu,
    )
