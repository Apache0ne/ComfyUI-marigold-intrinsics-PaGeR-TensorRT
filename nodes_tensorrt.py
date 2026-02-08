import os
import re
import json
import io
import logging
import warnings
import contextlib
from types import SimpleNamespace

import torch

import comfy.model_management as mm

from .nodes_pager import PAGER_MODEL_IDS, _ensure_pager_model_files
from .nodes_shared import (
    MODEL_REPO_ID_APPEARANCE,
    MODEL_REPO_ID_LIGHTING,
    _ensure_model_files,
    _require_module,
)

_MARIGOLD_TARGETS = {
    "marigold:lighting": (MODEL_REPO_ID_LIGHTING, "lighting"),
    "marigold:appearance": (MODEL_REPO_ID_APPEARANCE, "appearance"),
}

_MODEL_CHOICES = list(_MARIGOLD_TARGETS.keys()) + list(PAGER_MODEL_IDS)


def _sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    clean = clean.strip("._-")
    return clean or "unet"


def _trt_dtype_to_torch(dtype):
    # TensorRT enum names are stable across versions.
    name = str(dtype).split(".")[-1].upper()
    mapping = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "BF16": torch.bfloat16,
        "INT32": torch.int32,
        "INT8": torch.int8,
        "BOOL": torch.bool,
    }
    if name not in mapping:
        raise RuntimeError(f"Unsupported TensorRT dtype: {dtype!r}")
    return mapping[name]


def _load_unet_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "unet", "config.json")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"UNet config not found at '{cfg_path}'.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Invalid UNet config in '{cfg_path}'.")
    return cfg


class _TRTUNetOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class TensorRTUNetWrapper(torch.nn.Module):
    """
    Minimal UNet-compatible wrapper that runs a TensorRT .engine.
    Expected tensor names:
      - sample
      - timestep
      - encoder_hidden_states
      - noise_pred (output)
    """

    def __init__(
        self,
        engine_path: str,
        unet_config: dict,
        default_text_tokens: int = 2,
    ):
        super().__init__()
        _require_module("tensorrt", "pip install -r custom_nodes/ComfyUI-marigold-intrinsics-PaGeR-TensorRT/requirements.txt")

        self.engine_path = os.path.abspath(str(engine_path))
        if not os.path.isfile(self.engine_path):
            raise RuntimeError(f"TensorRT engine not found at '{self.engine_path}'.")

        cfg = dict(unet_config or {})
        if "cross_attention_dim" not in cfg:
            raise RuntimeError("UNet config is missing 'cross_attention_dim'.")
        self.config = SimpleNamespace(**cfg)
        self.default_text_tokens = int(default_text_tokens)

        # Anchor buffer to keep this as a normal nn.Module that supports .to().
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

        self._engine = None
        self._context = None
        self._runtime = None
        self._trt = None
        self._trt_logger = None
        self._use_v3 = False
        self._input_names = []
        self._output_names = []
        self._name_to_index = {}
        self._output_name = "noise_pred"
        self._load_engine()

    @classmethod
    def from_model_dir(
        cls,
        engine_path: str,
        model_dir: str,
        default_text_tokens: int = 2,
    ):
        return cls(
            engine_path=engine_path,
            unet_config=_load_unet_config(model_dir),
            default_text_tokens=int(default_text_tokens),
        )

    def _load_engine(self) -> None:
        import tensorrt as trt

        self._trt = trt
        self._trt_logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine '{self.engine_path}'.")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context for '{self.engine_path}'.")

        self._use_v3 = bool(hasattr(self._engine, "num_io_tensors") and hasattr(self._context, "execute_async_v3"))
        self._input_names = []
        self._output_names = []
        self._name_to_index = {}

        if self._use_v3:
            io_count = int(self._engine.num_io_tensors)
            for i in range(io_count):
                name = self._engine.get_tensor_name(i)
                self._name_to_index[name] = i
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_names.append(name)
                else:
                    self._output_names.append(name)
        else:
            num_bindings = int(self._engine.num_bindings)
            for i in range(num_bindings):
                name = self._engine.get_binding_name(i)
                self._name_to_index[name] = i
                if self._engine.binding_is_input(i):
                    self._input_names.append(name)
                else:
                    self._output_names.append(name)

        required = {"sample", "timestep", "encoder_hidden_states"}
        missing = sorted(required - set(self._input_names))
        if missing:
            raise RuntimeError(
                f"TensorRT engine is missing required inputs {missing}. "
                f"Available inputs: {self._input_names}"
            )

        if "noise_pred" in self._output_names:
            self._output_name = "noise_pred"
        elif len(self._output_names) == 1:
            self._output_name = self._output_names[0]
        else:
            raise RuntimeError(
                f"TensorRT engine must expose output 'noise_pred' (or a single output). "
                f"Available outputs: {self._output_names}"
            )

    def _tensor_dtype(self, name: str) -> torch.dtype:
        if self._use_v3:
            dtype = self._engine.get_tensor_dtype(name)
        else:
            dtype = self._engine.get_binding_dtype(self._name_to_index[name])
        return _trt_dtype_to_torch(dtype)

    def _normalize_timestep(self, timestep, batch_size: int, device: torch.device) -> torch.Tensor:
        target_dtype = self._tensor_dtype("timestep")
        if torch.is_tensor(timestep):
            t = timestep.to(device=device, dtype=target_dtype)
        else:
            t = torch.tensor([float(timestep)], device=device, dtype=target_dtype)
        if t.ndim == 0:
            t = t.reshape(1)
        if t.ndim != 1:
            t = t.reshape(-1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.repeat(batch_size)
        if t.shape[0] != batch_size:
            raise RuntimeError(f"timestep batch mismatch: got {int(t.shape[0])}, expected {batch_size}.")
        return t.contiguous()

    def _normalize_encoder_hidden_states(self, encoder_hidden_states, batch_size: int, device: torch.device) -> torch.Tensor:
        target_dtype = self._tensor_dtype("encoder_hidden_states")
        if not torch.is_tensor(encoder_hidden_states):
            raise RuntimeError("encoder_hidden_states must be a torch.Tensor.")
        e = encoder_hidden_states.to(device=device, dtype=target_dtype)
        if e.ndim != 3:
            raise RuntimeError(f"encoder_hidden_states must be [B,T,C], got shape={tuple(e.shape)}")
        if e.shape[0] == 1 and batch_size > 1:
            e = e.repeat(batch_size, 1, 1)
        if e.shape[0] != batch_size:
            raise RuntimeError(
                f"encoder_hidden_states batch mismatch: got {int(e.shape[0])}, expected {batch_size}."
            )
        return e.contiguous()

    @staticmethod
    def _dims_to_tuple(dims):
        try:
            return tuple(int(x) for x in dims)
        except Exception:
            return None

    def _profile_shape_bounds(self, name: str):
        # Returns (min_shape, opt_shape, max_shape) or (None, None, None).
        try:
            if hasattr(self._engine, "get_tensor_profile_shape"):
                lo, opt, hi = self._engine.get_tensor_profile_shape(name, 0)
                lo_t = self._dims_to_tuple(lo)
                opt_t = self._dims_to_tuple(opt)
                hi_t = self._dims_to_tuple(hi)
                if lo_t is not None and hi_t is not None:
                    return lo_t, opt_t, hi_t
        except Exception:
            pass

        try:
            if hasattr(self._engine, "get_profile_shape"):
                idx = self._name_to_index.get(name, None)
                if idx is None and hasattr(self._engine, "get_binding_index"):
                    idx = int(self._engine.get_binding_index(name))
                if idx is not None and int(idx) >= 0:
                    lo, opt, hi = self._engine.get_profile_shape(0, int(idx))
                    lo_t = self._dims_to_tuple(lo)
                    opt_t = self._dims_to_tuple(opt)
                    hi_t = self._dims_to_tuple(hi)
                    if lo_t is not None and hi_t is not None:
                        return lo_t, opt_t, hi_t
        except Exception:
            pass

        return None, None, None

    def _sample_profile_error(self, sample_shape: tuple[int, ...], lo: tuple[int, ...], hi: tuple[int, ...]) -> str:
        b, c, h, w = (int(sample_shape[0]), int(sample_shape[1]), int(sample_shape[2]), int(sample_shape[3]))
        min_b, min_c, min_h, min_w = lo
        max_b, max_c, max_h, max_w = hi

        # Convert latent profile to user-facing image resolution (x8 for SD/Marigold UNet latent scale).
        img_mul = 8
        img_h = h * img_mul
        img_w = w * img_mul
        min_img_h = min_h * img_mul
        max_img_h = max_h * img_mul
        min_img_w = min_w * img_mul
        max_img_w = max_w * img_mul

        parts = ["TensorRT engine resolution mismatch."]

        if (h < min_h or h > max_h) or (w < min_w or w > max_w):
            parts.append(
                f"Engine supports image resolution H {min_img_h}-{max_img_h}, W {min_img_w}-{max_img_w}."
            )
            parts.append(
                f"Current image resolution is H {img_h}, W {img_w}."
            )
            parts.append("Set the node resolution so the image size falls in this range.")

        if b < min_b or b > max_b:
            parts.append(f"Batch size must be {min_b}-{max_b}; current batch is {b}.")

        if c < min_c or c > max_c:
            parts.append("Selected TRT engine does not match this model (channel mismatch).")

        if len(parts) == 1:
            parts.append("Input shape is outside this engine profile.")

        return " ".join(parts)

    def _validate_sample_for_profile(self, sample: torch.Tensor) -> None:
        lo, _, hi = self._profile_shape_bounds("sample")
        if lo is None or hi is None or len(lo) != 4 or len(hi) != 4:
            return

        b, c, h, w = (int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2]), int(sample.shape[3]))
        min_b, min_c, min_h, min_w = lo
        max_b, max_c, max_h, max_w = hi

        if (b < min_b or b > max_b) or (c < min_c or c > max_c) or (h < min_h or h > max_h) or (w < min_w or w > max_w):
            raise RuntimeError(self._sample_profile_error((b, c, h, w), lo, hi))

    def _set_input_shape_v3(self, name: str, shape: tuple[int, ...]) -> None:
        # TensorRT may return None (older APIs) or bool.
        result = self._context.set_input_shape(name, tuple(int(x) for x in shape))
        if isinstance(result, bool) and not result:
            lo, _, hi = self._profile_shape_bounds(name)
            profile = f" Valid profile range: {lo}..{hi}." if lo is not None and hi is not None else ""
            raise RuntimeError(f"Failed to set TensorRT input shape for '{name}' to {shape}.{profile}")

    def _set_input_shape_v2(self, name: str, shape: tuple[int, ...]) -> None:
        idx = self._name_to_index[name]
        result = self._context.set_binding_shape(idx, tuple(int(x) for x in shape))
        if isinstance(result, bool) and not result:
            lo, _, hi = self._profile_shape_bounds(name)
            profile = f" Valid profile range: {lo}..{hi}." if lo is not None and hi is not None else ""
            raise RuntimeError(f"Failed to set TensorRT binding shape for '{name}' to {shape}.{profile}")

    def _execute_v3(self, tensors: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        self._set_input_shape_v3("sample", tuple(tensors["sample"].shape))
        self._set_input_shape_v3("timestep", tuple(tensors["timestep"].shape))
        self._set_input_shape_v3("encoder_hidden_states", tuple(tensors["encoder_hidden_states"].shape))

        output_shape = tuple(int(x) for x in self._context.get_tensor_shape(self._output_name))
        if any(int(x) < 0 for x in output_shape):
            raise RuntimeError(f"Unresolved TensorRT output shape for '{self._output_name}': {output_shape}")
        output = torch.empty(
            output_shape,
            device=device,
            dtype=self._tensor_dtype(self._output_name),
        )

        all_tensors = dict(tensors)
        all_tensors[self._output_name] = output
        for name, tensor in all_tensors.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream(device=device)
        ok = self._context.execute_async_v3(stream_handle=int(stream.cuda_stream))
        if isinstance(ok, bool) and not ok:
            raise RuntimeError("TensorRT execution failed (execute_async_v3).")
        return output

    def _execute_v2(self, tensors: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        self._set_input_shape_v2("sample", tuple(tensors["sample"].shape))
        self._set_input_shape_v2("timestep", tuple(tensors["timestep"].shape))
        self._set_input_shape_v2("encoder_hidden_states", tuple(tensors["encoder_hidden_states"].shape))

        out_idx = self._name_to_index[self._output_name]
        output_shape = tuple(int(x) for x in self._context.get_binding_shape(out_idx))
        if any(int(x) < 0 for x in output_shape):
            raise RuntimeError(f"Unresolved TensorRT output shape for '{self._output_name}': {output_shape}")

        output = torch.empty(
            output_shape,
            device=device,
            dtype=self._tensor_dtype(self._output_name),
        )

        bindings = [0] * int(self._engine.num_bindings)
        for name, tensor in tensors.items():
            bindings[self._name_to_index[name]] = int(tensor.data_ptr())
        bindings[out_idx] = int(output.data_ptr())

        stream = torch.cuda.current_stream(device=device)
        ok = self._context.execute_async_v2(bindings=bindings, stream_handle=int(stream.cuda_stream))
        if isinstance(ok, bool) and not ok:
            raise RuntimeError("TensorRT execution failed (execute_async_v2).")
        return output

    def forward(self, sample, timestep, encoder_hidden_states, return_dict: bool = True, **_kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT UNet requires CUDA.")
        if not torch.is_tensor(sample):
            raise RuntimeError("sample must be a torch.Tensor.")
        if not sample.is_cuda:
            raise RuntimeError("TensorRT UNet input 'sample' must be on CUDA.")
        if sample.ndim != 4:
            raise RuntimeError(f"sample must be [B,C,H,W], got shape={tuple(sample.shape)}")

        device = sample.device
        sample = sample.to(device=device, dtype=self._tensor_dtype("sample")).contiguous()
        self._validate_sample_for_profile(sample)
        batch_size = int(sample.shape[0])
        timestep = self._normalize_timestep(timestep, batch_size, device)
        encoder_hidden_states = self._normalize_encoder_hidden_states(encoder_hidden_states, batch_size, device)

        tensors = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        if self._use_v3:
            noise_pred = self._execute_v3(tensors, device)
        else:
            noise_pred = self._execute_v2(tensors, device)

        if return_dict:
            return _TRTUNetOutput(noise_pred)
        return (noise_pred,)

    @property
    def dtype(self):
        return self._anchor.dtype

    @property
    def device(self):
        return self._anchor.device

    # Compatibility no-ops for code paths that assume diffusers UNet hooks exist.
    def set_attn_processor(self, *_args, **_kwargs):
        return None

    def set_attention_processor(self, *_args, **_kwargs):
        return None

    def enable_xformers_memory_efficient_attention(self, *_args, **_kwargs):
        return None

    def disable_xformers_memory_efficient_attention(self, *_args, **_kwargs):
        return None


def _cross_attention_dim(unet) -> int:
    dim = getattr(getattr(unet, "config", None), "cross_attention_dim", None)
    if isinstance(dim, int) and dim > 0:
        return int(dim)
    if isinstance(dim, (tuple, list)):
        for item in dim:
            if isinstance(item, int) and item > 0:
                return int(item)
    raise RuntimeError(f"Could not resolve UNet cross_attention_dim from config value: {dim!r}")


def _set_standard_attention_processor(unet) -> None:
    from diffusers.models.attention_processor import AttnProcessor

    if hasattr(unet, "set_attn_processor"):
        unet.set_attn_processor(AttnProcessor())
    elif hasattr(unet, "set_attention_processor"):
        unet.set_attention_processor(AttnProcessor())


def _get_input(network, name: str):
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name == name:
            return inp
    raise ValueError(f"Missing ONNX input '{name}'.")


@contextlib.contextmanager
def _suppress_onnx_export_noise():
    # Suppress verbose ONNX export chatter (schema warnings, graph rewrite logs, etc.)
    sink = io.StringIO()
    logger_names = [
        "torch.onnx",
        "torch.onnx._internal",
        "torch.fx.experimental.symbolic_shapes",
        "onnx",
        "onnx_ir",
        "onnx_ir.passes",
        "onnxscript",
        "onnxscript.ir",
        "onnxscript.optimizer",
        "onnxscript._legacy_ir",
    ]
    saved_levels = []
    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            saved_levels.append((logger, logger.level))
            logger.setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"torch\.onnx(\.|$)")
            warnings.filterwarnings("ignore", module=r"torch\.fx\.experimental\.symbolic_shapes(\.|$)")
            warnings.filterwarnings("ignore", module=r"onnx(\.|$)")
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    finally:
        for logger, level in saved_levels:
            logger.setLevel(level)


def _export_unet_to_onnx(
    model_dir: str,
    onnx_path: str,
    device: torch.device,
    dtype: torch.dtype,
    opset: int,
    opt_resolution: int,
    latent_divisor: int,
) -> None:
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        model_dir,
        subfolder="unet",
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )
    _set_standard_attention_processor(unet)
    unet.eval().to(device)

    in_channels = int(unet.config.in_channels)
    cross_dim = _cross_attention_dim(unet)
    latent_size = max(1, int(opt_resolution) // int(latent_divisor))

    sample = torch.randn(1, in_channels, latent_size, latent_size, device=device, dtype=dtype)
    timestep = torch.tensor([1.0], device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(1, 2, cross_dim, device=device, dtype=dtype)

    class _Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.model(sample, timestep, encoder_hidden_states, return_dict=False)[0]

    wrapper = _Wrapper(unet).eval()

    dynamic_axes = {
        "sample": {0: "batch", 2: "height", 3: "width"},
        "timestep": {0: "batch"},
        "encoder_hidden_states": {0: "batch"},
        "noise_pred": {0: "batch", 2: "height", 3: "width"},
    }

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    export_kwargs = {
        "f": onnx_path,
        "opset_version": int(opset),
        "input_names": ["sample", "timestep", "encoder_hidden_states"],
        "output_names": ["noise_pred"],
        "dynamic_axes": dynamic_axes,
        "do_constant_folding": False,
        "optimize": False,
        "verbose": False,
        "external_data": True,
    }

    with torch.no_grad():
        try:
            with _suppress_onnx_export_noise():
                torch.onnx.export(wrapper, (sample, timestep, encoder_hidden_states), **export_kwargs)
        except TypeError:
            export_kwargs.pop("external_data", None)
            export_kwargs.pop("optimize", None)
            export_kwargs.pop("verbose", None)
            with _suppress_onnx_export_noise():
                torch.onnx.export(wrapper, (sample, timestep, encoder_hidden_states), **export_kwargs)


def _build_trt_engine(
    onnx_path: str,
    engine_path: str,
    enable_fp16: bool,
    workspace_gb: float,
    latent_divisor: int,
    min_side_res: int,
    opt_res: int,
    max_res: int,
    batch_min: int,
    batch_opt: int,
    batch_max: int,
) -> None:
    import tensorrt as trt

    if not (batch_min <= batch_opt <= batch_max):
        raise ValueError("Batch profile must satisfy batch_min <= batch_opt <= batch_max.")

    min_lat = max(1, int(min_side_res) // int(latent_divisor))
    opt_lat = max(1, int(opt_res) // int(latent_divisor))
    max_lat = max(1, int(max_res) // int(latent_divisor))
    if not (min_lat <= opt_lat <= max_lat):
        raise ValueError("Resolution profile must satisfy min <= opt <= max after latent downscale.")

    trt_logger = trt.Logger(trt.Logger.INFO)
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with (
        trt.Builder(trt_logger) as builder,
        builder.create_network(explicit_batch_flag) as network,
        trt.OnnxParser(network, trt_logger) as parser,
    ):
        parsed_ok = False
        # Prefer parsing from file path so TensorRT can resolve external ONNX data
        # (e.g. `<model>.onnx.data`) relative to the ONNX file location.
        if hasattr(parser, "parse_from_file"):
            try:
                parsed_ok = bool(parser.parse_from_file(str(onnx_path)))
            except Exception:
                parsed_ok = False

        if not parsed_ok:
            with open(onnx_path, "rb") as f:
                parsed_ok = bool(parser.parse(f.read()))

        if not parsed_ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            joined = "\n".join(errors) if errors else "Unknown parser error."
            raise RuntimeError(f"TensorRT failed to parse ONNX:\n{joined}")

        sample_inp = _get_input(network, "sample")
        _get_input(network, "timestep")
        text_inp = _get_input(network, "encoder_hidden_states")

        in_channels = int(sample_inp.shape[1])
        seq_len = int(text_inp.shape[1])
        hidden = int(text_inp.shape[2])
        if in_channels <= 0 or seq_len <= 0 or hidden <= 0:
            raise RuntimeError(
                "Could not infer static channel dimensions from ONNX inputs. "
                f"sample={tuple(sample_inp.shape)} text={tuple(text_inp.shape)}"
            )

        config = builder.create_builder_config()
        workspace_bytes = int(max(1.0, float(workspace_gb)) * (1 << 30))
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes

        if enable_fp16:
            if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
                print("[TensorRT Loader] FP16 requested but not supported by platform. Building with default precision.")
            else:
                config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "sample",
            (int(batch_min), in_channels, min_lat, min_lat),
            (int(batch_opt), in_channels, opt_lat, opt_lat),
            (int(batch_max), in_channels, max_lat, max_lat),
        )
        profile.set_shape(
            "encoder_hidden_states",
            (int(batch_min), seq_len, hidden),
            (int(batch_opt), seq_len, hidden),
            (int(batch_max), seq_len, hidden),
        )
        profile.set_shape(
            "timestep",
            (int(batch_min),),
            (int(batch_opt),),
            (int(batch_max),),
        )
        config.add_optimization_profile(profile)

        engine_bytes = None
        if hasattr(builder, "build_serialized_network"):
            engine_bytes = builder.build_serialized_network(network, config)

        if engine_bytes is None and hasattr(builder, "build_engine"):
            engine = builder.build_engine(network, config)
            if engine is not None:
                engine_bytes = engine.serialize()

        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed.")

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(bytes(engine_bytes))


class LoadUNetTensorRTEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_MODEL_CHOICES, {"default": "marigold:lighting"}),
                "marigold_variant": (["fp16", "fp32"], {"default": "fp16"}),
                "engine_precision": (["fp16", "fp32"], {"default": "fp16"}),
                "profile_min_side_resolution": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 1}),
                "profile_opt_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 1}),
                "profile_max_resolution": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 1}),
                "latent_divisor": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "batch_min": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "batch_opt": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "batch_max": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "workspace_gb": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 64.0, "step": 0.5}),
                "opset": ("INT", {"default": 18, "min": 13, "max": 20, "step": 1}),
                "output_folder_name": ("STRING", {"default": "trt"}),
                "filename_suffix": ("STRING", {"default": ""}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "keep_onnx": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("engine_path",)
    FUNCTION = "build"
    CATEGORY = "TensorRT/Loaders"

    DESCRIPTION = """
Builds a TensorRT engine file (.engine) from a selected Marigold or PaGeR UNet.
Downloads model files if missing and returns only the saved engine path.
"""

    def build(
        self,
        model_name: str,
        marigold_variant: str = "fp16",
        engine_precision: str = "fp16",
        profile_min_side_resolution: int = 384,
        profile_opt_resolution: int = 768,
        profile_max_resolution: int = 1536,
        latent_divisor: int = 8,
        batch_min: int = 1,
        batch_opt: int = 1,
        batch_max: int = 4,
        workspace_gb: float = 8.0,
        opset: int = 18,
        output_folder_name: str = "trt",
        filename_suffix: str = "",
        overwrite_existing: bool = False,
        keep_onnx: bool = False,
    ):
        _require_module("diffusers", "pip install -r custom_nodes/ComfyUI-marigold-intrinsics-PaGeR-TensorRT/requirements.txt")
        _require_module("onnx", "pip install -r custom_nodes/ComfyUI-marigold-intrinsics-PaGeR-TensorRT/requirements.txt")
        _require_module("tensorrt", "pip install -r custom_nodes/ComfyUI-marigold-intrinsics-PaGeR-TensorRT/requirements.txt")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to export and build TensorRT engines.")

        if model_name in _MARIGOLD_TARGETS:
            repo_id, kind = _MARIGOLD_TARGETS[model_name]
            model_dir = _ensure_model_files(repo_id, kind, marigold_variant)
        else:
            if model_name not in PAGER_MODEL_IDS:
                raise ValueError(f"Unknown model_name={model_name!r}")
            repo_id = model_name
            model_dir = _ensure_pager_model_files(repo_id)

        unet_dir = os.path.join(model_dir, "unet")
        if not os.path.isdir(unet_dir):
            raise RuntimeError(f"UNet folder not found at '{unet_dir}'.")

        out_folder = str(output_folder_name).strip() or "trt"
        out_root = os.path.join(model_dir, out_folder)
        suffix = _sanitize_name(filename_suffix) if str(filename_suffix).strip() else ""
        precision_suffix = _sanitize_name(engine_precision)
        if model_name in _MARIGOLD_TARGETS:
            _, kind = _MARIGOLD_TARGETS[model_name]
            basename = f"marigold_{_sanitize_name(kind)}_unet_{precision_suffix}"
        else:
            basename = f"pager_{_sanitize_name(model_name.replace('/', '_'))}_unet_{precision_suffix}"
        if suffix:
            basename = f"{basename}_{suffix}"

        engine_path = os.path.abspath(os.path.join(out_root, f"{basename}.engine"))
        onnx_path = os.path.abspath(os.path.join(out_root, f"{basename}.onnx"))

        if os.path.exists(engine_path) and not bool(overwrite_existing):
            print(f"[TensorRT Loader] Existing engine found, skipping build: {engine_path}")
            return (engine_path,)

        device = mm.get_torch_device()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            device = torch.device("cuda")

        export_dtype = torch.float16 if engine_precision == "fp16" else torch.float32
        print(f"[TensorRT Loader] Exporting ONNX for '{model_name}' from '{model_dir}'")
        _export_unet_to_onnx(
            model_dir=model_dir,
            onnx_path=onnx_path,
            device=device,
            dtype=export_dtype,
            opset=int(opset),
            opt_resolution=int(profile_opt_resolution),
            latent_divisor=int(latent_divisor),
        )

        print(f"[TensorRT Loader] Building engine: {engine_path}")
        _build_trt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            enable_fp16=(engine_precision == "fp16"),
            workspace_gb=float(workspace_gb),
            latent_divisor=int(latent_divisor),
            min_side_res=int(profile_min_side_resolution),
            opt_res=int(profile_opt_resolution),
            max_res=int(profile_max_resolution),
            batch_min=int(batch_min),
            batch_opt=int(batch_opt),
            batch_max=int(batch_max),
        )

        if not bool(keep_onnx):
            for path in (onnx_path, f"{onnx_path}.data", f"{onnx_path}_data"):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"[TensorRT Loader] Warning: failed to remove '{path}': {e}")

        print(f"[TensorRT Loader] Engine ready: {engine_path}")
        return (engine_path,)
