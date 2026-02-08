import os

import torch

import comfy.sd
import comfy.utils
import folder_paths


def _vae_list(video_taes: list[str]) -> list[str]:
    vaes = folder_paths.get_filename_list("vae")
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    sdxl_taesd_enc = False
    sdxl_taesd_dec = False
    sd1_taesd_enc = False
    sd1_taesd_dec = False
    sd3_taesd_enc = False
    sd3_taesd_dec = False
    f1_taesd_enc = False
    f1_taesd_dec = False

    f2_has_safetensors = False
    f2_has_enc = False
    f2_has_dec = False

    for v in approx_vaes:
        if v.startswith("taesd_decoder."):
            sd1_taesd_dec = True
        elif v.startswith("taesd_encoder."):
            sd1_taesd_enc = True
        elif v.startswith("taesdxl_decoder."):
            sdxl_taesd_dec = True
        elif v.startswith("taesdxl_encoder."):
            sdxl_taesd_enc = True
        elif v.startswith("taesd3_decoder."):
            sd3_taesd_dec = True
        elif v.startswith("taesd3_encoder."):
            sd3_taesd_enc = True
        elif v.startswith("taef1_encoder."):
            f1_taesd_enc = True
        elif v.startswith("taef1_decoder."):
            f1_taesd_dec = True
        elif v == "taef2.safetensors":
            f2_has_safetensors = True
        elif v.startswith("taef2_encoder."):
            f2_has_enc = True
        elif v.startswith("taef2_decoder."):
            f2_has_dec = True
        else:
            for tae in video_taes:
                if v.startswith(tae):
                    vaes.append(v)

    if sd1_taesd_dec and sd1_taesd_enc:
        vaes.append("taesd")
    if sdxl_taesd_dec and sdxl_taesd_enc:
        vaes.append("taesdxl")
    if sd3_taesd_dec and sd3_taesd_enc:
        vaes.append("taesd3")
    if f1_taesd_dec and f1_taesd_enc:
        vaes.append("taef1")

    if f2_has_safetensors or (f2_has_enc and f2_has_dec):
        vaes.append("taef2")

    vaes.append("pixel_space")
    return vaes


def _load_taesd(name: str) -> dict:
    sd: dict = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    try:
        encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
        decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))
    except StopIteration as e:
        raise RuntimeError(f"Could not find TAESD encoder/decoder for '{name}' in models/vae_approx") from e

    enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
    for k in enc:
        sd[f"taesd_encoder.{k}"] = enc[k]

    dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
    for k in dec:
        sd[f"taesd_decoder.{k}"] = dec[k]

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0609)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)

    return sd


def _load_vae_by_name(vae_name: str, video_taes: list[str]) -> comfy.sd.VAE:
    metadata = None

    if vae_name == "pixel_space":
        sd = {"pixel_space_vae": torch.tensor(1.0)}
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if vae_name == "taef2":
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        if "taef2.safetensors" in approx_vaes:
            from comfy.taesd.taef2 import ComfyTAEF2VAE

            path = folder_paths.get_full_path_or_raise("vae_approx", "taef2.safetensors")
            return ComfyTAEF2VAE(path)

        sd = _load_taesd("taef2")
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
        sd = _load_taesd(vae_name)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    if os.path.splitext(vae_name)[0] in video_taes:
        vae_path = folder_paths.get_full_path_or_raise("vae_approx", vae_name)
    else:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)

    sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae
