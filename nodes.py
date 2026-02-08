from .nodes_rt import (
    DownloadAndLoadMarigoldIIDAppearanceModelRT,
    DownloadAndLoadMarigoldIIDLightingModelRT,
    DownloadAndLoadPaGeRModelRT,
)
from .nodes_tensorrt import LoadUNetTensorRTEngine


NODE_CLASS_MAPPINGS = {
    "LoadUNetTensorRTEngine": LoadUNetTensorRTEngine,
    "DownloadAndLoadMarigoldIIDAppearanceModelRT": DownloadAndLoadMarigoldIIDAppearanceModelRT,
    "DownloadAndLoadMarigoldIIDLightingModelRT": DownloadAndLoadMarigoldIIDLightingModelRT,
    "DownloadAndLoadPaGeRModelRT": DownloadAndLoadPaGeRModelRT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadUNetTensorRTEngine": "Load UNet -> TensorRT Engine",
    "DownloadAndLoadMarigoldIIDAppearanceModelRT": "Load Marigold IID Appearance Model (TensorRT)",
    "DownloadAndLoadMarigoldIIDLightingModelRT": "Load Marigold IID Lighting Model (TensorRT)",
    "DownloadAndLoadPaGeRModelRT": "Load PaGeR Model (TensorRT)",
}
