# LoRA module building blocks. Public API re-exported here so
# `from networks.lora_modules import LoRAModule, ...` works unchanged.

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.chimera import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
)
from networks.lora_modules.hydra import HydraLoRAModule, _sigma_sinusoidal_features
from networks.lora_modules.lora import LoRAModule
from networks.lora_modules.ortho import (
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
)
from networks.lora_modules.loha import LohaModule
from networks.lora_modules.lokr import LokrModule
from networks.lora_modules.locon import LoConModule
from networks.lora_modules.reft import ReFTModule
from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule

__all__ = [
    "BaseLoRAModule",
    "ChimeraHydraInferenceModule",
    "ChimeraHydraLoRAModule",
    "HydraLoRAModule",
    "LohaModule",
    "LoConModule",
    "LokrModule",
    "LoRAModule",
    "OrthoHydraLoRAModule",
    "OrthoLoRAModule",
    "ReFTModule",
    "StackedExpertsLoRAModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]
