import folder_paths
import comfy.samplers
import numpy as np


class AnimaXYInputSeeds:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_count": ("INT", {"default": 5, "min": 1}),
                "first_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    def xy_input(self, seed_count, first_seed):
        values = [first_seed + i for i in range(seed_count)]
        return (
            {
                "type": "seeds",
                "label": f"Seeds: {first_seed}+{seed_count}",
                "values": values,
            },
        )


class AnimaXYInputSteps:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_step": ("INT", {"default": 1, "min": 1}),
                "last_step": ("INT", {"default": 20, "min": 1}),
                "step_count": ("INT", {"default": 10, "min": 2}),
            }
        }

    def xy_input(self, first_step, last_step, step_count):
        values = list(np.linspace(first_step, last_step, step_count).astype(int).tolist())
        return (
            {
                "type": "steps",
                "label": f"Steps: {first_step}-{last_step}",
                "values": values,
            },
        )


class AnimaXYInputCFG:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "last_cfg": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "cfg_count": ("INT", {"default": 5, "min": 2}),
            }
        }

    def xy_input(self, first_cfg, last_cfg, cfg_count):
        values = [round(first_cfg + i * (last_cfg - first_cfg) / (cfg_count - 1), 2) for i in range(cfg_count)]
        return (
            {
                "type": "cfg",
                "label": f"CFG: {first_cfg}-{last_cfg}",
                "values": values,
            },
        )


class AnimaXYInputDenoise:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "last_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "denoise_count": ("INT", {"default": 5, "min": 2}),
            }
        }

    def xy_input(self, first_denoise, last_denoise, denoise_count):
        values = [round(first_denoise + i * (last_denoise - first_denoise) / (denoise_count - 1), 2) for i in range(denoise_count)]
        return (
            {
                "type": "denoise",
                "label": f"Denoise: {first_denoise}-{last_denoise}",
                "values": values,
            },
        )


class AnimaXYInputSamplerScheduler:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        samplers = comfy.samplers.KSampler.SAMPLERS
        schedulers = comfy.samplers.KSampler.SCHEDULERS
        inputs = {"required": {"input_count": ("INT", {"default": 3, "min": 1, "max": 10})}}
        for i in range(1, 11):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)
        return inputs

    def xy_input(self, **kwargs):
        input_count = kwargs["input_count"]
        values = []
        for i in range(1, input_count + 1):
            sampler = kwargs.get(f"sampler_{i}")
            scheduler = kwargs.get(f"scheduler_{i}")
            values.append((sampler, scheduler))
        return ({"type": "sampler_scheduler", "label": "Sampler/Scheduler", "values": values},)


class AnimaXYInputPositivePromptSR:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {
            "search": ("STRING", {"default": "", "multiline": False}),
            "replace_count": ("INT", {"default": 2, "min": 1, "max": 10}),
        }}
        for i in range(1, 11):
            inputs["required"][f"replace_{i}"] = ("STRING", {"default": "", "multiline": False})
        return inputs

    def xy_input(self, **kwargs):
        search = kwargs["search"]
        replace_count = kwargs["replace_count"]
        values = [(search, None)]
        values.extend([(search, kwargs.get(f"replace_{i}", "")) for i in range(1, replace_count + 1)])
        return ({"type": "positive_prompt_sr", "label": "Positive Prompt S/R", "values": values, "search": search},)


class AnimaXYInputNegativePromptSR:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"required": {
            "search": ("STRING", {"default": "", "multiline": False}),
            "replace_count": ("INT", {"default": 2, "min": 1, "max": 10}),
        }}
        for i in range(1, 11):
            inputs["required"][f"replace_{i}"] = ("STRING", {"default": "", "multiline": False})
        return inputs

    def xy_input(self, **kwargs):
        search = kwargs["search"]
        replace_count = kwargs["replace_count"]
        values = [(search, None)]
        values.extend([(search, kwargs.get(f"replace_{i}", "")) for i in range(1, replace_count + 1)])
        return ({"type": "negative_prompt_sr", "label": "Negative Prompt S/R", "values": values, "search": search},)


class AnimaXYInputAnimaAdapter:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        inputs = {"required": {"input_count": ("INT", {"default": 2, "min": 1, "max": 10})}}
        for i in range(1, 11):
            inputs["required"][f"adapter_{i}"] = (lora_list,)
            inputs["required"][f"strength_lora_{i}"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
            inputs["required"][f"strength_reft_{i}"] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
        return inputs

    def xy_input(self, **kwargs):
        input_count = kwargs["input_count"]
        values = []
        for i in range(1, input_count + 1):
            adapter = kwargs.get(f"adapter_{i}", "")
            lora_str = kwargs.get(f"strength_lora_{i}", 1.0)
            reft_str = kwargs.get(f"strength_reft_{i}", 1.0)
            values.append((adapter, lora_str, reft_str))
        return ({"type": "anima_adapter", "label": "Anima Adapter", "values": values},)


class AnimaXYInputAnimaAdapterStrength:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_strength": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "last_strength": ("FLOAT", {"default": 2.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "strength_count": ("INT", {"default": 5, "min": 2}),
            }
        }

    def xy_input(self, first_strength, last_strength, strength_count):
        values = [round(first_strength + i * (last_strength - first_strength) / (strength_count - 1), 2) for i in range(strength_count)]
        return (
            {
                "type": "anima_adapter_strength",
                "label": f"Adapter Strength: {first_strength}-{last_strength}",
                "values": values,
            },
        )


class AnimaXYInputAnimaReFTStrength:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_strength": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "last_strength": ("FLOAT", {"default": 2.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "strength_count": ("INT", {"default": 5, "min": 2}),
            }
        }

    def xy_input(self, first_strength, last_strength, strength_count):
        values = [round(first_strength + i * (last_strength - first_strength) / (strength_count - 1), 2) for i in range(strength_count)]
        return (
            {
                "type": "anima_reft_strength",
                "label": f"ReFT Strength: {first_strength}-{last_strength}",
                "values": values,
            },
        )


class AnimaXYInputCheckpoint:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = folder_paths.get_filename_list("diffusion_models") or folder_paths.get_filename_list("unet")
        inputs = {"required": {"input_count": ("INT", {"default": 2, "min": 1, "max": 10})}}
        for i in range(1, 11):
            inputs["required"][f"unet_name_{i}"] = (unet_list,)
        return inputs

    def xy_input(self, **kwargs):
        input_count = kwargs["input_count"]
        values = [kwargs.get(f"unet_name_{i}", "") for i in range(1, input_count + 1)]
        return ({"type": "checkpoint", "label": "Checkpoint", "values": values},)


class AnimaXYInputVAE:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        vae_list = folder_paths.get_filename_list("vae")
        inputs = {"required": {"input_count": ("INT", {"default": 2, "min": 1, "max": 10})}}
        for i in range(1, 11):
            inputs["required"][f"vae_name_{i}"] = (vae_list,)
        return inputs

    def xy_input(self, **kwargs):
        input_count = kwargs["input_count"]
        values = [kwargs.get(f"vae_name_{i}", "") for i in range(1, input_count + 1)]
        return ({"type": "vae", "label": "VAE", "values": values},)


class AnimaXYInputLoRA:
    RETURN_TYPES = ("ANIMA_XY",)
    FUNCTION = "xy_input"
    CATEGORY = "Anima XY Plot/XY Input"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {"required": {"input_count": ("INT", {"default": 2, "min": 1, "max": 10})}}
        for i in range(1, 11):
            inputs["required"][f"lora_name_{i}"] = (lora_list,)
            inputs["required"][f"model_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_strength_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        return inputs

    def xy_input(self, **kwargs):
        input_count = kwargs["input_count"]
        values = []
        for i in range(1, input_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}", "None")
            model_str = kwargs.get(f"model_strength_{i}", 1.0)
            clip_str = kwargs.get(f"clip_strength_{i}", 1.0)
            values.append((lora_name, model_str, clip_str))
        return ({"type": "lora", "label": "LoRA", "values": values},)


XY_INPUT_CLASS_MAPPINGS = {
    "XY Input (Anima): Seeds": AnimaXYInputSeeds,
    "XY Input (Anima): Steps": AnimaXYInputSteps,
    "XY Input (Anima): CFG Scale": AnimaXYInputCFG,
    "XY Input (Anima): Denoise": AnimaXYInputDenoise,
    "XY Input (Anima): Sampler/Scheduler": AnimaXYInputSamplerScheduler,
    "XY Input (Anima): Positive Prompt S/R": AnimaXYInputPositivePromptSR,
    "XY Input (Anima): Negative Prompt S/R": AnimaXYInputNegativePromptSR,
    "XY Input (Anima): Anima Adapter": AnimaXYInputAnimaAdapter,
    "XY Input (Anima): Anima Adapter Strength": AnimaXYInputAnimaAdapterStrength,
    "XY Input (Anima): Anima ReFT Strength": AnimaXYInputAnimaReFTStrength,
    "XY Input (Anima): Checkpoint": AnimaXYInputCheckpoint,
    "XY Input (Anima): VAE": AnimaXYInputVAE,
    "XY Input (Anima): LoRA": AnimaXYInputLoRA,
}

XY_INPUT_DISPLAY_NAME_MAPPINGS = {
    "XY Input (Anima): Seeds": "XY Input (Anima): Seeds",
    "XY Input (Anima): Steps": "XY Input (Anima): Steps",
    "XY Input (Anima): CFG Scale": "XY Input (Anima): CFG Scale",
    "XY Input (Anima): Denoise": "XY Input (Anima): Denoise",
    "XY Input (Anima): Sampler/Scheduler": "XY Input (Anima): Sampler/Scheduler",
    "XY Input (Anima): Positive Prompt S/R": "XY Input (Anima): Positive Prompt S/R",
    "XY Input (Anima): Negative Prompt S/R": "XY Input (Anima): Negative Prompt S/R",
    "XY Input (Anima): Anima Adapter": "XY Input (Anima): Anima Adapter",
    "XY Input (Anima): Anima Adapter Strength": "XY Input (Anima): Anima Adapter Strength",
    "XY Input (Anima): Anima ReFT Strength": "XY Input (Anima): Anima ReFT Strength",
    "XY Input (Anima): Checkpoint": "XY Input (Anima): Checkpoint",
    "XY Input (Anima): VAE": "XY Input (Anima): VAE",
    "XY Input (Anima): LoRA": "XY Input (Anima): LoRA",
}
