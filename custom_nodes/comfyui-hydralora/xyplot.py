import logging
import torch
import folder_paths
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.utils
from nodes import CLIPTextEncode

from .adapter import apply_adapter
from .fera import apply_fera
from .grid import tensor_to_pil, create_grid
from .xy_inputs import XY_INPUT_CLASS_MAPPINGS, XY_INPUT_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger(__name__)


def _get_folder_list(folder_name, fallback=None):
    names = folder_paths.get_filename_list(folder_name)
    if not names and fallback:
        names = folder_paths.get_filename_list(fallback)
    return names


def _get_folder_path(folder_name, filename, fallback=None):
    p = folder_paths.get_full_path(folder_name, filename)
    if p is None and fallback:
        p = folder_paths.get_full_path(fallback, filename)
    return p


def _try_apply_adapter(model, lora_path, strength_lora, strength_reft):
    try:
        apply_adapter(model, lora_path, strength_lora, strength_reft)
        return True
    except Exception as e:
        logger.warning(f"apply_adapter failed for {lora_path}: {e}")
    try:
        apply_fera(model, lora_path, strength_lora)
        return True
    except Exception as e:
        logger.warning(f"apply_fera failed for {lora_path}: {e}")
    logger.error(f"All adapter types failed for: {lora_path}")
    return False


class AnimaEfficientLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_get_folder_list("diffusion_models", "unet"),),
                "clip_name": (_get_folder_list("text_encoders", "clip"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "empty_latent_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": 16384, "step": 64},
                ),
                "empty_latent_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": 16384, "step": 64},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "VAE",
        "CLIP",
        "DEPENDENCIES",
    )
    RETURN_NAMES = (
        "MODEL",
        "CONDITIONING+",
        "CONDITIONING-",
        "LATENT",
        "VAE",
        "CLIP",
        "DEPENDENCIES",
    )
    FUNCTION = "load"
    CATEGORY = "Anima XY Plot"
    DESCRIPTION = (
        "Anima Efficient Loader: loads UNet + CLIP + VAE + optional LoRA, "
        "encodes prompts, creates empty latent."
    )

    def load(
        self,
        unet_name,
        clip_name,
        vae_name,
        lora_name,
        strength_model,
        strength_clip,
        positive,
        negative,
        empty_latent_width,
        empty_latent_height,
        batch_size,
    ):
        unet_path = _get_folder_path("diffusion_models", unet_name, "unet")
        clip_path = _get_folder_path("text_encoders", clip_name, "clip")
        vae_path = folder_paths.get_full_path("vae", vae_name)

        model = comfy.sd.load_diffusion_model(unet_path)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        )
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)

        base_model = model.clone()

        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora_sd, strength_model, strength_clip
            )

        positive_encoded = CLIPTextEncode().encode(clip, positive)[0]
        negative_encoded = CLIPTextEncode().encode(clip, negative)[0]

        latent_tensor = torch.zeros(
            [
                batch_size,
                16,
                1,
                empty_latent_height // 8,
                empty_latent_width // 8,
            ]
        )
        latent_dict = {"samples": latent_tensor}

        dependencies = (
            base_model,
            clip,
            vae_name,
            lora_name,
            strength_model,
            strength_clip,
            positive,
            negative,
            empty_latent_width,
            empty_latent_height,
            batch_size,
        )

        return (
            model,
            positive_encoded,
            negative_encoded,
            latent_dict,
            vae,
            clip,
            dependencies,
        )


class AnimaXYPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("ANIMA_XY",),
                "grid_spacing": (
                    "INT",
                    {"default": 0, "min": 0, "max": 500, "step": 5},
                ),
                "XY_flip": (["False", "True"],),
                "Y_label_orientation": (["Horizontal", "Vertical"],),
                "ksampler_output_images": (["Images", "Plot"],),
            },
            "optional": {
                "y": ("ANIMA_XY",),
            },
        }

    RETURN_TYPES = ("ANIMA_XYPLOT",)
    RETURN_NAMES = ("xyplot",)
    FUNCTION = "plot"
    CATEGORY = "Anima XY Plot"
    DESCRIPTION = "Anima XY Plot: collects X and Y axis inputs for parameter sweep."

    def plot(
        self,
        x,
        grid_spacing,
        XY_flip,
        Y_label_orientation,
        ksampler_output_images,
        y=None,
    ):
        result = {
            "x_type": x["type"],
            "x_values": x["values"],
            "x_label": x.get("label", x["type"]),
            "x_search": x.get("search"),
            "y_type": y["type"] if y else None,
            "y_values": y["values"] if y else None,
            "y_label": y.get("label", y["type"]) if y else None,
            "y_search": y.get("search") if y else None,
            "grid_spacing": grid_spacing,
            "XY_flip": XY_flip,
            "Y_label_orientation": Y_label_orientation,
            "ksampler_output_images": ksampler_output_images,
        }
        return (result,)


class AnimaEfficientKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "add_noise": (["enable", "disable"],),
                "start_at_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000},
                ),
                "end_at_step": (
                    "INT",
                    {"default": 10000, "min": 0, "max": 10000},
                ),
                "return_with_leftover_noise": (["enable", "disable"],),
                "preview_method": (
                    ["auto", "latent2rgb", "taesd", "vae_decoded_only", "none"],
                ),
            },
            "optional": {
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "dependencies": ("DEPENDENCIES",),
                "xyplot": ("ANIMA_XYPLOT",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE")
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Anima XY Plot"
    DESCRIPTION = (
        "Anima Efficient KSampler: samples with optional XY Plot grid generation."
    )

    @staticmethod
    def _set_preview_method(method):
        from comfy.cli_args import args
        import latent_preview as _lp

        if method == "auto":
            args.preview_method = _lp.LatentPreviewMethod.Auto
        elif method == "latent2rgb":
            args.preview_method = _lp.LatentPreviewMethod.Latent2RGB
        elif method == "taesd":
            args.preview_method = _lp.LatentPreviewMethod.TAESD
        else:
            args.preview_method = _lp.LatentPreviewMethod.NoPreviews

    def _do_sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        positive,
        negative,
        latent_image,
        vae,
        add_noise="enable",
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise="disable",
    ):
        latent_image = (
            latent_image["samples"] if isinstance(latent_image, dict) else latent_image
        )
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_image, seed)

        if add_noise == "disable":
            noise = torch.zeros_like(latent_image)

        import latent_preview

        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
            disable_noise=(add_noise == "disable"),
            start_step=start_at_step if start_at_step > 0 else None,
            last_step=end_at_step if end_at_step < 10000 else None,
            force_full_denoise=(return_with_leftover_noise == "disable"),
            callback=callback,
            disable_pbar=False,
        )
        decoded = vae.decode(samples)
        if decoded.dim() == 5:
            decoded = decoded[:, 0]
        return decoded

    def _apply_param(self, param_type, param_val, cur, deps, xyplot, axis):
        if param_type == "seeds":
            cur["seed"] = int(param_val)
        elif param_type == "steps":
            cur["steps"] = int(param_val)
        elif param_type == "cfg":
            cur["cfg"] = float(param_val)
        elif param_type == "sampler_scheduler":
            cur["sampler_name"] = param_val[0]
            cur["scheduler"] = param_val[1]
        elif param_type == "denoise":
            cur["denoise"] = float(param_val)
        elif param_type == "positive_prompt_sr":
            search_txt, replace_txt = param_val
            positive_text = deps[6]
            clip = deps[1]
            if replace_txt is not None:
                new_text = positive_text.replace(search_txt, replace_txt, 1)
            else:
                new_text = positive_text
            cur["positive"] = CLIPTextEncode().encode(clip, new_text)[0]
        elif param_type == "negative_prompt_sr":
            search_txt, replace_txt = param_val
            negative_text = deps[7]
            clip = deps[1]
            if replace_txt is not None:
                new_text = negative_text.replace(search_txt, replace_txt, 1)
            else:
                new_text = negative_text
            cur["negative"] = CLIPTextEncode().encode(clip, new_text)[0]
        elif param_type == "anima_adapter":
            adapter_name, lora_str, reft_str = param_val
            adapter_path = folder_paths.get_full_path("loras", adapter_name)
            base_model = deps[0]
            new_model = base_model.clone()
            ok = _try_apply_adapter(new_model, adapter_path, lora_str, reft_str)
            if not ok:
                raise RuntimeError(
                    f"Failed to apply adapter '{adapter_name}' "
                    f"(path: {adapter_path}). Check ComfyUI log for details."
                )
            cur["model"] = new_model
        elif param_type == "anima_adapter_strength":
            lora_name = deps[3]
            if lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                base_model = deps[0]
                clip = deps[1]
                strength_clip = deps[5]
                positive_text = deps[6]
                negative_text = deps[7]
                lora_sd = comfy.utils.load_torch_file(lora_path)
                new_model, new_clip = comfy.sd.load_lora_for_models(
                    base_model, clip, lora_sd, float(param_val), strength_clip
                )
                cur["model"] = new_model
                cur["positive"] = CLIPTextEncode().encode(new_clip, positive_text)[0]
                cur["negative"] = CLIPTextEncode().encode(new_clip, negative_text)[0]
        elif param_type == "anima_reft_strength":
            lora_name = deps[3]
            if lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                base_model = deps[0]
                clip = deps[1]
                strength_model = deps[4]
                positive_text = deps[6]
                negative_text = deps[7]
                lora_sd = comfy.utils.load_torch_file(lora_path)
                new_model, new_clip = comfy.sd.load_lora_for_models(
                    base_model, clip, lora_sd, strength_model, float(param_val)
                )
                cur["model"] = new_model
                cur["positive"] = CLIPTextEncode().encode(new_clip, positive_text)[0]
                cur["negative"] = CLIPTextEncode().encode(new_clip, negative_text)[0]
        elif param_type == "lora":
            lora_name, model_str, clip_str = param_val
            base_model = deps[0]
            clip = deps[1]
            positive_text = deps[6]
            negative_text = deps[7]
            if lora_name != "None":
                lora_path = folder_paths.get_full_path("loras", lora_name)
                lora_sd = comfy.utils.load_torch_file(lora_path)
                new_model, new_clip = comfy.sd.load_lora_for_models(
                    base_model, clip, lora_sd, model_str, clip_str
                )
                cur["model"] = new_model
                cur["positive"] = CLIPTextEncode().encode(new_clip, positive_text)[0]
                cur["negative"] = CLIPTextEncode().encode(new_clip, negative_text)[0]
            else:
                cur["model"] = base_model.clone()
                cur["positive"] = CLIPTextEncode().encode(clip, positive_text)[0]
                cur["negative"] = CLIPTextEncode().encode(clip, negative_text)[0]
        elif param_type == "checkpoint":
            ckpt_path = _get_folder_path("diffusion_models", param_val, "unet")
            cur["model"] = comfy.sd.load_diffusion_model(ckpt_path)
        elif param_type == "vae":
            vae_path = folder_paths.get_full_path("vae", param_val)
            sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
            cur["vae"] = comfy.sd.VAE(sd=sd, metadata=metadata)

    def _make_label(self, param_type, param_val):
        if param_type == "seeds":
            return f"Seed: {param_val}"
        if param_type == "steps":
            return f"Steps: {param_val}"
        if param_type == "cfg":
            return f"CFG: {param_val:.1f}"
        if param_type == "sampler_scheduler":
            return f"{param_val[0]}/{param_val[1]}"
        if param_type == "denoise":
            return f"Denoise: {param_val:.2f}"
        if param_type in ("positive_prompt_sr", "negative_prompt_sr"):
            _, replace_txt = param_val
            return str(replace_txt if replace_txt is not None else param_val[0])[:25]
        if param_type == "anima_adapter":
            return param_val[0][:25]
        if param_type == "anima_adapter_strength":
            return f"Lora: {param_val:.2f}"
        if param_type == "anima_reft_strength":
            return f"ReFT: {param_val:.2f}"
        if param_type == "lora":
            import os

            return f"LoRA: {os.path.splitext(os.path.basename(param_val[0]))[0]}"[:25]
        if param_type == "anima_postfix":
            return param_val[0][:25]
        if param_type == "checkpoint":
            return param_val[:25]
        if param_type == "vae":
            return param_val[:25]
        return str(param_val)[:25]

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        add_noise="enable",
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise="disable",
        preview_method="auto",
        optional_vae=None,
        optional_clip=None,
        dependencies=None,
        xyplot=None,
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None,
    ):
        vae = optional_vae

        from comfy.cli_args import args

        prev_pm = args.preview_method
        try:
            if preview_method not in ("none", "vae_decoded_only"):
                self._set_preview_method(preview_method)

            if xyplot is None:
                images = self._do_sample(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    denoise,
                    positive,
                    negative,
                    latent_image,
                    vae,
                    add_noise=add_noise,
                    start_at_step=start_at_step,
                    end_at_step=end_at_step,
                    return_with_leftover_noise=return_with_leftover_noise,
                )
                from nodes import PreviewImage

                preview_images = PreviewImage().save_images(
                    images, prompt=prompt, extra_pnginfo=extra_pnginfo
                )["ui"]["images"]
                result = (model, positive, negative, latent_image, vae, images)
                return {"ui": {"images": preview_images}, "result": result}

            x_type = xyplot["x_type"]
            x_values = xyplot["x_values"]
            y_type = xyplot["y_type"]
            y_values = xyplot["y_values"] or [None]
            grid_spacing = xyplot.get("grid_spacing", 0)
            xy_flip = xyplot.get("XY_flip", "False") == "True"
            ksampler_output_images = xyplot.get("ksampler_output_images", "Plot")

            images_2d = []
            x_labels = [self._make_label(x_type, v) for v in x_values]
            y_labels = []
            all_images = []

            total_images = len(x_values) * len(y_values)
            overall_pbar = comfy.utils.ProgressBar(total_images)
            completed = 0

            for y_val in y_values:
                if y_type is not None and y_val is not None:
                    y_labels.append(self._make_label(y_type, y_val))
                row = []
                for x_val in x_values:
                    cur = {
                        "seed": seed,
                        "steps": steps,
                        "cfg": cfg,
                        "sampler_name": sampler_name,
                        "scheduler": scheduler,
                        "denoise": denoise,
                        "positive": positive,
                        "negative": negative,
                        "model": model,
                        "vae": vae,
                        "add_noise": add_noise,
                        "start_at_step": start_at_step,
                        "end_at_step": end_at_step,
                        "return_with_leftover_noise": return_with_leftover_noise,
                    }

                    self._apply_param(x_type, x_val, cur, dependencies, xyplot, "x")
                    if y_type is not None and y_val is not None:
                        self._apply_param(y_type, y_val, cur, dependencies, xyplot, "y")

                    img = self._do_sample(
                        cur["model"],
                        cur["seed"],
                        cur["steps"],
                        cur["cfg"],
                        cur["sampler_name"],
                        cur["scheduler"],
                        cur["denoise"],
                        cur["positive"],
                        cur["negative"],
                        latent_image,
                        cur["vae"],
                        add_noise=cur["add_noise"],
                        start_at_step=cur["start_at_step"],
                        end_at_step=cur["end_at_step"],
                        return_with_leftover_noise=cur["return_with_leftover_noise"],
                    )
                    completed += 1
                    overall_pbar.update_absolute(completed, total_images)
                    all_images.append(img)
                    row.append(tensor_to_pil(img))
                images_2d.append(row)

            if xy_flip:
                images_2d = [list(col) for col in zip(*images_2d)]
                x_labels, y_labels = y_labels, x_labels

            y_label_orientation = xyplot.get("Y_label_orientation", "Horizontal")
            grid_tensor = create_grid(
                images_2d,
                x_labels,
                y_labels,
                grid_spacing=grid_spacing,
                y_label_orientation=y_label_orientation,
            )

            if ksampler_output_images == "Images":
                output_images = torch.cat(all_images, dim=0)
            else:
                output_images = grid_tensor

            from nodes import PreviewImage

            preview_images = PreviewImage().save_images(
                output_images, prompt=prompt, extra_pnginfo=extra_pnginfo
            )["ui"]["images"]
            result = (model, positive, negative, latent_image, vae, output_images)
            return {"ui": {"images": preview_images}, "result": result}
        finally:
            args.preview_method = prev_pm


NODE_CLASS_MAPPINGS = {
    "Anima Efficient Loader": AnimaEfficientLoader,
    "Anima Efficient KSampler": AnimaEfficientKSampler,
    "Anima XY Plot": AnimaXYPlot,
}
NODE_CLASS_MAPPINGS.update(XY_INPUT_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima Efficient Loader": "Anima Efficient Loader",
    "Anima Efficient KSampler": "Anima Efficient KSampler",
    "Anima XY Plot": "Anima XY Plot",
}
NODE_DISPLAY_NAME_MAPPINGS.update(XY_INPUT_DISPLAY_NAME_MAPPINGS)
