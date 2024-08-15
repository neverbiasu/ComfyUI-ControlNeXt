import os
import torch
import cv2
import numpy as np
import argparse

from PIL import Image
from .utils import preprocess, tools
from .utils.model_download import ModelDownload

def model_check(model_name_or_path):
    model_download = ModelDownload(model_name_or_path)
    if model_download.check_model_exists():
        return
    else:
        model_download.download_model()

class ControlNextPipelineConfig:
    def __init__(self):
        self.pipeline = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pretrained_model_name_or_path": ("STRING", {"default": "Lykon/AAM_XL_AnimeMix"}),
                "controlnet_model_name_or_path": ("STRING", {"default": "pretrained/anime_canny/controlnet.safetensors"}),
                "unet_model_name_or_path": ("STRING", {"default": "pretrained/anime_canny/unet.safetensors"}),
                "vae_model_name_or_path": ("STRING", {"default": "madebyollin/sdxl-vae-fp16-fix"}),
                "lora_path": ("STRING", {"default": "lora/amiya.safetensors"}),
                "load_weight_increasement": ("BOOLEAN", {"default": False}),
                "enable_xformers": ("BOOLEAN", {"default": False}),
                "revision": ("STRING", {"default": None}),
                "variant": ("STRING", {"default": "fp16"}),
                "hf_cache_dir": ("STRING", {"default": None}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "ControlNet"

    def load_pipeline(self, pretrained_model_name_or_path, controlnet_model_name_or_path, unet_model_name_or_path, 
                      vae_model_name_or_path, lora_path, load_weight_increasement, enable_xformers, revision, 
                      variant, hf_cache_dir, device):

        model_check(controlnet_model_name_or_path)
        model_check(unet_model_name_or_path)

        print(f"Using device: {device}")

        self.pipeline = tools.get_pipeline(
            pretrained_model_name_or_path,
            unet_model_name_or_path,
            controlnet_model_name_or_path,
            vae_model_name_or_path=vae_model_name_or_path,
            lora_path=lora_path,
            load_weight_increasement=load_weight_increasement,
            enable_xformers_memory_efficient_attention=enable_xformers,
            revision=revision,
            variant=variant,
            hf_cache_dir=hf_cache_dir,
            device=device,
        )
        return (self.pipeline,)


class ControlNextSDXL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "seed": ("INT", {"default": None}),
                "validation_image": ("IMAGE",),
                "validation_prompt": ("STRING", {"default": "A photo of a cat"}),
                "negative_prompts": ("STRING", {"default": ""}),
                "validation_image_processor": ("STRING", {"default": None}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "num_validation_images": ("INT", {"default": 4}),
                "controlnet_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 20}),
                "output_dir": ("STRING", {"default": "output"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "ControlNet"

    def generate_image(self, pipeline, seed, validation_image, validation_prompt, negative_prompts, 
                       validation_image_processor, width, height, num_validation_images, controlnet_scale, 
                       num_inference_steps, output_dir):
        
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)

        extractor = preprocess.get_extractor(validation_image_processor)

        image_logs = []
        inference_ctx = torch.autocast(pipeline.device)

        validation_images, validation_prompts = self._prepare_validation_pairs(
            validation_image, validation_prompt)

        for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
            validation_image = Image.open(validation_image).convert("RGB")
            if extractor is not None:
                validation_image = extractor(validation_image)

            images = []
            negative_prompt = negative_prompts[i] if negative_prompts is not None else None
            validation_image = validation_image.resize((width, height))

            for _ in range(num_validation_images):
                with inference_ctx:
                    image = pipeline(
                        prompt=validation_prompt,
                        controlnet_image=validation_image,
                        controlnet_scale=controlnet_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                    ).images[0]

                images.append(image)

            image_logs.append({"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt})

        self._save_images(image_logs, output_dir)
        return (image_logs[-1]["images"][-1],)

    def _prepare_validation_pairs(self, validation_image, validation_prompt):
        if len(validation_image) == len(validation_prompt):
            validation_images = validation_image
            validation_prompts = validation_prompt
        elif len(validation_image) == 1:
            validation_images = validation_image * len(validation_prompt)
            validation_prompts = validation_prompt
        elif len(validation_prompt) == 1:
            validation_images = validation_image
            validation_prompts = validation_prompt * len(validation_image)
        else:
            raise ValueError("number of `validation_image` and `validation_prompt` should be checked")

        return validation_images, validation_prompts

    def _save_images(self, image_logs, output_dir):
        save_dir_path = os.path.join(output_dir, "eval_img")
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]

            formatted_images = []
            formatted_images.append(np.asarray(validation_image))
            for image in images:
                formatted_images.append(np.asarray(image))
            formatted_images = np.concatenate(formatted_images, 1)

            for j, validation_image in enumerate(images):
                file_path = os.path.join(save_dir_path, f"image_{i}-{j}.png")
                validation_image = np.asarray(validation_image)
                validation_image = cv2.cvtColor(validation_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(file_path, validation_image)
                print("Save images to:", file_path)

            file_path = os.path.join(save_dir_path, f"image_{i}.png")
            formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
            print("Save images to:", file_path)
            cv2.imwrite(file_path, formatted_images)
