import os
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from utils import preprocess, tools


def log_validation(
    args,
    device='cuda'
):
    pipeline = tools.get_pipeline(
        args.pretrained_model_name_or_path,
        args.unet_model_name_or_path,
        args.controlnet_model_name_or_path,
        vae_model_name_or_path=args.vae_model_name_or_path,
        lora_path=args.lora_path,
        load_weight_increasement=args.load_weight_increasement,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        revision=args.revision,
        variant=args.variant,
        hf_cache_dir=args.hf_cache_dir,
        device=device,
    )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
        assert len(validation_prompts) == len(validation_prompts)
    else:
        negative_prompts = None

    extractor = preprocess.get_extractor(args.validation_image_processor)

    image_logs = []
    inference_ctx = torch.autocast(device)

    for i, (validation_prompt, validation_image) in enumerate(zip(validation_prompts, validation_images)):
        validation_image = Image.open(validation_image).convert("RGB")
        if extractor is not None:
            validation_image = extractor(validation_image)

        images = []
        negative_prompt = negative_prompts[i] if negative_prompts is not None else None
        width = args.width if args.width is not None else validation_image.width
        height = args.height if args.height is not None else validation_image.height
        validation_image = validation_image.resize((width, height))

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    controlnet_image=validation_image,
                    controlnet_scale=args.controlnet_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    save_dir_path = os.path.join(args.output_dir, "eval_img")
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
            file_path = os.path.join(save_dir_path, "image_{}-{}.png".format(i, j))
            validation_image = np.asarray(validation_image)
            validation_image = cv2.cvtColor(validation_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_path, validation_image)
            print("Save images to:", file_path)

        file_path = os.path.join(save_dir_path, "image_{}.png".format(i))
        formatted_images = cv2.cvtColor(formatted_images, cv2.COLOR_BGR2RGB)
        print("Save images to:", file_path)
        cv2.imwrite(file_path, formatted_images)

    return image_logs


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset"
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae model or subset"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to lora"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_image_processor",
        type=str,
        default=None,
        choices=["canny"],
        help="The type of image processor to use for the validation images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps for the diffusion model",
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="Scale of the controlnet",
    )
    parser.add_argument(
        "--load_weight_increasement",
        action="store_true",
        help="Only load weight increasement",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Path to the cache directory for huggingface datasets and models.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.width is not None and args.width % 8 != 0:
        raise ValueError(
            "`--width` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    if args.height is not None and args.height % 8 != 0:
        raise ValueError(
            "`--height` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


if __name__ == "__main__":
    args = parse_args()
    log_validation(args)
