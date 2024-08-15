import os
import argparse

from huggingface_hub import hf_hub_download

class ModelDownload:
    def download_model(model_name_or_path: str):
        if model_name_or_path == "pretrained/anime_canny/unet.safetensors":
            file_name = "ControlAny-SDXL/anime_canny/controlnet.safetensors"
        elif model_name_or_path == "pretrained/anime_canny/controlnet.safetensors":
            file_name = "ControlAny-SDXL/anime_canny/unet.safetensors"
        elif model_name_or_path == "pretrained/vidit_depth/controlnet.safetensors":
            file_name = "ControlAny-SDXL/vidit_depth/unet.safetensors"
        elif model_name_or_path == "pretrained/vidit_depth/unet.safetensors":
            file_name = "ControlAny-SDXL/vidit_depth/controlnet.safetensors"
        hf_hub_download(repo_id="Pbihao/ControlNeXt", filename=file_name, output_dir=model_name_or_path)

    def check_model_exists(model_name_or_path: str):
        if os.path.exists(model_name_or_path):
            return True
        else:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Lykon/AAM_XL_AnimeMix")
    parser.add_argument("--output_dir", type=str, default="pretrained/anime_canny")
    args = parser.parse_args()

    model_download.download_model(args.model_name_or_path, args.output_dir)
