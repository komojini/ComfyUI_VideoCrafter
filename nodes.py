import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything


from .video_crafter_utils.utils import instantiate_from_config
from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    load_image_batch,
    get_filelist,
)

import folder_paths

EXTENSION_PATH = Path(__file__).parent

# ckpt_path_base = "checkpoints/base_1024_v1/model.ckpt"
config_base = EXTENSION_PATH / "configs" / "inference_t2v_1024_v1.0.yaml"

# ckpt_path_i2v = "checkpoints/i2v_512_v1/model.ckpt"
config_i2v = EXTENSION_PATH / "configs" / "inference_i2v_512_v1.0.yaml"

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "checkpoints")


class VideoCrafterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["text2video", "image2video"],),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, type, ckpt_name, **kwargs):

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        print(f"videocrafter ckpt path: {ckpt_path}")

        if type == "text2video":
            config = config_base
        elif type == "image2video":
            config = config_i2v  

        config = OmegaConf.load(config)
        model_config = config.pop("model", OmegaConf.create())
        model = instantiate_from_config(model_config)
        model = model.cuda()
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()

        return (model,)


def run_video_crafter(
    task,
    model,
    ddim_steps,
    unconditional_guidance_scale,
    seed,
    prompt,
    image=None,
):
    width = 1024 if task == "text2video" else 512
    height = 576 if task == "text2video" else 320

    if task == "image2video":
        assert image is not None, "Please provide image for image2video generation."

    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    seed_everything(seed)

    args = argparse.Namespace(
        mode="base" if task == "text2video" else "i2v",
        n_samples=1,
        ddim_steps=ddim_steps,
        ddim_eta=1.0,
        bs=1,
        height=height,
        width=width,
        frames=-1,
        fps=28 if task == "text2video" else 8,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_guidance_scale_temporal=None,
    )

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    batch_size = 1
    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps] * batch_size).to(model.device).long()
    prompts = [prompt]
    text_emb = model.get_learned_conditioning(prompts)

    if args.mode == "base":
        cond = {"c_crossattn": [text_emb], "fps": fps}
    elif args.mode == "i2v":
        # cond_images = load_image_batch([image], (args.height, args.width))
        cond_images = (image - 0.5) * 2
        cond_images = cond_images.permute(0, 3, 1, 2).float()
        cond_images = cond_images.to(model.device)
        img_emb = model.get_image_embeds(cond_images)
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}
    else:
        raise NotImplementedError

    ## inference
    batch_samples = batch_ddim_sampling(
        model,
        cond,
        noise_shape,
        args.n_samples,
        args.ddim_steps,
        args.ddim_eta,
        args.unconditional_guidance_scale,
    )

    vid_tensor = batch_samples[0]
    video = vid_tensor.detach().cpu()
    print("debug")
    print(torch.min(video), torch.max(video))
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
        for framesheet in video
    ]  # [3, 1*h, n*w]
    grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
    grid = (grid + 1.0) / 2.0
    print(torch.min(grid), torch.max(grid))

    grid = grid.permute(0, 2, 3, 1) # [t, n*h, w, 3]
    grid = torch.clamp(grid, 0, 1)
    # grid = torch.clamp((grid * 255.0).round(), 0, 255) / 255.
    return grid



class I2V_VideoCrafter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "ddim_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),            
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
        
    
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    def run(self, model, image, ddim_steps, cfg, seed, prompt, **kwargs):
        return (run_video_crafter("image2video", model, ddim_steps, cfg, seed, image=image, prompt=prompt),)


class T2V_VideoCrafter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ddim_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),            
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    CATEGORY = "videocrafter"

    def run(self, model, ddim_steps, cfg, seed, prompt, **kwargs):
        return (run_video_crafter("text2video", model, ddim_steps, cfg, seed, prompt=prompt),)


NODE_CLASS_MAPPINGS = {
    "VideoCrafterModelLoader": VideoCrafterLoader,
    "T2V_VideoCrafterSampler": T2V_VideoCrafter,
    "I2V_VideoCrafterSampler": I2V_VideoCrafter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCrafterModelLoader": "VideoCrafter Model Loader",
    "T2V_VideoCrafterSampler": "T2V VideoCrafter Sampler",
    "I2V_VideoCrafterSampler": "I2V VideoCrafter Sampler",
}