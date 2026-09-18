"""Generate one MiniMax-H3 DMD sample with the training FSDP2 shard plan."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers.utils.export_utils import encode_video
from diffusers.video_processor import VideoProcessor
from loguru import logger

from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed, load_config, setup_logger
from lightx2v_train.runtime.distributed import is_main_process
from lightx2v_train.runtime.fsdp import apply_fsdp2
from lightx2v_train.schedulers.dmd_scheduler import DMDFlowMatchingScheduler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Directory containing student pytorch_lora_weights.safetensors")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", action="store_true", help="Generate from the unadapted base transformer")
    return parser.parse_args()


def broadcast_from_rank_zero(value):
    if torch.is_tensor(value) and dist.is_initialized():
        dist.broadcast(value, src=0)
    return value


def decode_and_save(model, latents, output_path):
    shape = latents.shape
    patch_t, patch_h, patch_w = model.patch_size
    channels = model.video_latent_channels
    video = latents.video.reshape(
        1,
        shape.latent_frames // patch_t,
        shape.latent_height // patch_h,
        shape.latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    video = video.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        1, channels, shape.latent_frames, shape.latent_height, shape.latent_width
    )
    video_vae = model.video_vae
    mean = video.new_tensor(video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = video.new_tensor(video_vae.config.latents_std).view(1, -1, 1, 1, 1)
    video = video_vae.decode(video * std + mean, return_dict=False)[0]
    pixel_mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    pixel_std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
    frames = VideoProcessor(vae_scale_factor=16, do_normalize=False).postprocess_video(video, output_type="pil")[0]

    audio = latents.audio.reshape(2, shape.audio_latents, model.audio_latent_channels).permute(0, 2, 1).contiguous()
    audio_vae = model.audio_vae
    mean = audio.new_tensor(audio_vae.config.latents_mean).view(1, -1, 1)
    std = audio.new_tensor(audio_vae.config.latents_std).view(1, -1, 1)
    audio = audio_vae.decode(audio * std + mean, return_dict=False)[0].float().permute(1, 0, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        frames,
        fps=24,
        output_path=str(output_path),
        audio=audio[0].cpu(),
        audio_sample_rate=model.audio_sampling_rate,
    )
    logger.info("[h3-infer] saved {}", output_path)


def main():
    args = parse_args()
    config = load_config(args.config)
    init_distributed(config)
    setup_logger(config)
    try:
        if not dist.is_initialized() or dist.get_world_size() < 2:
            raise ValueError("H3 FSDP inference requires torchrun with at least two NPU ranks.")
        if config["model"]["name"] != "minimax_h3_t2av":
            raise ValueError("This entry point requires model.name=minimax_h3_t2av.")
        torch.manual_seed(args.seed)
        prompt = next(line.strip() for line in Path(args.prompt_file).read_text().splitlines() if line.strip())
        model = build_model(config)
        model.load_components(load_transformer=True, load_vae=True, load_condition_encoder=True)
        model.denoiser_module().requires_grad_(False).eval()
        if not args.base:
            lora = config["training"]["student"]["lora"]
            model.add_lora(int(lora["rank"]), float(lora["alpha"]), list(lora["target_modules"]))
            model.load_lora_weights_for_resume(args.checkpoint)
            model.denoiser_module().requires_grad_(False).eval()
        apply_fsdp2(model, config)
        logger.info("[h3-infer] rank={} model sharded", dist.get_rank())

        capability = model.capabilities.require(DistributionMatchingCapability)
        sample = {"conditioning": {"prompt": prompt}}
        shape = capability.latent_shape(
            sample,
            config["training"]["dmd"]["generation_shapes"],
            broadcast_from_rank_zero,
        )
        condition, _ = capability.encode_conditions(sample, "", 1.0, lambda x: x)
        scheduler = DMDFlowMatchingScheduler(config)
        steps = int(config["training"]["dmd"]["num_inference_steps"])
        scheduler.set_timesteps(steps, device=model.device)
        latents = capability.initial_latents(shape, dtype=torch.float32, broadcast=broadcast_from_rank_zero)
        with torch.no_grad():
            for index in range(steps):
                sigma = scheduler.sigma_at(index, device=model.device, dtype=torch.float32)
                velocity = capability.predict_velocity(latents, sigma, condition)
                latents, _ = capability.step(scheduler, velocity, index, latents)
                if is_main_process():
                    logger.info("[h3-infer] denoise step {}/{}", index + 1, steps)
        dist.barrier()
        if is_main_process():
            decode_and_save(model, latents, Path(args.output))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
