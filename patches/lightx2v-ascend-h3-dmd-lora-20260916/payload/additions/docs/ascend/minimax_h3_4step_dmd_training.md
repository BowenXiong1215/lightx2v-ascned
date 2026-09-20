# MiniMax-H3 4-step DMD LoRA training on Ascend

This patch keeps the upstream CUDA path intact and adds a configuration-driven
Ascend path for the official LightX2V-Train MiniMax-H3 DMD recipe.

## Scope

- MiniMax-H3 T2AV
- DMD, four student model evaluations
- LoRA student (`rank=128`, `alpha=8`)
- configurable full or LoRA fake score model
- FSDP2 over eight NPUs
- CPU staging before FSDP2 to avoid a full unsharded H3 copy on every NPU
- HCCL collectives
- Diffusers `_native_npu` attention dispatched by `torch_npu`
- chunked global L2 gradient clipping on NPU to bound temporary HBM usage

This is an engineering bring-up recipe. Quality parity must be established on
Ascend hardware with the same base checkpoint, prompt set, seed, and exported
LoRA used by the CUDA baseline.

## Environment

Use an Ascend container whose PyTorch and `torch_npu` builds match its CANN
release. Install the remaining Python dependencies without replacing the
preinstalled torch packages:

```bash
pip install -r requirements/ascend-training.txt
pip install -e . --no-deps
```

The model path must be the converted Diffusers root containing
`transformer/config.json` for `MiniMaxH3Transformer3DModel`.
Use the MiniMax-H3-compatible Diffusers revision supplied with the model until
that implementation is available in a stable Diffusers release; a generic old
Diffusers wheel does not contain the required transformer or NPU backend.

## Initialize from a public LoRA

`training.student.init_lora_path` accepts a single native PEFT safetensors
file. Loading occurs after LoRA injection and before FSDP2 wrapping. The loader
requires every configured adapter tensor to be present with the expected
shape, so an incompatible rank, target list, or transformer partition fails
before training starts.

For the LightX2V 544p four-step v0.1 release, use rank 128, alpha 8, video
shift 12, and audio shift 3. This is a weight-only initialization: optimizer,
fake-model, scheduler, and iteration state start fresh.

The 124x544x960 v0.1 recipe keeps the public training geometry. On 64 GiB
NPUs, the NPU path computes the same global L2 clipping coefficient in bounded
float32 chunks instead of launching one `LpNormV2` over each complete gradient.
Configure the fake score model as LoRA to avoid allocating full-model Adam
state after the first iteration.

## Launch

Edit the model and prompt paths in
`lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml`, then:

```bash
cd lightx2v_train
bash scripts/run_minimax_h3_t2av_dmd_ascend.sh
```

Override `H3_NUM_PROCESSES` for a different topology. The configured FSDP2
size follows `WORLD_SIZE` automatically.

## Acceptance sequence

1. Run one process and one training iteration at a reduced generation shape.
2. Run eight processes and verify all student, fake, and teacher forwards.
3. Save and resume one FSDP2 checkpoint.
4. Export the LoRA and compare its keys, rank, alpha, and one fixed-seed sample
   against the CUDA baseline.
5. Restore the official 124x768x1344 shape and complete the 1,000-iteration
   run only after the smoke tests pass.

Do not change the video/audio flow shifts from 6/3 for this 768p distilled
recipe, and do not replace four evaluations with a generic four-point schedule.
