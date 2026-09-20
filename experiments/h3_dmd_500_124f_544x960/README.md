# MiniMax-H3 DMD-LoRA: 124-frame training for 544×960 inference

This package contains 300 training prompts and 24 held-out validation prompts.
Every prompt names a visible scene and a natural sound. The training set uses
100 distinct audiovisual events with three camera variants; the validation
events are separate.

The 500-step run is a first experiment, with checkpoints every 100 steps.
Select a checkpoint using held-out prompts rather than assuming the last
checkpoint is best. The training config initializes the student from the
public LightX2V 544p four-step v0.1 LoRA, while the fake score model starts
with a fresh LoRA. It does not resume the earlier 22-frame run.

Direct training at 124 frames and 544×960 exhausted 64 GiB HBM during the
student backward pass. The generated config therefore uses the verified
124×320×576 training canvas; evaluate its checkpoints at 124×544×960.
Run the three-step probe first. The second and third
steps exercise the steady-state memory footprint after Adam allocates its
optimizer states.

Download the exact v0.1 file from ModelScope:

```bash
mkdir -p /hpc-to-ds-0115/x00876811/models/Minimax-h3-Turbo
modelscope download --model lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v0.1.safetensors --local_dir /hpc-to-ds-0115/x00876811/models/Minimax-h3-Turbo
sha256sum /hpc-to-ds-0115/x00876811/models/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v0.1.safetensors
```

The expected SHA-256 is
`5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29`.

After copying this entire directory to
`/hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/`, create the two
configs from the server's known-good overnight config:

```bash
python /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/build_train_configs.py
```

The generated configs are `h3_probe_3step.yaml` and
`h3_train_500step.yaml`. The builder keeps the existing model path, LoRA
targets, and FSDP configuration. It uses rank 128, alpha 8, a conservative
student learning rate of 5e-6, LoRA for both student and fake, four inference
steps, video shift 12, and audio shift 3.

Run the three-step probe:

```bash
ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 /hpc-to-ds-0115/x00876811/light/LightX2V-main/lightx2v_train/train.py --config /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/h3_probe_3step.yaml
```

Only after the log reports `finished iter=3/3`, run the fresh
500-step job:

```bash
ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29531 /hpc-to-ds-0115/x00876811/light/LightX2V-main/lightx2v_train/train.py --config /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/h3_train_500step.yaml
```

The held-out prompts must not be added to `data.train.data_path`. Evaluate
the 100, 200, 300, 400, and 500 checkpoints with the same seeds and prompts.
Audio inference should keep the CPU audio-VAE decode workaround until the
Ascend decoder issue is resolved.
