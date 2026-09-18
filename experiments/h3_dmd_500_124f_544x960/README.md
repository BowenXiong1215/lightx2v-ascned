# MiniMax-H3 DMD-LoRA: 124 frames, 544×960

This package contains 300 training prompts and 24 held-out validation prompts.
Every prompt names a visible scene and a natural sound. The training set uses
100 distinct audiovisual events with three camera variants; the validation
events are separate.

The 500-step run is a first experiment, with checkpoints every 100 steps.
Select a checkpoint using held-out prompts rather than assuming the last
checkpoint is best. The training config starts from the base checkpoint in a
fresh output directory; it does not resume the earlier 22-frame run.

Training at 124 frames and 544×960 has about 24 times as many video tokens as
22 frames and 256×448. Run the one-step probe first. Its success is required
before starting the 500-step run, but it does not guarantee that the full run
will remain free of memory pressure.

After copying this entire directory to
`/hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/`, create the two
configs from the server's known-good overnight config:

```bash
python /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/build_train_configs.py
```

The generated configs are `h3_probe_1step.yaml` and
`h3_train_500step.yaml`. The builder keeps the existing model path, LoRA
parameters, FSDP configuration, and optimizer settings. It sets four
inference steps, video shift 12, and audio shift 3 for the new training run.

Run the one-step probe:

```bash
ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29530 /hpc-to-ds-0115/x00876811/light/LightX2V-main/lightx2v_train/train.py --config /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/h3_probe_1step.yaml
```

Only after the probe completes and saves a checkpoint, run the fresh
500-step job:

```bash
ASCEND_LAUNCH_BLOCKING=1 torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29531 /hpc-to-ds-0115/x00876811/light/LightX2V-main/lightx2v_train/train.py --config /hpc-to-ds-0115/x00876811/light/h3_dmd_500_124f_544x960/h3_train_500step.yaml
```

The held-out prompts must not be added to `data.train.data_path`. Evaluate
the 100, 200, 300, 400, and 500 checkpoints with the same seeds and prompts.
Audio inference should keep the CPU audio-VAE decode workaround until the
Ascend decoder issue is resolved.
