# LightX2V H3 DMD-LoRA Ascend patch

Upstream: ModelTC/LightX2V commit `bb8301a7dfe8180772bb864d269d8bd05a938f8e`.

This bundle rebases the September 7 Ascend adaptation onto the official source. The only upstream conflict was `cache_build.py`; its new generation-shape validation is preserved while NPU seeding is added.

Install into a clean checkout of the pinned upstream commit:

```bash
bash patches/lightx2v-ascend-h3-dmd-lora-20260916/install.sh /path/to/LightX2V
```

The installer checks source and payload SHA-256 values before writing, and `verify.sh` checks the installed result. Hardware acceptance steps are in `docs/ascend/minimax_h3_4step_dmd_training.md`.

To upgrade an existing patched checkout in place while retaining local edits
in the DMD trainer, extract the release archive and run:

```bash
python lightx2v-ascend-h3-dmd-lora-20260916/upgrade_official_lora_warmstart.py /path/to/LightX2V
```

This adds native single-file PEFT LoRA initialization and bounded-memory NPU
gradient clipping. It backs up each replaced file with the suffix
`.pre_official_lora_init` and patches gradient-clipping calls in the existing
trainer instead of replacing that trainer wholesale.

For a fixed-seed sample from a saved student LoRA, run `lightx2v_train/infer_minimax_h3_dmd_fsdp.py` with `torchrun` on the same eight NPU ranks used for training. It reuses the training FSDP2 shard plan and four-step sigma schedule. Ascend generation and media decoding still require hardware validation.
