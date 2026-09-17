# LightX2V H3 DMD-LoRA Ascend patch

Upstream: ModelTC/LightX2V commit `bb8301a7dfe8180772bb864d269d8bd05a938f8e`.

This bundle rebases the September 7 Ascend adaptation onto the official source. The only upstream conflict was `cache_build.py`; its new generation-shape validation is preserved while NPU seeding is added.

Install into a clean checkout of the pinned upstream commit:

```bash
bash patches/lightx2v-ascend-h3-dmd-lora-20260916/install.sh /path/to/LightX2V
```

The installer checks source and payload SHA-256 values before writing, and `verify.sh` checks the installed result. Hardware acceptance steps are in `docs/ascend/minimax_h3_4step_dmd_training.md`.
