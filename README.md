# LightX2V MiniMax-H3 DMD-LoRA 昇腾训练补丁

本仓库提供官方 [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) 固定版本 `bb8301a7dfe8180772bb864d269d8bd05a938f8e` 的昇腾训练补丁。它只发布适配代码，不包含上游完整源码、模型权重或训练数据。

## 训练方案

- MiniMax-H3 文本生成有声视频（T2AV），四次 student forward 的 DMD 训练。
- student 为 rank 128 LoRA；teacher 冻结；fake score 模型全参数训练。
- 视频、音频 flow shift 分别为 6 和 3。
- 通过 `torch_npu`、HCCL、FSDP2 以及 Diffusers `_native_npu` attention 运行。

## 安装

```bash
git clone https://github.com/ModelTC/LightX2V.git
git -C LightX2V checkout bb8301a7dfe8180772bb864d269d8bd05a938f8e
git clone https://github.com/BowenXiong1215/lightx2v-ascned.git
bash lightx2v-ascned/patches/lightx2v-ascend-h3-dmd-lora-20260916/install.sh LightX2V
```

安装脚本逐项检查上游文件和补丁文件的 SHA-256，拒绝覆盖未知修改，可重复执行。离线环境可下载本仓库中的 `lightx2v-ascend-h3-dmd-lora-20260916.tar.gz`。

## 配置与启动

按补丁安装后的 [昇腾训练说明](patches/lightx2v-ascend-h3-dmd-lora-20260916/payload/additions/docs/ascend/minimax_h3_4step_dmd_training.md) 准备与 CANN 匹配的 PyTorch、`torch_npu`、MiniMax-H3 Diffusers 模型和 prompt 数据。填写 `lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml` 中的模型与数据路径，再运行：

```bash
cd LightX2V/lightx2v_train
bash scripts/run_minimax_h3_t2av_dmd_ascend.sh
```

## 验证状态

补丁已在固定版本的干净源码上完成安装与重复安装校验，静态验证通过，加速器单元测试 5/5 通过。当前开发机没有 `torch_npu` 和昇腾设备；实际 H3 前向、反向、八卡 checkpoint 恢复及质量对齐仍需按训练说明在目标机器上验收。
