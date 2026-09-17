# 在 Quay 基础镜像上构建 H3 DMD-LoRA 昇腾训练镜像

## 基础镜像

默认使用：

```text
quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11
```

Quay 的该标签同时提供 `linux/amd64` 和 `linux/arm64`。Dockerfile 固定了对应的多架构 manifest digest。镜像内使用 CANN 9.1.0、PyTorch 2.7.1 和 `torch_npu` 2.7.1.post8。构建前先确认宿主机 910B 驱动与容器内的 CANN 9.1.0 用户态兼容。

旧的 `3.2.1-cann9.0.0-torch_npu2.7.1.post4-910b-ubuntu22.04-py3.11` 标签虽然存在，但 Quay 当前只提供 `linux/arm64`，不要在 x86_64 服务器上直接使用该标签。

## 构建

在有 Docker daemon 的昇腾服务器上运行；以下示例用于 x86_64 主机：

```bash
git clone https://github.com/BowenXiong1215/lightx2v-ascned.git
cd lightx2v-ascned
docker pull quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
  -t lightx2v-h3-dmd-ascend:20260916 .
```

ARM 服务器改为 `--platform linux/arm64`。构建时 Dockerfile 会拉取固定 commit `bb8301a7dfe8180772bb864d269d8bd05a938f8e` 的官方 LightX2V、应用本仓库补丁并校验、安装 H3 训练所需 Python 包。它会约束基础镜像中的 `torch`、`torch_npu` 版本，避免 pip 把 NPU 版 PyTorch 换成其他构建。

构建镜像不包含 MiniMax-H3 权重或训练数据；它们通过容器挂载提供。

## 启动与训练

先从 [昇腾训练配置](../patches/lightx2v-ascend-h3-dmd-lora-20260916/payload/additions/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml) 复制一份到宿主机，修改以下字段：

- `model.pretrained_model_name_or_path: /models/MiniMax-H3-Diffusers`
- `data.train.data_path: [/data/prompts.txt]`
- `training.output_dir: /outputs`

模型目录须是含 `transformer/config.json` 的 H3 Diffusers 格式。假设宿主机路径分别保存在 `MODEL_HOST`、`PROMPTS_HOST`、`CONFIG_HOST`、`OUTPUT_HOST`：

```bash
mkdir -p "$OUTPUT_HOST"
docker run --rm -it \
  --name lightx2v-h3-dmd-ascend \
  --network host --ipc host --privileged \
  --shm-size 256g \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v "$MODEL_HOST:/models/MiniMax-H3-Diffusers:ro" \
  -v "$PROMPTS_HOST:/data/prompts.txt:ro" \
  -v "$CONFIG_HOST:/configs/h3_dmd_ascend.yaml:ro" \
  -v "$OUTPUT_HOST:/outputs" \
  lightx2v-h3-dmd-ascend:20260916 \
  bash -lc 'H3_CONFIG_PATH=/configs/h3_dmd_ascend.yaml H3_NUM_PROCESSES=8 bash scripts/run_minimax_h3_t2av_dmd_ascend.sh'
```

`--privileged` 与 `--ipc host` 延续你现有的 910B 容器启动方式。若集群采用受限容器运行方式，需按宿主机部署方式显式映射 NPU 设备和驱动目录。首次启动建议先把配置里的训练步数与生成尺寸调小，依次验收单卡一步、八卡一步、checkpoint 恢复、LoRA 导出与固定 seed 推理；再恢复 768p、124 帧的正式配置。

本仓库的镜像构建说明已静态检查；开发机没有 Docker daemon 或 910B，尚未执行实际镜像构建与 NPU 训练。
