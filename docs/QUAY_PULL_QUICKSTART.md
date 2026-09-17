# 直接 `docker pull` 的 H3 DMD-LoRA 昇腾训练流程

这套流程拉取 Quay 上现成的 **昇腾基础镜像**，在容器里安装固定版本的官方 LightX2V 和本仓库补丁。无需执行 `docker build`，但首次安装需要容器能访问 GitHub 和 PyPI。当前没有发布独立的、已经装好 LightX2V 的 Quay 镜像。

基础镜像：

```text
quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11
```

该标签在 Quay 上有 `linux/amd64` 和 `linux/arm64`。宿主机应是昇腾 910B，并有与 CANN 9.1.0 用户态兼容的驱动。

## 1. 宿主机准备

```bash
git clone https://github.com/BowenXiong1215/lightx2v-ascned.git
export PATCH_REPO_HOST="$(pwd)/lightx2v-ascned"

# 换成你机器上的绝对路径：
export MODEL_HOST=/path/to/MiniMax-H3-Diffusers
export PROMPTS_HOST=/path/to/prompts.txt
export OUTPUT_HOST=/path/to/h3-dmd-output
export CONFIG_HOST=/path/to/h3_dmd_ascend.yaml

mkdir -p "$OUTPUT_HOST"
cp "$PATCH_REPO_HOST/patches/lightx2v-ascend-h3-dmd-lora-20260916/payload/additions/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora_ascend.yaml" "$CONFIG_HOST"
```

编辑 `$CONFIG_HOST`，将以下三个值设为容器内路径：

```yaml
model:
    pretrained_model_name_or_path: /models/MiniMax-H3-Diffusers
data:
    train:
        data_path:
            - /data/prompts.txt
training:
    output_dir: /outputs
```

这里只展示需要修改的字段，保留配置文件其余内容。模型目录须为包含 `transformer/config.json` 的 Diffusers 格式。

## 2. 拉取并启动容器

```bash
export BASE_IMAGE=quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11
docker pull "$BASE_IMAGE"

docker run -it \
  --name lightx2v-h3-dmd-ascend \
  --network host --ipc host --privileged --shm-size 256g \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v "$PATCH_REPO_HOST:/workspace/lightx2v-ascned:ro" \
  -v "$MODEL_HOST:/models/MiniMax-H3-Diffusers:ro" \
  -v "$PROMPTS_HOST:/data/prompts.txt:ro" \
  -v "$CONFIG_HOST:/configs/h3_dmd_ascend.yaml:ro" \
  -v "$OUTPUT_HOST:/outputs" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$BASE_IMAGE" bash
```

驱动若不在 `/usr/local/Ascend/driver`，按服务器实际位置调整最后一个挂载。`--privileged` 与 `--ipc host` 延续本项目其他 910B 训练仓库的启动方式。

## 3. 容器内安装并开训

```bash
bash /workspace/lightx2v-ascned/scripts/setup_in_quay_container.sh

cd /workspace/LightX2V/lightx2v_train
H3_CONFIG_PATH=/configs/h3_dmd_ascend.yaml \
H3_NUM_PROCESSES=8 \
bash scripts/run_minimax_h3_t2av_dmd_ascend.sh
```

安装脚本锁定官方 LightX2V commit、校验补丁、安装 H3 所需依赖，并保持基础镜像内 PyTorch／`torch_npu` 的版本不变。再次使用已安装的容器：

```bash
docker start lightx2v-h3-dmd-ascend
docker exec -it lightx2v-h3-dmd-ascend bash
```

首次训练先在配置里缩小生成尺寸并把 `training.max_train_iters` 设为 1，完成单卡和八卡 smoke、checkpoint 保存恢复、LoRA 导出后再恢复 768p 正式配置。开发机没有 910B，镜像内安装和训练仍需在目标服务器验证。
