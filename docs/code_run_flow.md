# WorldArena 机器人视频世界模型：代码 / 运行流程说明

> 版本：2026-05-09  
> 适用范围：ABot-PhysWorld、Wan2.1 baseline、VideoX-Fun LoRA SFT、SeedVR 后处理、WorldArena 指标评测  
> 本文档用于交接和复现，不是展示页；展示页见 `index.html`。

---

## 1. 项目目标

本项目围绕 **WorldArena Challenge Track 1** 的机器人视频生成任务，完成下面这条链路：

```text
论文 / 仓库复现
  → 数据准备
  → ABot / Wan2.1 baseline 推理
  → RoboTwin 数据转 VideoX-Fun 格式
  → Wan2.1 LoRA SFT 训练
  → checkpoint 推理
  → SeedVR 视频后处理
  → WorldArena 指标 + VLM Judge
  → 对比分析与网页展示
```

核心对比对象：

| 模型 | 作用 | 当前状态 |
|---|---|---|
| ABot-PhysWorld | 机器人领域 world model baseline | 10-sample 正式规格推理与评测完成 |
| Vanilla Wan2.1 | 通用视频生成 baseline | 10-sample 正式规格推理与评测完成 |
| SFT-Wan2.1 | 本项目用 RoboTwin 训练的 LoRA 模型 | checkpoint-2200 正在做 1000-video full eval |
| SeedVR | 视频清晰度 / artifact 后处理 | 已接入 10-sample 和 1000-video 流程 |

---

## 2. 关键目录

> 以下为当前服务器上的实际路径。如果迁移到另一台机器，需要整体替换 `/data/alice/cjtest` 前缀。

| 模块 | 路径 | 说明 |
|---|---|---|
| 项目根目录 | `/data/alice/cjtest` | 总工作区 |
| 展示网站 | `/data/alice/cjtest/github-pages-site` | GitHub Pages 静态页 |
| VideoX-Fun | `/data/alice/cjtest/VideoX-Fun` | Wan2.1 LoRA 训练 / 推理主框架 |
| ABot-PhysWorld | `/data/alice/cjtest/model_repros/ABot-PhysWorld` | ABot 复现仓库 |
| WorldArena metrics | `/data/alice/cjtest/model_repros/WorldArena/video_quality` | 标准指标和 VLM judge |
| SeedVR | `/data/alice/cjtest/model_repros/FlowWAM_WorldArena/inference/refiner/SeedVR` | 视频后处理 |
| RoboTwin SFT 数据 | `/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50` | 2500 个训练视频 |
| WorldArena test dataset | `/data/alice/cjtest/VideoX-Fun/test_dataset` | 1000 个测试 episode |
| LoRA checkpoints | `/data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora` | checkpoint-100 到 checkpoint-2200 |

---

## 3. 环境说明

不要把所有依赖装进同一个环境；当前项目故意拆成多个环境，避免 PyTorch / CUDA / flash-attn 冲突。

| 环境 | 路径 / 名称 | 用途 |
|---|---|---|
| `videox_fun_wan` | `/data/envs/videox_fun_wan` | VideoX-Fun 训练、Wan2.1 推理、LoRA 推理 |
| `WorldArena` | `/data2/miniconda3/envs/WorldArena` | WorldArena 标准视觉指标 |
| `WorldArena_VLM` | `/data2/miniconda3/envs/WorldArena_VLM` | Qwen / VLM judge |
| `seedvr` | `/data2/envs/seedvr` | SeedVR 后处理 |

常用检查：

```bash
nvidia-smi
/data/envs/videox_fun_wan/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

---

## 4. 数据准备流程

### 4.1 RoboTwin → VideoX-Fun 训练数据

训练数据已经准备好：

```text
/data/alice/cjtest/datasets/worldarena_wan_i2v_clean50/
├── train/                 # 2500 个 MP4
├── metadata.json          # VideoX-Fun 训练元数据
├── validation_examples.json
└── quality_report.json
```

如果要重新生成，参考根目录下的数据脚本：

```bash
python /data/alice/cjtest/convert_robotwin_to_videox_fun.py
python /data/alice/cjtest/prepare_wan_lora_training.py
```

### 4.2 WorldArena test dataset

1000 个推理样本位于：

```text
/data/alice/cjtest/VideoX-Fun/test_dataset/
├── raw/instructions_2/fixed_scene_task/     # 1000 个 JSON 指令
├── raw/first_frame/fixed_scene_task/        # 1000 个首帧 PNG
├── validation/validation_config.json        # VideoX-Fun 推理配置
└── manifests_1000_gpu0567_fastresume/       # 多 GPU 分片 manifest
```

---

## 5. LoRA 训练流程

训练入口：

```bash
cd /data/alice/cjtest/VideoX-Fun
export CUDA_VISIBLE_DEVICES=6,7
export PYTHONNOUSERSITE=1
bash scripts/wan2.1/wait_and_train_worldarena_wan_i2v_lora.sh
```

关键配置：

| 参数 | 当前值 |
|---|---|
| Base model | `Wan2.1-I2V-14B-480P` |
| Framework | VideoX-Fun |
| Train data | 2500 RoboTwin videos |
| LoRA rank | 32 |
| Precision | bf16 |
| GPUs | 2 GPUs |
| Save interval | 每 100 steps |
| 当前最新 checkpoint | `checkpoint-2200.safetensors` |

监控训练：

```bash
tail -f /data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/train.log
ls /data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-*.safetensors | tail
```

---

## 6. 推理流程

### 6.1 单个 manifest 推理

核心脚本：

```text
/data/alice/cjtest/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py
```

示例：

```bash
cd /data/alice/cjtest/VideoX-Fun
export CUDA_VISIBLE_DEVICES=5
export PYTHONNOUSERSITE=1

/data/envs/videox_fun_wan/bin/python scripts/wan2.1/batch_predict_i2v_worldarena.py \
  --manifest /data/alice/cjtest/VideoX-Fun/test_dataset/manifests_1000_gpu0567_fastresume/part_01.json \
  --dataset-root /data/alice/cjtest/VideoX-Fun/test_dataset \
  --base-model /data/alice/cjtest/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P \
  --lora-path /data/alice/cjtest/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/checkpoint-2200.safetensors \
  --output-dir /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw \
  --enable-teacache \
  --teacache-threshold 0.20 \
  --gpu-memory-mode none
```

### 6.2 1000-video 多 GPU 推理

当前采用多 manifest 分片，并且脚本支持“已存在则跳过”，所以中断后可以继续跑。

主要输出：

```text
/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw/
```

查看进度：

```bash
find /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_raw -maxdepth 1 -name '*.mp4' | wc -l
tail -f /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_finalize.log
```

当前检查点：

```text
raw inference: 953 / 1000
SeedVR refined: 952 / 1000
```

---

## 7. SeedVR 后处理流程

SeedVR 脚本会循环监听 raw 视频目录，把新增视频软链到输入目录，并选择空闲 GPU 做 refinement。

入口：

```bash
bash /data/alice/cjtest/VideoX-Fun/seedvr_eval_ckpt_latest_test1000.sh
```

输入 / 输出：

```text
输入：/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr_in
输出：/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr
日志：/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr.log
```

查看进度：

```bash
find /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr -maxdepth 1 -name '*.mp4' | wc -l
tail -f /data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_seedvr.log
```

注意：SeedVR 主要提高视觉清晰度和减少 artifact，不会从根本上修复动作语义错误。

---

## 8. 自动收尾与评测流程

自动收尾脚本：

```bash
bash /data/alice/cjtest/VideoX-Fun/finish_eval_ckpt_latest_test1000.sh
```

它会循环等待 raw MP4 达到 1000 个，然后执行：

1. 复制 / flatten 视频到 `eval_ckpt_latest_test1000_flat/`
2. 合并 manifest：`eval_ckpt_latest_test1000_manifest.json`
3. 生成 VLM summary：`eval_ckpt_latest_test1000_summary_vlm.json`
4. 构造 WorldArena generated dataset
5. 运行标准视觉指标：image quality、aesthetic、background consistency、dynamic degree、flow score、subject consistency
6. 运行 VLM judge：interaction quality、perspectivity、instruction following
7. 评测结束后自动恢复 LoRA 训练

关键输出：

```text
/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_flat
/data/alice/cjtest/VideoX-Fun/eval_ckpt_latest_test1000_generated_dataset
/data/alice/cjtest/VideoX-Fun/metrics_output_ckpt_latest_test1000
/data/alice/cjtest/VideoX-Fun/output_VLM_ckpt_latest_test1000
```

---

## 9. 展示网站运行流程

网站目录：

```bash
cd /data/alice/cjtest/github-pages-site
```

提取展示视频：

```bash
bash extract_videos.sh
```

本地预览：

```bash
python3 -m http.server 8000
```

当前 Kimaki tunnel 预览：

```text
https://ad0a87d8a00e83ac86be-8000-tunnel.kimaki.dev
```

页面主要文件：

```text
index.html                 # 展示页主体
css/style.css              # 样式
videos/*.mp4               # 展示视频
docs/code_run_flow.md      # 本文档
docs/reproduction_guide.md # 英文复现指南
```

---

## 10. 常见问题

### 10.1 网站打不开

可能原因：

1. GitHub Pages 没有在仓库 Settings → Pages 中启用。
2. 静态服务器没有启动。
3. `videos/` 目录为空，页面能开但视频无法播放。

快速修复：

```bash
cd /data/alice/cjtest/github-pages-site
bash extract_videos.sh
python3 -m http.server 8000
```

如果需要公网预览，用 Kimaki tunnel 暴露端口 8000。

### 10.2 推理中断

重新运行同一个 manifest 即可；脚本会跳过已经生成的 MP4。

### 10.3 显存不足

优先尝试：

```text
--enable-teacache --teacache-threshold 0.20
--gpu-memory-mode sequential_cpu_offload
```

### 10.4 flash-attn 编译很久

这是正常现象。H200 / CUDA 多架构编译会生成大量 CUDA kernel object，可能耗时很久。推理流程不强依赖它，若已能跑通，可以先不阻塞主流程。

---

## 11. 一句话交接版

如果只需要继续当前任务：

1. 等 `eval_ckpt_latest_test1000_raw` 达到 1000 个 MP4。
2. 等 `eval_ckpt_latest_test1000_seedvr` 达到 1000 个 MP4。
3. 确认 `finish_eval_ckpt_latest_test1000.sh` 自动进入 metrics / VLM 阶段。
4. 评测完成后，把最终指标更新到 `github-pages-site/index.html`。
5. 重新运行 `python3 -m http.server 8000` 或部署到 GitHub Pages。
