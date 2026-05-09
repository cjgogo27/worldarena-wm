# WorldArena World Model Challenge — Reproduction & Training Guide

> **Project**: Robot Video World Models on WorldArena  
> **GitHub**: https://github.com/cjgogo27/worldarena-wm  
> **HF Models**: https://huggingface.co/cjgogo/models  
> **Project Page**: https://cjgogo27.github.io/worldarena-wm/

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [GigaWorld Reproduction](#4-gigaworld-reproduction)
5. [ABot-PhysWorld Reproduction](#5-abot-physworld-reproduction)
6. [Wan2.1 Baseline Inference](#6-wan21-baseline-inference)
7. [VideoX-Fun + RoboTwin SFT Training](#7-videox-fun--robotwin-sft-training)
8. [SFT-Wan2.1 Inference](#8-sft-wan21-inference)
9. [WorldArena Evaluation](#9-worldarena-evaluation)
10. [SeedVR Post-Processing](#10-seedvr-post-processing)
11. [Reproducing the 1000-Video Inference](#11-reproducing-the-1000-video-inference)
12. [Checkpoints](#12-checkpoints)
13. [Project Structure](#13-project-structure)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Project Overview

This project evaluates and adapts robot video world models on the **WorldArena Challenge** (Track 1: Text-to-Video). We compare three approaches:

| Model | Description | Type |
|-------|-------------|------|
| **ABot-PhysWorld** | Robot-specific world model with 3M video SFT + physics-aware DPO | Robot-oriented |
| **Vanilla Wan2.1** | General-purpose 14B I2V diffusion transformer | General baseline |
| **SFT-Wan2.1 (Ours)** | Wan2.1 fine-tuned with LoRA on 2,500 RoboTwin videos | Adapted model |

**Key Findings**:
- ABot excels at robot identity preservation and background stability
- Vanilla Wan2.1 has superior image quality but suffers from human hand hallucination
- Our SFT-Wan2.1 improves subject consistency and flow score over vanilla Wan2.1
- The gap between "visual quality" and "action correctness" remains significant

---

## 2. Environment Setup

### 2.1 System Requirements

- **GPU**: 1–8 × NVIDIA H200 (80GB+) or A100 (80GB). 14B models require ~70GB VRAM for inference.
- **CUDA**: 12.4+ recommended. Multiple CUDA versions may coexist (see Section 14).
- **Storage**: ~200GB for models, ~100GB for datasets, ~500GB for generated videos.
- **RAM**: 64GB+ recommended.

### 2.2 Base Conda Environments

We use separate conda environments for different components due to dependency conflicts:

```bash
# ---------- Environment 1: ABot-PhysWorld ----------
conda create -n abot_physworld_v2 python=3.10
conda activate abot_physworld_v2
cd /path/to/ABot-PhysWorld
pip install -r requirements.txt
# Note: requires torch with CUDA 12.4+
# If you encounter flash_attn issues, see Troubleshooting section

# ---------- Environment 2: VideoX-Fun (training + inference) ----------
# Follow the setup script:
bash /path/to/VideoX-Fun/scripts/wan2.1/setup_worldarena_wan_env.sh
# This creates a 'videox_fun_wan' environment with all dependencies

# ---------- Environment 3: SeedVR ----------
conda create -n seedvr python=3.10
conda activate seedvr
# Install SeedVR requirements (diffusers, torch, etc.)
# See SeedVR docs for details
```

> **Important**: Never try to unify all dependencies into one environment. Multiple isolated environments are more maintainable.

### 2.3 Flash Attention (Optional but Recommended)

flash_attn accelerates Wan2.1 inference. Install from source:

```bash
conda activate videox_fun_wan

# Set CUDA_HOME to the pip-installed CUDA toolkit matching PyTorch
export CUDA_HOME="$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)) + '/lib/python3.10/site-packages/nvidia/cu13')" 2>/dev/null)"
export CPATH="/usr/local/cuda-12.8/include:$CPATH"
export MAX_JOBS=8

pip install flash-attn --no-build-isolation --no-cache-dir
```

> **Troubleshooting**: If you get `cannot find -lcudart`, symlink `libcudart.so` into the `lib64/` directory:
> ```bash
> ln -sf $CUDA_HOME/lib/libcudart.so.13 $CUDA_HOME/lib64/libcudart.so
> ```
> If you get `CUDA version mismatch`, ensure you use the **environment's pip** (not system pip).

---

## 3. Data Preparation

### 3.1 RoboTwin Dataset (for SFT Training)

The RoboTwin dataset (`aloha-agilex_clean_50`) contains 2,500 robot manipulation videos across 50 tasks:

```bash
# The dataset is already prepared at:
/path/to/datasets/worldarena_wan_i2v_clean50/

# Structure:
#   metadata.json           # Dataset annotations (file_path, text, type, etc.)
#   validation_examples.json
#   quality_report.json
#   video_0.mp4 ... video_2499.mp4
```

**Dataset specs**:
- Source: TianxingChen/RoboTwin2.0
- Resolution: 320×240 @ 30fps (resized to 640×640 for training)
- Tasks: 50 (pick-place, stack, drawer, etc.)
- Videos: 2,500 (50 per task)
- Format: MP4

To prepare the dataset from scratch:

```bash
# Clone the dataset generation repo
# Then run the conversion script to VideoX-Fun format:
python scripts/prepare_robotwin_dataset.py \
  --source /path/to/raw_robotwin \
  --output /path/to/worldarena_wan_i2v_clean50 \
  --resolution 640 \
  --validate
```

### 3.2 WorldArena Test Dataset

For evaluation, we use the WorldArena Track 1 test set:

```bash
# Located at:
/path/to/worldarena_gigaworld_public/datasets/worldarena/test_dataset/

# Structure:
#   test_manifest.json     # All test samples with instructions
#   test/                  # First-frame images
```

---

## 4. GigaWorld Reproduction

### 4.1 Setup

```bash
git clone https://github.com/open-gigaai/giga-world-0.git
cd giga-world-0
git submodule update --init --recursive
pip install -r requirements.txt
```

### 4.2 Download Checkpoint

```bash
python scripts/download.py --model video_gr1
```

### 4.3 Run Inference

```bash
python scripts/inference.py \
  --model video_gr1 \
  --input run_inputs/test_data.json \
  --output run_outputs/video_gr1_run1/
```

### 4.4 Expected Output

A smoke-test video at `run_outputs/video_gr1_run1/0.mp4`.

> **Note**: This is a smoke test, not a full benchmark run. The public GigaWorld checkpoint does not match the closed-source leaderboard results.

---

## 5. ABot-PhysWorld Reproduction

### 5.1 Setup

```bash
git clone https://github.com/amap-cvlab/ABot-PhysWorld.git
cd ABot-PhysWorld
conda create -n abot_physworld_v2 python=3.10
conda activate abot_physworld_v2
pip install -r requirements.txt
```

### 5.2 Download Weights

ABot uses Wan2.1-I2V-14B-480P as base model. Weights are on ModelScope:

```python
# Download script
from modelscope.hub.snapshot_download import snapshot_download

# Base model (Wan2.1 I2V 14B)
snapshot_download('Wan-AI/Wan2.1-I2V-14B-480P',
                  local_dir='models/Wan-AI/Wan2.1-I2V-14B-480P')

# ABot checkpoint
snapshot_download('amap_cvlab/Abot-PhysWorld',
                  local_dir='checkpoints/amap_cvlab/Abot-PhysWorld')
```

### 5.3 Smoke Test Inference

```bash
conda activate abot_physworld_v2

CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
  --base_model models/Wan-AI/Wan2.1-I2V-14B-480P \
  --checkpoint checkpoints/amap_cvlab/Abot-PhysWorld/abotpw_i2v_480p.safetensors \
  --prompt "A robotic arm picks up the object on the table." \
  --image_path assets/demo.jpg \
  --output_dir outputs/smoke/ \
  --num_frames 121 \
  --fps 24
```

### 5.4 WorldArena Formal Inference

```bash
# For Track 1 (Text-to-Video, 10 samples):
CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
  --base_model models/Wan-AI/Wan2.1-I2V-14B-480P \
  --checkpoint checkpoints/amap_cvlab/Abot-PhysWorld/abotpw_i2v_480p.safetensors \
  --manifest /path/to/manifests/test_manifest.json \
  --output_dir outputs/test10_t1_flat/ \
  --num_frames 121 \
  --fps 24 \
  --sample_steps 50
```

> **Key config for WorldArena Track 1**:
> - Frames: 121
> - FPS: 24
> - Resolution: 832×480
> - Steps: 50
> - Action mode: I2V (Image-to-Video)

---

## 6. Wan2.1 Baseline Inference

### 6.1 Setup

```bash
git clone https://github.com/Wan-AI/Wan2.1.git
# Or use the base model directly via our inference wrapper:
# /path/to/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py
```

### 6.2 Run Baseline Inference

```bash
conda activate videox_fun_wan

python /path/to/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py \
  --manifest /path/to/manifests/test_manifest.json \
  --dataset-root /path/to/dataset \
  --base-model /path/to/Wan2.1-I2V-14B-480P \
  --output-dir /path/to/outputs/test10_t1_raw/ \
  --num-frames 121 \
  --sample-steps 50 \
  --enable-teacache \
  --teacache-threshold 0.20
```

### 6.3 TeaCache Acceleration

TeaCache can accelerate inference by ~30-40% with minimal quality loss:

```bash
--enable-teacache --teacache-threshold 0.20
```

Recommended threshold: `0.20` (balance between speed and quality).

---

## 7. VideoX-Fun + RoboTwin SFT Training

### 7.1 Setup VideoX-Fun

```bash
git clone https://github.com/ali-vilab/VideoX-Fun.git
cd VideoX-Fun
bash scripts/wan2.1/setup_worldarena_wan_env.sh
```

This creates a `videox_fun_wan` conda environment.

### 7.2 Training Script

The main training script is:

```bash
# /path/to/VideoX-Fun/scripts/wan2.1/train_worldarena_wan_i2v_lora.sh
```

Key configuration:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base model | `Wan2.1-I2V-14B-480P` | Pretrained I2V model |
| Framework | VideoX-Fun | Training framework |
| Data | RoboTwin aloha-agilex_clean_50 | 2,500 videos, 50 tasks |
| Resolution | 640×640 | Training resolution |
| Frames | 81 | Frames per video |
| Batch size | 1 per GPU | Global batch = num_GPUs |
| GPUs | 2 | CUDA:6,7 recommended |
| LoRA rank | 32 | LoRA decomposition rank |
| LoRA alpha | 16 | LoRA scaling factor |
| LoRA targets | `q,k,v,ffn.0,ffn.2` | Attention + FFN layers |
| Learning rate | 1e-4 | Constant with warmup |
| Warmup steps | 100 | Linear warmup |
| Precision | bf16 | Mixed precision |
| Checkpoint freq | 100 steps | Save interval |
| Max steps | 12,500 | 10 epochs × 2,500 / 2 GPUs |
| Resume | latest | Auto-resume from latest ckpt |

### 7.3 Launch Training

```bash
cd /path/to/VideoX-Fun

# Manual launch:
bash scripts/wan2.1/train_worldarena_wan_i2v_lora.sh

# Or with auto-wait for GPU availability:
bash scripts/wan2.1/wait_and_train_worldarena_wan_i2v_lora.sh
```

The training script automatically detects existing checkpoints and resumes from the latest one.

### 7.4 Monitor Training

```bash
# Training log
tail -f /path/to/output_dir_wan2.1_i2v_robotwin_lora/train.log

# TensorBoard
tensorboard --logdir /path/to/output_dir_wan2.1_i2v_robotwin_lora/logs

# Check progress
grep "Steps:" /path/to/output_dir_wan2.1_i2v_robotwin_lora/train.log | tail -1
# Expected: Steps:  X/12500 [time elapsed<remaining, loss]
```

### 7.5 Training Output Structure

```
/path/to/output_dir_wan2.1_i2v_robotwin_lora/
├── checkpoint-100.safetensors
├── checkpoint-100-compatible_with_comfyui.safetensors
├── checkpoint-200.safetensors
├── ...
├── checkpoint-2200.safetensors          # Latest (243MB)
├── checkpoint-2200-compatible_with_comfyui.safetensors
├── train.log                             # Full training log
├── logs/                                 # TensorBoard logs
├── sample/                               # Validation samples
└── sanity_check/                         # Sanity check outputs
```

### 7.6 Custom Training (Alternative Approach)

If you prefer using a custom diffusers pipeline:

```bash
python /path/to/scripts/wan2.1/train_lora.py \
  --pretrained_model_name_or_path=/path/to/Wan2.1-I2V-14B-480P \
  --train_data_dir=/path/to/dataset \
  --train_data_meta=/path/to/metadata.json \
  --output_dir=/path/to/output \
  --train_batch_size=1 \
  --num_train_epochs=10 \
  --learning_rate=1e-4 \
  --rank=32 \
  --network_alpha=16 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora \
  --mixed_precision="bf16" \
  --gradient_checkpointing
```

> **Warning**: The custom diffusers pipeline was found to be slower and less stable than VideoX-Fun. We recommend using VideoX-Fun for any serious training.

---

## 8. SFT-Wan2.1 Inference

After training, run inference with the SFT LoRA weights:

```bash
conda activate videox_fun_wan

python /path/to/VideoX-Fun/scripts/wan2.1/batch_predict_i2v_worldarena.py \
  --manifest /path/to/manifest.json \
  --dataset-root /path/to/dataset \
  --base-model /path/to/Wan2.1-I2V-14B-480P \
  --lora-path /path/to/checkpoint-2200.safetensors \
  --output-dir /path/to/eval_output/ \
  --num-frames 121 \
  --sample-steps 50 \
  --enable-teacache \
  --teacache-threshold 0.20
```

For different checkpoints, simply change `--lora-path` to point to the desired `.safetensors` file.

---

## 9. WorldArena Evaluation

### 9.1 Standard Metrics

```bash
# Run WorldArena evaluation on generated videos:
python /path/to/worldarena/evaluate.py \
  --generated_dir /path/to/generated_videos \
  --output_dir /path/to/metrics_output/ \
  --manifest /path/to/test_manifest.json
```

This produces:
- `generated_results.json`: Standard metrics (Image Quality, Aesthetic Quality, Background Consistency, Subject Consistency, Flow Score, Dynamic Degree)

### 9.2 VLM Evaluation

```bash
# VLM-based evaluation (InternVL):
python /path/to/worldarena/evaluate_vlm.py \
  --generated_dir /path/to/generated_videos \
  --manifest /path/to/test_manifest.json \
  --output_dir /path/to/vlm_output/
```

This produces:
- `*_summary_val_all_intern.json`: VLM metrics (Interaction Quality, Perspectivity, Instruction Following)

### 9.3 Metrics Reference (10 samples)

| Metric | ABot | Vanilla Wan2.1 | SFT-Wan2.1 (ckpt300) |
|--------|------|----------------|---------------------|
| Image Quality | 55.20 | **70.44** | 48.80 |
| Aesthetic Quality | 38.54 | **45.35** | 40.64 |
| Background Consistency | **91.18** | 69.58 | 90.40 |
| Dynamic Degree | **37.70** | 27.18 | 37.53 |
| Flow Score | 17.96 | 16.18 | **21.29** |
| Subject Consistency | **82.55** | 59.90 | 82.33 |
| Interaction Quality | 54.00 | **58.00** | — |
| Perspectivity | 88.00 | **92.00** | — |
| Instruction Following | 54.00 | 54.00 | — |

---

## 10. SeedVR Post-Processing

### 10.1 Setup

```bash
# SeedVR requires a separate environment
conda create -n seedvr python=3.10
conda activate seedvr
# Install SeedVR dependencies (see SeedVR repo for details)
```

### 10.2 Running SeedVR

We provide an automated loop that monitors the inference output directory:

```bash
bash /path/to/seedvr_eval_ckpt_latest_test1000.sh
```

This script:
1. Symlinks new raw videos into the SeedVR input directory
2. Automatically selects the GPU with the most free memory
3. Runs SeedVR refinement on all pending videos
4. Removes processed symlinks
5. Repeats every 5 minutes until all videos are processed

### 10.3 Manual SeedVR Inference

```bash
cd /path/to/SeedVR
CUDA_VISIBLE_DEVICES=X python projects/inference_seedvr2_3b.py \
  --video_path /path/to/input_videos \
  --output_dir /path/to/output_refined \
  --seed 42 \
  --res_h 480 \
  --res_w 832 \
  --out_fps 24
```

---

## 11. Reproducing the 1000-Video Inference

The 1000-video inference run uses 3 parallel GPU workers for efficiency.

### 11.1 Manifest Structure

Manifests split the 1000 samples into disjoint sets:

```
/path/to/test_dataset/manifests_1000_gpu0567_fastresume/
├── part_00.json              # 90 samples
├── part_01.json              # 90 samples  
├── part_02.json              # 90 samples
├── part_03.json              # 90 samples
├── part_00_overflow_gpu5.json  # 30 samples (overflow)
├── part_00_overflow_gpu6.json  # 30 samples
└── part_00_overflow_gpu7.json  # 29 samples
```

Total: 449 items covering all 1000 samples (some from earlier runs).

### 11.2 Launch Workers

```bash
# Example: Launch worker on GPU 5
export CUDA_VISIBLE_DEVICES=5
python scripts/wan2.1/batch_predict_i2v_worldarena.py \
  --manifest manifests/part_01.json \
  --dataset-root /path/to/dataset \
  --base-model /path/to/Wan2.1-I2V-14B-480P \
  --lora-path /path/to/checkpoint-2200.safetensors \
  --output-dir /path/to/eval_ckpt_latest_test1000_raw \
  --enable-teacache --teacache-threshold 0.20

# Repeat with different manifests on GPUs 6, 7
```

### 11.3 Auto-Skip Logic

The inference script automatically skips already-generated videos. If a worker is restarted, it checks if `output_video` exists before re-generating.

### 11.4 Automatic Finalization

After all 1000 videos are generated:

```bash
bash /path/to/finish_eval_ckpt_latest_test1000.sh
```

This runs:
1. WorldArena evaluation on all 1000 generated videos
2. Training resume (continues SFT from latest checkpoint)

---

## 12. Checkpoints

### 12.1 Hugging Face

Checkpoints are uploaded to [https://huggingface.co/cjgogo/models](https://huggingface.co/cjgogo/models):

| Checkpoint | Steps | Description |
|-----------|-------|-------------|
| `checkpoint-1200.safetensors` | 1,200 | ~1 epoch, earlier training state |
| `checkpoint-2200.safetensors` | 2,200 | Latest, used for 1000-video inference |

### 12.2 Local Checkpoints

All local checkpoints are at:

```
/path/to/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora/
├── checkpoint-100.safetensors through checkpoint-2200.safetensors
├── checkpoint-*-compatible_with_comfyui.safetensors
```

Each checkpoint is a LoRA weight file (~243MB).

---

## 13. Project Structure

```
worldarena-wm/
├── index.html                        # GitHub Pages project page
├── css/style.css                     # Page styles
├── docs/
│   ├── reproduction_guide.md         # This document
│   └── failure_taxonomy.md           # Failure case analysis
├── videos/                           # Showcase videos
│   ├── episode10_abot.mp4
│   ├── episode10_wan.mp4
│   ├── episode10_sft.mp4
│   ├── episode106_abot.mp4
│   ├── episode106_wan.mp4
│   ├── episode106_sft.mp4
│   ├── episode1_wan_fail.mp4
│   ├── episode100_abot_fail.mp4
│   ├── episode105_wan_raw.mp4
│   └── episode105_wan_seedvr.mp4
├── assets/
│   └── keyframes/                    # Keyframe comparisons
├── README.md                         # Quick start
└── extract_videos.sh                 # Video extraction script
```

### 13.1 Local Working Directories (Reference)

These are the actual paths used in our runs. Adapt for your setup.

| Component | Path |
|-----------|------|
| GigaWorld | `/path/to/model_repros/giga-world-0` |
| ABot-PhysWorld | `/path/to/model_repros/ABot-PhysWorld` |
| Wan2.1 base model | `/path/to/model_repros/ABot-PhysWorld/models/Wan-AI/Wan2.1-I2V-14B-480P` |
| WorldArena ABot outputs | `/path/to/model_repros/worldarena_abot_public/outputs/` |
| WorldArena Wan outputs | `/path/to/model_repros/worldarena_wan_public/outputs/` |
| VideoX-Fun | `/path/to/VideoX-Fun` |
| SFT output | `/path/to/VideoX-Fun/output_dir_wan2.1_i2v_robotwin_lora` |
| SFT inference (ckpt300) | `/path/to/VideoX-Fun/eval_ckpt300_test10` |
| 1000-video inference | `/path/to/VideoX-Fun/eval_ckpt_latest_test1000_raw` |
| SeedVR output | `/path/to/VideoX-Fun/eval_ckpt_latest_test1000_seedvr` |
| RoboTwin dataset | `/path/to/datasets/worldarena_wan_i2v_clean50` |
| WorldArena test dataset | `/path/to/datasets/worldarena/test_dataset` |

---

## 14. Troubleshooting

### 14.1 CUDA Version Mismatch

**Problem**: `RuntimeError: The detected CUDA version mismatches the version that was used to compile PyTorch`

**Solution**: The environment has multiple CUDA toolkits. Check which CUDA version your PyTorch expects, and set `CUDA_HOME` accordingly:
```bash
python -c "import torch; print(torch.version.cuda)"
# Set CUDA_HOME to the matching toolkit
export CUDA_HOME=/path/to/cuda-X.X
```

If you have a pip-installed CUDA toolkit (e.g., `nvidia/cu13`), use that:
```bash
export CUDA_HOME=$(python -c "import nvidia.cu13; import os; print(os.path.dirname(nvidia.cu13.__file__))")
```

### 14.2 flash_attn Build Failures

**Problem 1**: `cannot find -lcudart`
**Fix**: Symlink `libcudart.so` to where the linker looks:
```bash
ln -sf $CUDA_HOME/lib/libcudart.so.13 $CUDA_HOME/lib64/libcudart.so
```

**Problem 2**: CUDA version check fails during pip install
**Fix**: Use the **environment's pip** explicitly:
```bash
/path/to/env/bin/pip install flash-attn --no-build-isolation
```

**Problem 3**: Missing thrust headers
**Fix**: Add system CUDA include path:
```bash
export CPATH=/usr/local/cuda-12.8/include:$CPATH
```

### 14.3 OOM During Inference

**Problem**: `CUDA out of memory` when loading 14B model.

**Solutions**:
1. Enable CPU offloading:
```python
pipe.enable_model_cpu_offload()
```
2. Use `sequential_cpu_offload` for tighter memory
3. Enable TeaCache: `--enable-teacache --teacache-threshold 0.20`
4. Reduce batch size to 1

### 14.4 LoRA Not Taking Effect

**Problem**: Training loss decreases but LoRA weights don't affect output.

**Check**: Verify that LoRA parameters are actually being trained:
```python
# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable}/{total} ({100*trainable/total:.2f}%)")
```

If `trainable` is 0, check:
- The correct target module names (`pipe.dit` not `pipe.transformer` for Wan2.1 in diffusers)
- LoRA is applied before the training loop starts

### 14.5 Multi-GPU Cache Conflicts

**Problem**: Multiple inference workers crash on shared memory.

**Fix**: Set independent cache directories per worker:
```python
# In inference script
os.environ['WAN_CACHE_DIR'] = f'/tmp/wan_cache_gpu_{gpu_id}'
```

### 14.6 SeedVR Config Not Found

**Problem**: `FileNotFoundError: '/home/user/configs_3b/main.yaml'`

**Fix**: Run SeedVR from its project root directory:
```bash
cd /path/to/SeedVR && python projects/inference_seedvr2_3b.py ...
```

### 14.7 Training Interruption and Resume

**Problem**: Training was killed and needs to resume.

**Solution**: The training script supports `--resume_from_checkpoint latest`:
```bash
bash train_worldarena_wan_i2v_lora.sh
# This auto-detects the latest checkpoint and resumes
```

To manually specify a checkpoint:
```bash
--resume_from_checkpoint /path/to/checkpoint-1200
```

---

## Appendix A: Quick Command Reference

```bash
# ABot Smoke Test
conda activate abot_physworld_v2
python inference/inference.py --base_model models/Wan-AI/Wan2.1-I2V-14B-480P --checkpoint checkpoints/amap_cvlab/Abot-PhysWorld/abotpw_i2v_480p.safetensors --prompt "pick up object" --image_path assets/demo.jpg --output_dir outputs/smoke/ --num_frames 121 --fps 24

# Wan2.1 Baseline
conda activate videox_fun_wan
python scripts/wan2.1/batch_predict_i2v_worldarena.py --manifest manifests/test.json --dataset-root /path/to/data --base-model /path/to/Wan2.1-I2V-14B-480P --output-dir outputs/wan_baseline/ --enable-teacache --teacache-threshold 0.20

# SFT Training
bash scripts/wan2.1/train_worldarena_wan_i2v_lora.sh

# SFT Inference (with LoRA)
python scripts/wan2.1/batch_predict_i2v_worldarena.py --manifest manifests/test.json --dataset-root /path/to/data --base-model /path/to/Wan2.1-I2V-14B-480P --lora-path /path/to/checkpoint-2200.safetensors --output-dir outputs/sft_wan/ --enable-teacache --teacache-threshold 0.20

# SeedVR Post-Processing
bash seedvr_eval_ckpt_latest_test1000.sh

# WorldArena Evaluation
python path/to/worldarena/evaluate.py --generated_dir outputs/ --output_dir metrics/

# View Training Progress
grep "Steps:" train.log | tail -1
tensorboard --logdir logs/
```

---

## Appendix B: File Sizes & Storage

| Item | Size |
|------|------|
| Wan2.1-I2V-14B-480P (base model) | ~45GB |
| LoRA checkpoint (single) | ~243MB |
| Single video (480p, 121 frames) | ~250-350KB |
| 1000 videos | ~300-350MB |
| SeedVR refined video | ~500KB-1MB |
| Full project dataset (RoboTwin) | ~5GB |

---

*Last updated: May 2026*
