# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CE-VAE (Capsule Enhanced Variational AutoEncoder) for underwater image enhancement. Combines VAE architecture with Capsule Networks and optional TUDA (domain adaptation) for real-world generalization. Based on the WACV 2024 paper by Pucci & Martinel.

## Commands

### Training
```bash
# Basic end-to-end training
python main.py --config configs/cevae_E2E_lsui.yaml

# With TUDA domain adaptation
python main.py --config configs/cevae_E2E_lsui_tuda.yaml

# GAN finetuning (requires checkpoint from E2E phase)
python main.py --config configs/cevae_GAN_lsui.yaml
```

### Inference
```bash
python test.py --config configs/cevae_E2E_lsui.yaml \
  --checkpoint path/to/checkpoint.ckpt \
  --data-path path/to/input/images \
  --output-path path/to/output
```

### Dataset Preparation
```bash
bash scripts/generate_dataset_txt.sh /path/to/LSUI/dataset
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Training Pipeline
Three-phase training: E2E reconstruction → optional GAN refinement → optional TUDA alignment. Configs in `configs/` control which phase runs. Training uses PyTorch Lightning (`main.py` entry point).

### Model (`src/models/`)
- **`cevae.py`** — Core model: Encoder → PrimaryCaps → DigitCaps → Decoder with dual pathways (spatial detail + capsule-based entity structure). Encoder compresses 256×256×3 → 256×16×16.
- **`cevae_tuda.py`** — Extends CEVAE with feature-level adversarial domain alignment using unpaired real underwater images.
- **`base.py`** — PyTorch Lightning base with alternating generator/discriminator training steps.

### Modules (`src/modules/`)
- **`autoencoder/`** — Encoder (multi-resolution downsample + attention) and Decoder (progressive upsample + skip connections + capsule routing).
- **`capsules/`** — PrimaryCaps and DigitCaps with dynamic routing (3 iterations, Wasserstein distance).
- **`losses/combined.py`** — Multi-component loss: L1 pixel + perceptual (LPIPS) + gradient domain + color + MS-SSIM. Weights configurable per-loss.
- **`discriminator/`** — PatchGAN (for GAN phase) and feature-level discriminator with WGAN-GP (for TUDA).

### Data (`src/data/`)
- **`image_enhancement.py`** — Paired training data with augmentations (crop, flip, jitter). Images normalized to [-1, 1].
- **`real_underwater_dataset.py`** — Unpaired real underwater images for TUDA (no ground truth needed).
- **`base.py`** — Test/validation paired dataset (no augmentation).

### Configuration System
YAML configs parsed via OmegaConf. `src/build/from_config.py` provides factory pattern (`instantiate_from_config`) for dynamic class instantiation from config `target` strings.

### Metrics (`src/metrics/`)
Reference: PSNR, SSIM, LPIPS. Non-reference (underwater-specific): UIQM, UCIQE, NIQE.

## Key Details
- Multi-GPU training supported; learning rate auto-scales by GPU count.
- W&B integration for experiment tracking.
- Checkpoints and logs go to `training_logs/`.
- Model outputs go to `output/`.
