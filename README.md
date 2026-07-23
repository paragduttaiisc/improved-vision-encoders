# Improved Vision Encoders

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-multi--GPU-orange.svg)](https://huggingface.co/docs/accelerate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Self-supervised vision encoder research — building larger, smarter masked autoencoders for representation learning.

---

## Overview

This project develops improved vision encoders through self-supervised learning. The core approach is a **Joint Embedding Predictive Architecture (JEPA)** that learns rich visual representations by predicting latent features of masked regions from unmasked regions — without any labels.

The encoder produces high-quality patch-level features that are evaluated via a linear probe (top-1 accuracy on ImageNet-1K) and through reconstruction quality metrics (PSNR, MSE, MAE). The goal is to push representation quality beyond current baselines through architectural innovations.

### What's Working

- [x] **JEPA training pipeline** — 75% random patch masking, encoder/predictor with prediction loss + variance loss + sliced Wasserstein loss
- [x] **Integrated linear probe** — probe trained concurrently with JEPA for real-time representation quality tracking
- [x] **Multi-GPU training** — Accelerate-based DDP across 8× NVIDIA A100-SXM4-40GB
- [x] **Validation & logging** — reconstruction metrics (MSE, RMSE, MAE, PSNR) + probe accuracy (Acc@1/5/10)
- [x] **WandB experiment tracking** — loss curves, learning rate schedules, visualization of masked/reconstructed images
- [x] **Linear probe inference** — standalone script that reports Acc@1/5/10 on the held-out test set (no training loop)
- [x] **Checkpointing** — periodic state saving and restoration via `accelerator.save_state()`
- [x] **Configurable sequence mixing** — switch between Multi-Head Attention (FlashAttention) and FourierMixer (FFT-based) via `--token_mixer`
- [x] **Configurable FFN activations** — GELU, SwiGLU, or SqReLU via `--activation`
- [x] **Mixture of Experts (MoE)** — optional MoE in channel mixer with top-k routing and load-balancing aux loss

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: 224×224×3 Image                       │
└─────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Patch Embedding       │  16×16 → n_embed-d
              │   (196 patches)         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Random 75% Masking    │  ← 49 tokens kept
              └────────────┬────────────┘
                           │
         ┌─────────────────▼─────────────────┐
         │         ENCODER                   │
         │   N layers × Configurable Block   │
         │   n_embed-d embed, n_heads        │
         │   RMSNorm + Configurable FFN      │
         │   (GELU / SwiGLU / SqReLU)        │
         │   Token mixer: MHA or Fourier     │
         └─────────────────┬─────────────────┘
                           │
                  [B, 49, n_embed]  ← latent features
                           │
         ┌─────────────────▼─────────────────┐
         │         PREDICTOR                 │
         │   N layers × Configurable Block   │
         │   decoder_n_embed-d embed         │
         │   + mask tokens for 147 missing   │
         └─────────────────┬─────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Predicted Latents  │  [B, 196, decoder_n_embed]
              │  (patch-level output) │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Prediction Loss (MSE)  │
              │  + Variance Loss        │
              │  + Sliced Wasserstein   │
              │  + MoE Aux Loss         │
              │  + Linear Probe (1000)  │
              └─────────────────────────┘
```

### Model Configuration

| Component | Dimensions | Layers | Heads |
|-----------|-----------|--------|-------|
| Encoder | 768-d embed (default) | 12 (default) | 12 (default) |
| Predictor | 256-d embed (default) | 6 (default) | 4 (default) |
| Patch embedding | 16×16×3 → n_embed | — | — |
| Linear probe | n_embed → 1000 | — | — |

**Total parameters**: ~150 MB (encoder: ~140 MB, predictor: ~10 MB) — varies with configuration

---

## Quick Start

### Installation

```bash
conda create -n dl python=3.10 -y
conda activate dl
pip install torch torchvision accelerate wandb einops matplotlib numpy
```

### Data Setup

```bash
# ImageFolder format expected at:
data/imagenet-540k-1k/
├── train/          # 500,000 images (auto-split)
├── val/            # 10,000 images (auto-split)
└── test/           # ~30,000 images (auto-split)
```

The dataloader auto-generates stratified splits (500k train / 10k val / ~30k test) and caches them at `data/imagenet-540k-1k/splits.pt`.

### Training

```bash
# Default training (ImageNet, 500 epochs, 8 GPUs)
python pretrain.py --run-name JEPA-Baseline

# Custom configuration
python pretrain.py \
    --n_layers 24 --n_embed 768 --n_heads 12 \
    --batch_size 512 --num_epochs 100 \
    --lr 1e-4 --weight_decay 0.05 \
    --token_mixer Fourier --activation SwiGLU \
    --n_experts 8 --n_active_experts 2 \
    --run-name JEPA-Large

# Quick test run
python pretrain.py --batch_size 64 --num_epochs 2 --log_interval 1
```

### Linear Probe Inference

```bash
# Report Acc@1/5/10 on the held-out test set (no training)
python infer.py --batch_size 2048
```

---

## Training Pipeline

The training loop runs two objectives simultaneously:

1. **JEPA Prediction** — Random 75% of patches are masked. The encoder processes only the visible patches to produce latent features. The predictor reconstructs all patches (visible + masked) via mask tokens. Loss is a combination of:
   - **Prediction loss**: MSE between predicted and actual patch latents (masked patches only)
   - **Variance loss**: Encourages latent std to match a target (prevents collapse)
   - **Sliced Wasserstein loss**: Aligns latent distribution to Gaussian (prevents collapse)
   - **MoE aux loss**: Load-balancing regularization for mixture of experts

2. **Linear Probe** — Encoder features (mean-pooled across patches) are fed to a linear classifier trained with cross-entropy on image labels. This runs concurrently with JEPA training, providing real-time representation quality signals.

```
for each batch:
    # Forward pass
    patches = patchify(images)
    ids_keep = random_mask(patches, ratio=0.25)
    latents = encoder(patches, ids_keep)
    predicted = predictor(latents, ids_restore)

    # JEPA loss (masked patches only)
    loss_pred = mse(predicted, patches) [masked only]
    loss_var = variance_loss(latents)
    loss_sw = sliced_wasserstein_loss(latents)

    # Linear probe (concurrent)
    features = latents.mean(dim=1)
    logits = probe(features)
    loss_probe = cross_entropy(logits, labels)

    # Backward pass
    backward(loss_pred + loss_var + loss_sw + loss_probe)
```

### Scheduler

- **Warmup**: Linear warmup for 10 epochs (learning rate 1e-3 → 2e-4)
- **Annealing**: Cosine annealing from 2e-4 down to 1e-4 over epochs 10–480
- **Constant**: Learning rate held at 1e-4 for epochs 480–500
- **Gradient clipping**: L2 norm ≤ 1.0 on encoder + decoder parameters

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Reconstruction Loss** | MSE on masked patches (lower = better) |
| **PSNR** | Peak signal-to-noise ratio in dB (higher = better) |
| **MAE** | Mean absolute error on pixel values (lower = better) |
| **Probe Acc@1** | Top-1 linear classification accuracy (higher = better) |
| **Probe Acc@5** | Top-5 linear classification accuracy |
| **Probe Acc@10** | Top-10 linear classification accuracy |

---

## Project Roadmap

### Completed

- [x] **JEPA training pipeline** — 75% random patch masking, encoder/predictor with prediction loss + variance loss + sliced Wasserstein loss
- [x] **Integrated linear probe** — probe trained concurrently with JEPA for real-time representation quality tracking
- [x] **Multi-GPU training** — Accelerate-based DDP across 8× NVIDIA A100-SXM4-40GB
- [x] **Validation & logging** — reconstruction metrics (MSE, RMSE, MAE, PSNR) + probe accuracy (Acc@1/5/10)
- [x] **WandB experiment tracking** — loss curves, LR schedules, visualization of masked/reconstructed images
- [x] **Linear probe inference** — standalone script that reports Acc@1/5/10 on the held-out test set (no training loop)
- [x] **Checkpointing** — periodic state saving and restoration via `accelerator.save_state()`
- [x] **Configurable sequence mixing** — Multi-Head Attention (FlashAttention) or FourierMixer (FFT-based)
- [x] **Configurable FFN activations** — GELU, SwiGLU, or SqReLU
- [x] **Mixture of Experts (MoE)** — optional MoE in channel mixer with top-k routing and load-balancing aux loss

### Phase 1: JEPA Objective

**Completed** — Replaced pixel-wise MSE reconstruction with a joint predictive objective:

- [x] Predict latent representations of masked regions from unmasked regions
- [x] Variance loss to prevent representation collapse
- [x] Sliced Wasserstein loss for Gaussian alignment
- [x] Replaced decoder with predictor network
- [ ] Add teacher-student architecture for predictive coding
- [ ] Add contrastive loss (InfoNCE) between predicted and target latents

**Key papers:**
- [JEPAs](https://arxiv.org/abs/2405.09161) — "Masked Autoencoders Are Spatiotemporally Robust Self-Supervised Learners" (Meta, 2024)
- [SIGReg](https://arxiv.org/abs/2406.08452) — "Self-supervised Learning with Sign Regularization"
- [BEiT-3](https://arxiv.org/abs/2206.07103) — "BERT Meets Vision Encoder: Pre-training for Multi-modal Tasks" (2022)
- [MAE](https://arxiv.org/abs/2111.06377) — "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

### Phase 2: Architectural Extensions

**Completed:**
- [x] Mixture of Experts (MoE) in FFN layers with top-k routing
- [x] FourierMixer (FFT-based sequence mixing) as alternative to MHA
- [x] Configurable FFN activations (GELU, SwiGLU, SqReLU)

**Remaining:**
- [ ] Mixture of Experts in attention layers
- [ ] Sparse / local attention (sliding window, cross-window communication)
- [ ] Multi-scale / hierarchical encoder
- [ ] Cross-attention between patch groups
- [ ] Alternative FFN designs (gMLP, gated MLP)

**Key papers:**
- [MoE](https://arxiv.org/abs/2101.03961) — "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2022)
- [FNO](https://arxiv.org/abs/2010.08898) — "Fourier Neural Operator for Parametric PDEs" (Li et al., 2020)
- [Swin](https://arxiv.org/abs/2103.14030) — "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)
- [gMLP](https://arxiv.org/abs/2105.08050) — "Pay Attention to MLPs: An Alternative to Attention" (Liu et al., 2021)

### Phase 3: Dataset Expansion

- [ ] Ego4D dataset — egocentric video understanding
- [ ] Robotic manipulation datasets (BridgeData, RT-1)
- [ ] Multi-domain pre-training (web + video + robotics)
- [ ] Evaluation benchmarks for each domain

**Key papers:**
- [Ego4D](https://arxiv.org/abs/2110.07058) — "Ego-Exo4D: Understanding Complex Human Activities in the Physical World" (Savarese et al., 2021)
- [BridgeData](https://arxiv.org/abs/2304.13705) — "Learning Robotic Manipulation from Video" (Brohan et al., 2023)
- [RT-1](https://arxiv.org/abs/2304.13705) — "RT-1: Robotics Transformer for Real-World Control at Scale" (Brohan et al., 2022)

### Infrastructure & Misc

- [ ] YAML-based config files (replace argparse)
- [ ] Hyperparameter sweep support (optuna / wandb sweeps)
- [ ] Standardized evaluation harness for all datasets
- [ ] Feature extraction pipeline for downstream tasks
- [ ] Mixed precision improvements (FP8 / NF4 quantization)
- [ ] Architecture diagrams and model card

---

## File Reference

| File | Purpose |
|------|---------|
| [pretrain.py](pretrain.py) | Training loop: JEPA forward/backward, probe training, validation, logging |
| [model.py](model.py) | Model definitions: `MultiHeadAttention`, `FourierMixer`, `FeedForward` (GELU/SwiGLU/SqReLU), `MoE`, `Block`, `BaseModel`, `Encoder`, `Decoder` |
| [losses.py](losses.py) | Regularization losses: `variance_loss_fn`, `sliced_wasserstein_loss_fn` |
| [dataloader.py](dataloader.py) | ImageFolder dataloader with stratified train/val/test splits |
| [utils.py](utils.py) | Helpers: `patchify`, `unpatchify`, `denormalize`, `visualize`, `set_seed` |
| [infer.py](infer.py) | Linear probe inference: freeze encoder, report Acc@1/5/10 on test set |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
