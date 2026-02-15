# CE-VAE + TUDA: Complete Technical Explanation

> This document explains everything about our project — the problem, the theory, exactly what code we wrote, how training works, and what improvements we expect. Written for a professor-level audience in simple, accessible language.

---

## 1. The Core Problem: Domain Gap

### What is CE-VAE?

CE-VAE (Capsule-Enhanced Variational Autoencoder) is an underwater image enhancement model. It takes a degraded underwater image (blurry, greenish/bluish tint, low contrast) and outputs a clean, enhanced version.

**Architecture (simplified):**

```
Degraded Image → [Encoder] → [Capsule Network] → [Decoder] → Enhanced Image
                    ↓
           256×16×16 feature map
           (compressed representation)
```

- The **Encoder** reduces a 256×256×3 image down to a 256×16×16 feature map (65,536 dimensions)
- The **Capsule Network** (PrimaryCaps + DigitCaps) refines these features to capture spatial relationships
- The **Decoder** reconstructs the enhanced image from the capsule-refined features

### Why does CE-VAE struggle on real-world images?

CE-VAE is trained on the **LSUI dataset** — a collection of paired images:

- **Input**: Degraded underwater image
- **Target**: Corresponding clean image (manually collected by divers at close range)

The problem is that LSUI images, while real, represent a **specific distribution** of underwater conditions. When the model encounters images from different oceans, depths, or turbidity levels, it often produces artifacts because it has **never seen** these conditions during training.

This is called the **domain gap** — the statistical difference between training data and real-world data.

```
Training Data (LSUI)          Real World
├── Specific ocean             ├── Any ocean
├── Specific depth range       ├── Any depth
├── Specific camera            ├── Any camera
└── ~3,800 paired images       └── Infinite variety
         ↓                              ↓
    Model learns these           Model fails on these
    conditions well              unseen conditions
```

---

## 2. Our Solution: Feature-Level Domain Adaptation (from TUDA)

### What is TUDA?

TUDA stands for **Two-phase Underwater Domain Adaptation** (from the research paper). It addresses the domain gap by forcing the model to learn **domain-invariant features** — representations that look the same regardless of whether the input is from the training set or the real world.

### The Key Insight

Instead of trying to fix the problem at the **pixel level** (which requires paired data), we fix it at the **feature level**. The encoder's 256×16×16 feature map should encode **scene content** (what objects are there, their structure, lighting relationships) and NOT encode **domain-specific information** (this is from LSUI, this is from the Pacific Ocean, etc.).

### How it works — The Adversarial Game

We add a new component: a **Feature-Level Discriminator**. This creates an adversarial game (similar to GANs):

```
                                        ┌─────────────────────────┐
  LSUI Image ──→ [Encoder] ──→ Features ──→ [Feature Discriminator]
                                   ↑           "Is this from LSUI
  Real Image ──→ [Encoder] ──→ Features ──→    or real world?"
                                        └─────────────────────────┘
```

**Two players:**

1. **The Encoder** (Generator): Tries to produce features that the discriminator **cannot distinguish** — it wants features from LSUI images and real images to look identical.

2. **The Feature Discriminator** (Critic): Tries to **correctly classify** whether features came from an LSUI image or a real underwater image.

**The game:**

- The discriminator gets better at telling them apart → The encoder must try harder to make them similar
- The encoder gets better at fooling the discriminator → The discriminator must try harder to tell them apart
- At **equilibrium**: The encoder produces features that are truly domain-invariant

### Why WGAN-GP (Wasserstein GAN with Gradient Penalty)?

We don't use a standard GAN loss. Instead, we use **WGAN-GP** which has two advantages:

1. **Wasserstein distance**: Instead of binary classification (real/fake), the discriminator outputs a continuous score. This provides smooth, meaningful gradients throughout training — no mode collapse.

2. **Gradient Penalty (GP)**: Instead of weight clipping (which can cause vanishing gradients), we penalize the discriminator if its gradients deviate from norm 1. This keeps training stable.

The gradient penalty formula:

```
GP = E[(||∇D(x̃)||₂ - 1)²]

where x̃ = α·x_real + (1-α)·x_fake  (random interpolation)
```

---

## 3. Implementation Details — What Code We Wrote

### 3.1 New Files Created

We created **3 new files** and modified **2 existing files**:

#### File 1: `src/modules/discriminator/feature_discriminator.py`

**Purpose**: The Feature-Level Discriminator — the "critic" in our adversarial game.

```python
# Architecture (simplified):
class FeatureLevelDiscriminator(nn.Module):
    # Input:  (Batch, 256, 16, 16) — encoder feature maps
    # Output: (Batch, 1)           — scalar "realness" score

    def __init__(self, in_channels=256, ndf=128):
        self.net = nn.Sequential(
            # Conv 256→128, stride 2: (B,256,16,16) → (B,128,8,8)
            SpectralNormConv2d(256, 128, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),

            # Conv 128→256, stride 2: (B,128,8,8) → (B,256,4,4)
            SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),

            # Conv 256→256, stride 2: (B,256,4,4) → (B,256,2,2)
            SpectralNormConv2d(256, 256, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),
        )
        # Global Average Pool → Linear → scalar
        self.classifier = nn.Sequential(
            AdaptiveAvgPool2d(1),  # (B,256,2,2) → (B,256,1,1)
            Flatten(),             # (B,256)
            Linear(256, 1)         # (B,1)
        )
```

**Key design choices:**

- **Spectral Normalization** on all layers: Controls the Lipschitz constant of the discriminator, essential for WGAN stability
- **Only ~2.1M parameters**: Small and efficient — minimal GPU overhead
- **3 convolutional layers**: Enough capacity to distinguish domains without being too powerful (which would overwhelm the encoder)

We also implemented the **gradient penalty function**:

```python
def compute_gradient_penalty(discriminator, real_features, fake_features, device):
    # 1. Random interpolation between real and fake features
    alpha = torch.rand(batch_size, 1, 1, 1)
    interpolated = alpha * real_features + (1 - alpha) * fake_features

    # 2. Forward pass on interpolated features
    d_output = discriminator(interpolated)

    # 3. Compute gradients of output w.r.t. interpolated input
    gradients = torch.autograd.grad(d_output, interpolated, ...)

    # 4. Penalize if gradient norm deviates from 1
    penalty = ((gradients.norm(2) - 1) ** 2).mean()
    return penalty
```

#### File 2: `src/data/real_underwater_dataset.py`

**Purpose**: Loads unpaired real underwater images (no ground truth needed).

```python
class UnpairedRealUnderwaterDataset(Dataset):
    # Loads 890 real underwater images from UIEB dataset
    # Applies: resize, random crop (256×256), random flip
    # Normalizes to [-1, 1] range
    # Returns: {"image": normalized_image}
```

**Why unpaired?** This is the beauty of our approach — we don't need clean/degraded pairs for the real images. We only need real underwater images to teach the encoder what "real world features look like." The enhancement quality still comes from the LSUI paired training.

#### File 3: `src/models/cevae_tuda.py`

**Purpose**: The main model — extends CE-VAE with the TUDA training loop.

This is the most important file. It inherits from the base CE-VAE model and overrides the training step with our **3-phase optimization**:

```python
class CEVAE_TUDA(BaseModel):
    def __init__(self, ...):
        # Standard CE-VAE components (Encoder, Decoder, Capsules)
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.primary = PrimaryCaps()
        self.digitcaps = DigitCaps(...)

        # NEW: Feature discriminator
        self.feature_disc = FeatureLevelDiscriminator(in_channels=256, ndf=128)

        # NEW: Freeze early encoder layers (transfer learning)
        self._freeze_early_encoder(n_blocks=3)  # Freeze first 3 of 5 blocks

        # Use manual optimization (we control gradient flow ourselves)
        self.automatic_optimization = False
```

### 3.2 The 3-Phase Training Step (The Heart of Our Implementation)

Every training iteration runs 3 phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONE TRAINING STEP                            │
│                                                                 │
│  Phase 1: GENERATOR UPDATE                                      │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ 1. Forward pass: x → Encoder → Capsules → Decoder → x̂│      │
│  │ 2. Compute reconstruction loss:                       │      │
│  │    L_rec = 10·L1 + 1·LPIPS + 1·SSIM                  │      │
│  │ 3. Get encoder features for LSUI images               │      │
│  │ 4. Feed features to discriminator                     │      │
│  │ 5. Generator wants to FOOL discriminator:             │      │
│  │    L_feat = -0.0005 · mean(D(features_paired))        │      │
│  │ 6. Total: L_gen = L_rec + L_feat                      │      │
│  │ 7. Backprop + clip gradients + update encoder/decoder │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  Phase 2: STANDARD DISCRIMINATOR (disabled in our config)       │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ Skipped — we set disc_enabled: False because          │      │
│  │ the feature discriminator is sufficient                │      │
│  └───────────────────────────────────────────────────────┘      │
│                                                                 │
│  Phase 3: FEATURE DISCRIMINATOR UPDATE                          │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ 1. Get encoder features for LSUI images (detached)    │      │
│  │ 2. Get encoder features for real images (detached)    │      │
│  │ 3. Discriminator scores both:                         │      │
│  │    D(real_features) and D(paired_features)            │      │
│  │ 4. WGAN loss: L = mean(D(paired)) - mean(D(real))     │      │
│  │ 5. Gradient penalty: GP on interpolated features      │      │
│  │ 6. Total: L_disc = L + 10·GP                          │      │
│  │ 7. Backprop + clip gradients + update discriminator   │      │
│  └───────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**Why 3 separate phases?** In adversarial training, you cannot update the generator and discriminator simultaneously — they have opposing objectives. The standard approach is to alternate updates. We use PyTorch Lightning's "manual optimization" to control this precisely.

### 3.3 Configuration File: `configs/cevae_E2E_lsui_tuda.yaml`

```yaml
model:
  target: src.models.cevae_tuda.CEVAE_TUDA
  params:
    ckpt_path: data/lsui-cevae-epoch119.ckpt # Start from pre-trained CE-VAE
    feature_alignment_weight: 0.0005 # λ₄ from TUDA paper
    feature_disc_start: 0 # Start alignment immediately
    gradient_penalty_weight: 10.0 # Standard WGAN-GP value
    freeze_encoder_blocks: 3 # Freeze first 3 of 5 blocks
    real_images_list_file: data/real_underwater_images.txt # 890 UIEB images
    real_batch_size: 4 # Per-step real image batch

lightning:
  trainer:
    max_epochs: 30 # Fine-tuning doesn't need many epochs
    precision: 16-mixed # Half-precision for 2x speed + half VRAM
    devices: 1 # Single T4 GPU
```

---

## 4. Training Strategy — Why We Made These Choices

### 4.1 Transfer Learning (Not Training from Scratch)

We do NOT train a new model from scratch. We:

1. **Start from the pre-trained CE-VAE** (epoch 119 checkpoint — already trained for 119 epochs on LSUI)
2. **Freeze the first 3 encoder blocks** (out of 5) — these capture low-level features (edges, textures) that are already well-learned
3. **Only fine-tune** the last 2 encoder blocks + decoder + capsules + new discriminator

This gives us:

- **81.4M trainable parameters** (out of 100M total)
- **18.9M frozen parameters** (early encoder + LPIPS loss network)
- Training time: **~8 hours** on a free T4 GPU (vs. days from scratch)

### 4.2 The Loss Function Breakdown

The total generator loss combines multiple terms:

```
L_total = L_reconstruction + L_feature_alignment

Where:
  L_reconstruction = 10·L₁ + 1·LPIPS + 1·MS-SSIM
                     ↑        ↑         ↑
                     Pixel    Perceptual Structural
                     accuracy similarity similarity

  L_feature_alignment = 0.0005 · (-mean(D(encoder_features)))
                        ↑
                        Very small weight — we don't want to
                        overwhelm the reconstruction objective
```

**Why λ = 0.0005 for alignment?** If this weight is too high, the model prioritizes fooling the discriminator over producing good reconstructions. The TUDA paper recommends this value, and it provides a gentle "nudge" toward domain-invariant features without degrading reconstruction quality.

### 4.3 Gradient Clipping

We clip gradients to `max_norm=1.0` in all three optimization phases. This prevents exploding gradients during adversarial training, which can happen when the discriminator sends very strong gradient signals to the encoder.

### 4.4 Feature Discriminator Learning Rate

The feature discriminator uses **2× the learning rate** of the generator:

```python
# Generator: lr = 2.7e-5
# Feature Discriminator: lr = 5.4e-5 (2× generator)
```

This is standard in GAN training — the discriminator should learn slightly faster to provide meaningful feedback to the generator.

---

## 5. How This Improves CE-VAE Scores

### Before (Vanilla CE-VAE)

- Trained only on LSUI paired data
- Encoder learns LSUI-specific feature representations
- Works well on LSUI test set, but degrades on unseen real-world images
- May produce color artifacts or over-smoothing on diverse underwater conditions

### After (CE-VAE + TUDA)

- Same reconstruction training on LSUI (quality preserved)
- **Additionally**: encoder features are aligned with real-world features
- The encoder learns to produce features that are useful for enhancement regardless of the input domain
- Better generalization = better metrics on diverse test sets

### Expected Improvements

1. **PSNR**: Should maintain or slightly improve on LSUI test set (reconstruction loss is still primary)
2. **SSIM**: Structural similarity should improve as features become more content-focused
3. **UIQM/UCIQE**: Real-world quality metrics should improve most significantly — this is where domain adaptation shines
4. **Visual quality**: Fewer color artifacts, more natural-looking results on real underwater photos

### Why Might Scores _Not_ Improve on LSUI Test Set?

The LSUI test set is from the **same distribution** as the training set. Domain adaptation primarily helps on **out-of-distribution** data. If we test on LSUI only, improvements may be modest because there's no domain gap to bridge. The real benefit shows on diverse real-world images.

---

## 6. What Makes This Different from Just Re-training

| Approach                                   | What it Does                        | Problem                                                           |
| ------------------------------------------ | ----------------------------------- | ----------------------------------------------------------------- |
| Re-training on more data                   | Gets more diverse training pairs    | Requires expensive paired data collection                         |
| Data augmentation                          | Simulates variations artificially   | Doesn't capture real-world distributions                          |
| Image-level transfer                       | Transforms images between domains   | Computationally expensive, can introduce artifacts                |
| **Our approach: Feature-level adaptation** | **Aligns internal representations** | **None of the above — unsupervised, efficient, minimal overhead** |

Our method requires:

- ✅ No new paired data (only unpaired real images)
- ✅ No architectural changes to the inference pipeline
- ✅ No additional compute at inference time (discriminator is removed)
- ✅ Only ~2.1M extra parameters during training

---

## 7. Datasets Used

| Dataset  | Purpose                        | Images                | Type                    |
| -------- | ------------------------------ | --------------------- | ----------------------- |
| **LSUI** | Main training (reconstruction) | 3,879 train + 400 val | Paired (input + target) |
| **UIEB** | Real-world feature alignment   | 890 images            | Unpaired (input only)   |

---

## 8. Technical Summary Table

| Component              | Details                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| **Base Model**         | CE-VAE (Capsule-Enhanced VAE), 100M params                           |
| **New Addition**       | Feature-Level Discriminator (2.1M params)                            |
| **Training Method**    | 3-phase manual optimization (generator, standard disc, feature disc) |
| **Loss Function**      | L₁ + LPIPS + MS-SSIM + WGAN-GP feature alignment                     |
| **Alignment Weight**   | λ = 0.0005 (from TUDA paper)                                         |
| **Gradient Penalty**   | WGAN-GP, weight = 10.0                                               |
| **Encoder Freezing**   | First 3 of 5 blocks frozen (transfer learning)                       |
| **Training Time**      | ~8 hours on T4 GPU (30 epochs)                                       |
| **Inference Overhead** | Zero (discriminator removed at test time)                            |
| **Framework**          | PyTorch + PyTorch Lightning                                          |
| **Precision**          | FP16 mixed precision                                                 |

---

## 9. File Structure — Where Each Piece Lives

```
ce-vae-underwater-image-enhancement/
├── configs/
│   ├── cevae_E2E_lsui.yaml          # Original CE-VAE config
│   └── cevae_E2E_lsui_tuda.yaml     # Our TUDA-enhanced config (NEW)
├── src/
│   ├── models/
│   │   ├── base.py                   # Base model (inherited)
│   │   └── cevae_tuda.py             # Our TUDA model (NEW)
│   ├── modules/
│   │   ├── discriminator/
│   │   │   └── feature_discriminator.py  # Feature discriminator (NEW)
│   │   └── losses/
│   │       ├── combined.py           # Reconstruction + GAN losses
│   │       └── lpips.py              # Perceptual loss
│   └── data/
│       ├── image_enhancement.py      # Paired dataset loader
│       └── real_underwater_dataset.py # Unpaired real image loader (NEW)
├── notebooks/
│   └── train_cevae_tuda.ipynb        # One-click Kaggle training notebook
└── main.py                          # Training entry point
```

---

## 10. References

1. **CE-VAE**: Capsule-Enhanced Variational Autoencoder for Underwater Image Enhancement
2. **TUDA**: Li et al., "Two-phase Underwater Domain Adaptation for Single Underwater Image Enhancement" — Source of the feature-level alignment idea
3. **WGAN-GP**: Gulrajani et al., "Improved Training of Wasserstein GANs" — The gradient penalty method we use
4. **Spectral Normalization**: Miyato et al., "Spectral Normalization for GANs" — Used in our discriminator for training stability
