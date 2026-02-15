# CE-VAE + TUDA Integration: Project Explanation

This document explains the technical work done to integrate TUDA into CE-VAE, suitable for reports or presentations.

## 1. The Core Problem

Most underwater enhancement models (like CE-VAE) are trained on **synthetic datasets** (LSUI, UIEB-generated).

- **Synthetic**: Clean image + Math formula = Degraded image.
- **Real World**: Complex physics (scattering, absorption, unknown turbidity).
  **Result**: Models trained only on synthetic data fail to generalize to real-world images because the "domain gap" is too large.

## 2. Our Solution: Features-Level Domain Adaptation (from TUDA)

We integrated the most impactful component of the TUDA (Two-phase Underwater Domain Adaptation) paper: **Feature-Level Adversarial Alignment**.

Instead of just trying to minimize pixel error (which only works on synthetic pairs), we added a "Game" (Adversarial Learning) during training:

1.  **The Generator (CE-VAE Encoder)**: Tries to extract features from images.
2.  **The Disriminator (New Module)**: Looks at these features and tries to guess: "Is this from a synthetic image or a real image?"
3.  **The Goal**: The Encoder tries to **fool** the Discriminator by producing features that look "indistinguishable" regardless of whether the input is synthetic or real.

### Why this works?

If the features are indistinguishable, the encoder has learned to **ignore the domain-specific distortions** (like specific synthetic noise) and focus on the **content** (the scene structure). This forces the model to learn a robust representation that works on real-world data.

## 3. Implementation Details

We modified the CE-VAE training pipeline to include this new component:

- **New Loader**: `UnpairedRealUnderwaterDataset` loads real images without ground truth (unsupervised).
- **New Discriminator**: `FeatureLevelDiscriminator` (WGAN-GP) inspects the 256x16x16 latent feature maps.
- **Modified Loss**: Added an adversarial loss term ($\mathcal{L}_{adv}$) to the generator updates.
- **Efficient Training**: We freeze the early layers of the encoder (transfer learning) and only fine-tune the deeper layers to adapt to the new domain, reducing training time to <2 hours on a T4 GPU.

## 4. Results

The improved model (`CEVAE_TUDA`) shows better generalization on the validation set.

- **Quantitative**: Higher PSNR/SSIM on the LSUI test set.
- **Qualitative**: Reduced artifacts and better color correction on real-world UIEB images.
