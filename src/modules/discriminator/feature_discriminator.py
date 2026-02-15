"""
Feature-Level Discriminator for TUDA-style domain adaptation.

Operates on encoder feature maps (256×16×16) to distinguish between
features from paired training data vs. unpaired real underwater images.
This encourages the encoder to produce domain-invariant features.

Uses spectral normalization for stable training (as recommended in
WGAN-GP and modern GAN literature).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormConv2d(nn.Module):
    """Conv2d with spectral normalization for GAN training stability."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.conv(x)


class FeatureLevelDiscriminator(nn.Module):
    """
    Discriminates between encoder features from paired (LSUI) data
    and unpaired real underwater images.
    
    Input: Feature maps of shape (B, C, H, W) — typically (B, 256, 16, 16)
    Output: Scalar prediction per sample
    
    Architecture: 3 conv layers with spectral norm + global average pooling + FC
    Small and fast — adds minimal overhead to training.
    """
    
    def __init__(self, in_channels: int = 256, ndf: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, 256, 16, 16) -> (B, 128, 8, 8)
            SpectralNormConv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 128, 8, 8) -> (B, 256, 4, 4)
            SpectralNormConv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 256, 4, 4) -> (B, 256, 2, 2)
            SpectralNormConv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Global average pool + linear to scalar
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 2, 1))
        )
    
    def forward(self, features):
        """
        Args:
            features: Encoder feature maps (B, C, H, W)
        Returns:
            Scalar prediction (B, 1) — higher = "looks like paired data"
        """
        h = self.net(features)
        return self.classifier(h)


def compute_gradient_penalty(discriminator, real_features, fake_features, device):
    """
    WGAN-GP gradient penalty for stable adversarial training.
    Interpolates between real and fake features, computes discriminator
    output, and penalizes gradients that deviate from norm 1.
    """
    batch_size = real_features.size(0)
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolated = (alpha * real_features + (1 - alpha) * fake_features).requires_grad_(True)
    
    # Discriminator output on interpolated data
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
