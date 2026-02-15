"""
CE-VAE with TUDA Feature-Level Alignment.

Extends the base CE-VAE model with a feature-level discriminator from TUDA
that encourages domain-invariant encoder features. This improves
generalization to diverse real-world underwater environments.

Key changes from vanilla CE-VAE:
1. Adds a FeatureLevelDiscriminator on encoder outputs
2. Training step alternates: standard reconstruction + feature alignment
3. Supports freezing early encoder layers for efficient fine-tuning
4. At inference, identical to vanilla CE-VAE (no overhead)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.base import BaseModel, _torch_imgs_to_np
from src.modules.autoencoder import Encoder, Decoder
from src.modules.capsules import PrimaryCaps, DigitCaps
from src.modules.discriminator.feature_discriminator import (
    FeatureLevelDiscriminator,
    compute_gradient_penalty
)
from src.data.real_underwater_dataset import UnpairedRealUnderwaterDataset
from src.metrics import compute as compute_metrics
import logging

logger = logging.getLogger(__name__)


class CEVAE_TUDA(BaseModel):
    """CE-VAE enhanced with TUDA feature-level domain adaptation."""

    def __init__(self,
                 ddconfig: dict,
                 lossconfig: dict = None,
                 embed_dim: int = 256,
                 optimizer: dict = None,
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 image_key: str = "image",
                 monitor: str = None,
                 discriminator: bool = True,
                 # TUDA alignment params
                 feature_alignment_weight: float = 0.0005,
                 feature_disc_start: int = 0,
                 gradient_penalty_weight: float = 10.0,
                 feature_disc_ndf: int = 128,
                 freeze_encoder_blocks: int = 0,
                 real_images_list_file: str = None,
                 real_images_dir: str = None,
                 real_batch_size: int = 4,
                 ):
        super(CEVAE_TUDA, self).__init__(lossconfig=lossconfig,
                                         image_key=image_key,
                                         ignore_keys=ignore_keys, monitor=monitor)

        # --- Standard CE-VAE components ---
        self._use_capsules = ddconfig.get("use_capsules", True)
        self.encoder = Encoder(**ddconfig)
        self.primary = PrimaryCaps() if self._use_capsules else nn.Sequential()
        self.digitcaps = DigitCaps(ddconfig["z_channels"]) if self._use_capsules else nn.Sequential()
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim,
                                          1) if self._use_capsules else nn.Sequential()
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"],
                                               1) if self._use_capsules else nn.Sequential()
        self.discriminator = discriminator

        # --- TUDA Feature Alignment components ---
        self._feat_align_weight = feature_alignment_weight
        self._feat_disc_start = feature_disc_start
        self._gp_weight = gradient_penalty_weight
        self._freeze_encoder_blocks = freeze_encoder_blocks
        self._real_images_list_file = real_images_list_file
        self._real_images_dir = real_images_dir
        self._real_batch_size = real_batch_size

        # Feature discriminator
        feat_channels = ddconfig.get("z_channels", 256)
        self.feature_disc = FeatureLevelDiscriminator(
            in_channels=feat_channels, ndf=feature_disc_ndf
        )

        # Real data iterator (set up in on_fit_start)
        self._real_dataloader = None
        self._real_iter = None

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Freeze early encoder blocks if specified
        if freeze_encoder_blocks > 0:
            self._freeze_early_encoder(freeze_encoder_blocks)

        # Optimizer config
        self.optimizer_config = {'beta1': 0.5, 'beta2': 0.9, 'learning_rate': 1e-4}
        if optimizer:
            self.optimizer_config |= dict(optimizer)
            if "base_learning_rate" in optimizer:
                self.optimizer_config["learning_rate"] = optimizer["base_learning_rate"]
        self.automatic_optimization = False

    def _freeze_early_encoder(self, n_blocks: int):
        """Freeze the first n encoding blocks for efficient fine-tuning."""
        # Freeze conv_in
        for param in self.encoder.conv_in.parameters():
            param.requires_grad = False

        # Freeze first n down blocks
        for i in range(min(n_blocks, len(self.encoder.down))):
            for param in self.encoder.down[i].parameters():
                param.requires_grad = False

        frozen_params = sum(1 for p in self.encoder.parameters() if not p.requires_grad)
        total_params = sum(1 for p in self.encoder.parameters())
        logger.info(f"Froze {frozen_params}/{total_params} encoder parameters "
                    f"({n_blocks} blocks)")

    def encode(self, x):
        enc = self.encoder(x)
        if self._use_capsules:
            x = self.primary(enc)
            _, x = self.digitcaps(x)
            x = self.quant_conv(x)
        return enc, x

    def decode(self, enc, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(enc, quant)
        return dec

    def forward(self, x):
        enc, quant = self.encode(x)
        dec = self.decode(enc, quant)
        return dec

    def on_fit_start(self):
        """Set up the real underwater image dataloader when training starts."""
        if self._real_images_list_file is not None or self._real_images_dir is not None:
            real_dataset = UnpairedRealUnderwaterDataset(
                images_list_file=self._real_images_list_file,
                images_dir=self._real_images_dir,
                size=256,
                random_crop=True,
                random_flip=True,
                max_size=288,
            )
            self._real_dataloader = DataLoader(
                real_dataset,
                batch_size=self._real_batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True,
                persistent_workers=False,
            )
            self._real_iter = iter(self._real_dataloader)
            logger.info(f"Loaded {len(real_dataset)} unpaired real underwater images "
                        f"for feature alignment")

    def _get_real_batch(self):
        """Get next batch of real underwater images, cycling through dataset."""
        if self._real_dataloader is None:
            return None
        try:
            batch = next(self._real_iter)
        except StopIteration:
            self._real_iter = iter(self._real_dataloader)
            batch = next(self._real_iter)
        return batch

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        y = self.get_input(batch, 'target')
        bs = x.shape[0]

        xrec = self(x)

        optimizers = self.optimizers()

        # We have 3 optimizers: [generator, disc (optional), feature_disc]
        if isinstance(optimizers, (list, tuple)):
            optimizer_g = optimizers[0]
            has_disc = self.discriminator and len(optimizers) >= 2
            # Feature disc optimizer is always last
            optimizer_feat_d = optimizers[-1] if len(optimizers) >= 2 else None
            if has_disc:
                optimizer_d = optimizers[1]
                if len(optimizers) >= 3:
                    optimizer_feat_d = optimizers[2]
                else:
                    optimizer_feat_d = None
            else:
                optimizer_d = None
        else:
            optimizer_g = optimizers
            optimizer_d = None
            optimizer_feat_d = None

        # ===== Step 1: Train generator (standard CE-VAE losses) =====
        self.toggle_optimizer(optimizer_g)

        # Standard reconstruction loss
        g_loss, log_dict = self.loss(y, xrec, 0, self.global_step,
                                     last_layer=self.get_last_layer(), split="train")

        # Feature alignment loss (generator side)
        feat_align_loss = torch.tensor(0.0, device=x.device)
        if self._real_dataloader is not None and self.global_step >= self._feat_disc_start:
            real_batch = self._get_real_batch()
            if real_batch is not None:
                real_imgs = real_batch["image"]
                if len(real_imgs.shape) == 3:
                    real_imgs = real_imgs[..., None]
                real_imgs = real_imgs.permute(0, 3, 1, 2).to(x.device).float()

                # Align batch sizes (paired batch may differ from real batch)
                min_bs = min(x.shape[0], real_imgs.shape[0])

                enc_paired = self.encoder(x[:min_bs])  # Features from paired data
                enc_real = self.encoder(real_imgs[:min_bs])  # Features from real data

                # Generator wants feature disc to think paired features look like real
                # (minimize domain gap by fooling discriminator)
                feat_pred_paired = self.feature_disc(enc_paired)
                feat_align_loss = -torch.mean(feat_pred_paired)

                # Scale and add to generator loss
                feat_align_loss = self._feat_align_weight * feat_align_loss
                g_loss = g_loss + feat_align_loss

                log_dict["train/feat_align_loss"] = feat_align_loss.detach()

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True,
                      on_epoch=True, batch_size=bs, sync_dist=True)
        self.manual_backward(g_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # ===== Step 2: Train standard discriminator (if enabled) =====
        if self.discriminator and optimizer_d is not None:
            self.toggle_optimizer(optimizer_d)
            d_loss, log_dict_disc = self.loss(y, xrec, 1, self.global_step,
                                              last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True,
                          on_epoch=True, batch_size=bs, sync_dist=True)
            self.manual_backward(d_loss)
            torch.nn.utils.clip_grad_norm_(self.loss.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

        # ===== Step 3: Train feature discriminator =====
        if (optimizer_feat_d is not None and
                self._real_dataloader is not None and
                self.global_step >= self._feat_disc_start):

            self.toggle_optimizer(optimizer_feat_d)

            # Get features (detached from encoder graph)
            with torch.no_grad():
                real_batch = self._get_real_batch()
                if real_batch is not None:
                    real_imgs = real_batch["image"]
                    if len(real_imgs.shape) == 3:
                        real_imgs = real_imgs[..., None]
                    real_imgs = real_imgs.permute(0, 3, 1, 2).to(x.device).float()
                    # Align batch sizes (paired batch may differ from real batch)
                    min_bs = min(x.shape[0], real_imgs.shape[0])
                    enc_paired = self.encoder(x[:min_bs]).detach()
                    enc_real = self.encoder(real_imgs[:min_bs]).detach()
                else:
                    enc_paired = self.encoder(x).detach()
                    enc_real = enc_paired  # Fallback

            # WGAN loss: maximize D(real) - D(paired)
            pred_real = self.feature_disc(enc_real)
            pred_paired = self.feature_disc(enc_paired)

            # Wasserstein loss
            feat_d_loss = torch.mean(pred_paired) - torch.mean(pred_real)

            # Gradient penalty (both tensors now have same batch size)
            gp = compute_gradient_penalty(
                self.feature_disc, enc_real, enc_paired, x.device
            )
            feat_d_loss = feat_d_loss + self._gp_weight * gp

            feat_d_log = {
                "train/feat_d_loss": feat_d_loss.detach(),
                "train/feat_d_pred_real": pred_real.detach().mean(),
                "train/feat_d_pred_paired": pred_paired.detach().mean(),
                "train/feat_d_gp": gp.detach(),
            }
            self.log_dict(feat_d_log, prog_bar=False, logger=True, on_step=True,
                          on_epoch=True, batch_size=bs, sync_dist=True)

            self.manual_backward(feat_d_loss)
            torch.nn.utils.clip_grad_norm_(self.feature_disc.parameters(), max_norm=1.0)
            optimizer_feat_d.step()
            optimizer_feat_d.zero_grad()
            self.untoggle_optimizer(optimizer_feat_d)

    def configure_optimizers(self):
        lr = self.optimizer_config['learning_rate']
        betas = (self.optimizer_config['beta1'], self.optimizer_config['beta2'])

        # Generator parameters (encoder + decoder + capsules)
        params_to_optimize = list(self.encoder.parameters()) + \
                             list(self.decoder.parameters()) + \
                             list(self.post_quant_conv.parameters())

        if self._use_capsules:
            params_to_optimize += list(self.primary.parameters()) + \
                                  list(self.digitcaps.parameters()) + \
                                  list(self.quant_conv.parameters())

        # Only include unfrozen params
        params_to_optimize = [p for p in params_to_optimize if p.requires_grad]

        optimizers = [
            torch.optim.Adam(params_to_optimize, lr=lr, betas=betas)
        ]

        # Standard discriminator (if enabled)
        if self.discriminator and self.loss is not None and self.loss.discriminator is not None:
            optimizers.append(
                torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=betas)
            )

        # Feature discriminator (always added)
        optimizers.append(
            torch.optim.Adam(self.feature_disc.parameters(), lr=lr * 2, betas=betas)
        )

        return optimizers, []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def validation_step(self, batch, batch_idx):
        """Standard validation - no alignment needed."""
        x = self.get_input(batch, self.image_key)
        y = self.get_input(batch, 'target')
        bs = x.shape[0]

        xrec = self(x)
        g_loss, log_dict = self.loss(y, xrec, 0, self.global_step,
                                     last_layer=self.get_last_layer(), split="val")

        # Metrics
        y_np = _torch_imgs_to_np(y)
        xrec_np = _torch_imgs_to_np(xrec)

        rec_metrics = {'val/psnr': 0, 'val/ssim': 0, 'val/uiqm': 0, 'val/uciqe': 0}
        for rec, gt in zip(xrec_np, y_np):
            res = compute_metrics(rec, gt)
            for k in ['psnr', 'ssim', 'uiqm', 'uciqe']:
                rec_metrics[f'val/{k}'] += res[k]

        rec_metrics = {k: torch.tensor(v / bs, device=x.device) for k, v in rec_metrics.items()}
        self.log_dict(log_dict | rec_metrics, logger=True, on_step=True,
                      on_epoch=True, batch_size=bs, sync_dist=True)

        return self.log_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
