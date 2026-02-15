"""
Smoke test for TUDA integration into CE-VAE.
Verifies all new modules import, instantiate, and run correctly on CPU.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import traceback

def test_feature_discriminator():
    """Test the feature-level discriminator module."""
    print("=" * 60)
    print("TEST 1: Feature-Level Discriminator")
    print("=" * 60)
    
    from src.modules.discriminator.feature_discriminator import (
        FeatureLevelDiscriminator, compute_gradient_penalty
    )
    
    # Create discriminator
    disc = FeatureLevelDiscriminator(in_channels=256, ndf=128)
    print(f"  ✅ Instantiated FeatureLevelDiscriminator")
    
    # Count parameters
    params = sum(p.numel() for p in disc.parameters())
    print(f"  Parameters: {params:,} ({params * 4 / 1024**2:.1f} MB)")
    
    # Forward pass with dummy encoder features (256×16×16)
    dummy_features = torch.randn(2, 256, 16, 16)
    output = disc(dummy_features)
    print(f"  Input shape:  {dummy_features.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    print(f"  ✅ Forward pass works!")
    
    # Test gradient penalty
    real_feat = torch.randn(2, 256, 16, 16, requires_grad=True)
    fake_feat = torch.randn(2, 256, 16, 16, requires_grad=True)
    gp = compute_gradient_penalty(disc, real_feat, fake_feat, 'cpu')
    print(f"  Gradient penalty: {gp.item():.4f}")
    print(f"  ✅ Gradient penalty works!")
    
    return True


def test_unpaired_dataset():
    """Test the unpaired real underwater dataset."""
    print("\n" + "=" * 60)
    print("TEST 2: Unpaired Real Underwater Dataset")
    print("=" * 60)
    
    from src.data.real_underwater_dataset import UnpairedRealUnderwaterDataset
    
    # Create a temporary directory with dummy images
    import tempfile
    from PIL import Image
    import numpy as np
    
    tmpdir = tempfile.mkdtemp()
    n_images = 6
    for i in range(n_images):
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        img.save(os.path.join(tmpdir, f"test_{i}.jpg"))
    
    # Test directory mode
    dataset = UnpairedRealUnderwaterDataset(
        images_dir=tmpdir,
        size=256,
        random_crop=True,
        random_flip=True,
        max_size=288
    )
    print(f"  ✅ Dataset created with {len(dataset)} images")
    
    # Test __getitem__
    sample = dataset[0]
    print(f"  Sample image shape: {sample['image'].shape}")
    print(f"  Sample value range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    assert sample['image'].shape == (256, 256, 3), f"Expected (256, 256, 3), got {sample['image'].shape}"
    assert sample['image'].min() >= -1.0 and sample['image'].max() <= 1.0
    print(f"  ✅ Dataset loading works!")
    
    # Test with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    print(f"  Batch image shape: {batch['image'].shape}")
    print(f"  ✅ DataLoader works!")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    return True


def test_cevae_tuda_model():
    """Test the TUDA-enhanced CE-VAE model."""
    print("\n" + "=" * 60)
    print("TEST 3: CEVAE_TUDA Model (Full Integration)")
    print("=" * 60)
    
    from src.models.cevae_tuda import CEVAE_TUDA
    
    # Create dummy real images directory
    import tempfile
    from PIL import Image
    import numpy as np
    
    tmpdir = tempfile.mkdtemp()
    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        img.save(os.path.join(tmpdir, f"real_{i}.jpg"))
    
    # Model config matching the YAML
    ddconfig = {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
    }
    
    lossconfig = {
        "target": "src.modules.losses.combined.ReconstructionLoss",
        "params": {
            "pixelloss_weight": 10.0,
            "perceptual_weight": 0.0,  # Disable LPIPS for CPU test (needs VGG)
            "gdl_loss_weight": 0.0,
            "ssim_loss_weight": 0.0,
        }
    }
    
    model = CEVAE_TUDA(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        embed_dim=256,
        discriminator=False,
        feature_alignment_weight=0.0005,
        feature_disc_start=0,
        gradient_penalty_weight=10.0,
        feature_disc_ndf=128,
        freeze_encoder_blocks=3,
        real_images_dir=tmpdir,
        real_batch_size=2,
    )
    print(f"  ✅ CEVAE_TUDA model instantiated")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Frozen params:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # Forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, f"Expected {dummy_input.shape}, got {output.shape}"
    print(f"  ✅ Forward pass works!")
    
    # Test encoder feature extraction
    with torch.no_grad():
        enc = model.encoder(dummy_input)
    print(f"  Encoder output shape: {enc.shape}")
    print(f"  ✅ Encoder produces expected features!")
    
    # Test feature discriminator on encoder output
    with torch.no_grad():
        feat_pred = model.feature_disc(enc)
    print(f"  Feature disc prediction shape: {feat_pred.shape}")
    print(f"  ✅ Feature discriminator works on encoder output!")
    
    # Test configure_optimizers
    optimizers, schedulers = model.configure_optimizers()
    print(f"  Number of optimizers: {len(optimizers)}")
    print(f"  ✅ Optimizers configured!")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    return True


def test_with_real_checkpoint():
    """Test loading the TUDA model with the real CE-VAE checkpoint."""
    print("\n" + "=" * 60)
    print("TEST 4: Loading Real Checkpoint into CEVAE_TUDA")
    print("=" * 60)
    
    ckpt_path = "data/lsui-cevae-epoch119.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️ Checkpoint not found at {ckpt_path}, skipping")
        return True
    
    from src.models.cevae_tuda import CEVAE_TUDA
    
    ddconfig = {
        "double_z": False,
        "z_channels": 256,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
    }
    
    model = CEVAE_TUDA(
        ddconfig=ddconfig,
        embed_dim=256,
        ckpt_path=ckpt_path,
        discriminator=False,
        feature_alignment_weight=0.0005,
        freeze_encoder_blocks=3,
    )
    print(f"  ✅ Loaded checkpoint into CEVAE_TUDA!")
    
    # Run on test images if available
    test_img_dir = "test_images"
    if os.path.exists(test_img_dir):
        from PIL import Image
        import numpy as np
        
        img_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.png', '.jpg'))]
        print(f"  Found {len(img_files)} test images")
        
        model.eval()
        for img_file in img_files[:6]:
            img_path = os.path.join(test_img_dir, img_file)
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            img_np = np.array(img).astype(np.float32)
            img_tensor = torch.from_numpy((img_np / 127.5 - 1.0)).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            # Save enhanced image
            os.makedirs("output_tuda_test", exist_ok=True)
            out_np = torch.clamp(output.squeeze(0), -1, 1)
            out_np = ((out_np + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            Image.fromarray(out_np).save(f"output_tuda_test/{img_file}")
            print(f"  ✅ Processed {img_file} -> output_tuda_test/{img_file}")
    else:
        print(f"  ⚠️ No test_images directory found, skipping inference test")
    
    return True



def test_learning_rate_config():
    """Test that base_learning_rate is correctly adopted."""
    print("\n" + "=" * 60)
    print("TEST 3.5: Learning Rate Configuration")
    print("=" * 60)
    
    from src.models.cevae_tuda import CEVAE_TUDA
    
    # Minimal dummy config
    ddconfig = {
        "double_z": False, "z_channels": 256, "resolution": 256,
        "in_channels": 3, "out_ch": 3, "ch": 128,
        "ch_mult": [1], "num_res_blocks": 1, "attn_resolutions": [],
        "dropout": 0.0,
    }
    
    # Test case: optimizer has base_learning_rate
    optimizer_cfg = {"base_learning_rate": 4.5e-6}
    
    model = CEVAE_TUDA(
        ddconfig=ddconfig,
        embed_dim=256,
        discriminator=False,
        optimizer=optimizer_cfg
    )
    
    lr = model.optimizer_config.get("learning_rate")
    print(f"  Config passed: {optimizer_cfg}")
    print(f"  Model learning_rate: {lr}")
    
    assert lr == 4.5e-6, f"Expected 4.5e-6, got {lr}"
    print(f"  ✅ Learning rate correctly adopted from base_learning_rate!")
    
    return True


if __name__ == "__main__":
    print("🔧 TUDA Integration Smoke Test")
    print("Running on CPU — verifying everything works before Kaggle/Colab\n")
    
    results = {}
    
    for test_name, test_fn in [
        ("Feature Discriminator", test_feature_discriminator),
        ("Unpaired Dataset", test_unpaired_dataset),
        ("CEVAE_TUDA Model", test_cevae_tuda_model),
        ("Learning Rate Config", test_learning_rate_config),
        ("Real Checkpoint", test_with_real_checkpoint),
    ]:
        try:
            results[test_name] = test_fn()
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status} — {name}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! Ready for Kaggle/Colab training.")
    else:
        print("\n⚠️ Some tests failed. Fix issues before training.")
