"""
Print parameter counts for all 3 options:
  1. Full ViT (9 classes)
  2. ViT + LoRA + SupCon
  3. ViT + LoRA (no SupCon)

Run: python verify_params.py
Or in Kaggle/Jupyter: %run verify_params.py
"""
import torch
from timm.models import create_model
import models

adapt_blocks = list(range(12))
num_classes = 9
projection_dim = 128


def apply_lora_freezing(model, use_supcon=True):
    """Apply same freezing logic as main.py (LoRA mode)."""
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        elif 'head' in n:
            # Matches classifier head and projection_head; main.py freezes bias
            if 'bias' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
        elif 'norm' in n:
            p.requires_grad = True


# =============================================================================
# OPTION 1: Full ViT (9 classes)
# =============================================================================
model_full = create_model(
    'vit_base_patch16_224_in21k',
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.1,
    adapt_blocks=[],
    use_lora=False,
)
n_full_total = sum(p.numel() for p in model_full.parameters())
n_full_trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)

print("=" * 65)
print("OPTION 1: Full ViT (9 classes)")
print("=" * 65)
print(f"  Total:      {n_full_total:>12,}  ({n_full_total/1e6:.2f}M)")
print(f"  Trainable:  {n_full_trainable:>12,}  ({n_full_trainable/1e6:.2f}M)")
print(f"  Frozen:     {n_full_total - n_full_trainable:>12,}")
print()

# =============================================================================
# OPTION 2: ViT + LoRA + SupCon
# =============================================================================
model_supcon = create_model(
    'vit_base_patch16_224_in21k',
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.1,
    adapt_blocks=adapt_blocks,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
)
model_supcon.init_projection_head(projection_dim=projection_dim)
apply_lora_freezing(model_supcon, use_supcon=True)

n_supcon_total = sum(p.numel() for p in model_supcon.parameters())
n_supcon_trainable = sum(p.numel() for p in model_supcon.parameters() if p.requires_grad)
n_supcon_frozen = n_supcon_total - n_supcon_trainable

print("=" * 65)
print("OPTION 2: ViT + LoRA + SupCon")
print("=" * 65)
print(f"  Total:      {n_supcon_total:>12,}  ({n_supcon_total/1e6:.2f}M)")
print(f"  Trainable:  {n_supcon_trainable:>12,}  ({n_supcon_trainable/1e6:.2f}M)")
print(f"  Frozen:     {n_supcon_frozen:>12,}  ({n_supcon_frozen/1e6:.2f}M)")
print()

# =============================================================================
# OPTION 3: ViT + LoRA (no SupCon)
# =============================================================================
model_lora = create_model(
    'vit_base_patch16_224_in21k',
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.1,
    adapt_blocks=adapt_blocks,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,
)
apply_lora_freezing(model_lora, use_supcon=False)

n_lora_total = sum(p.numel() for p in model_lora.parameters())
n_lora_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
n_lora_frozen = n_lora_total - n_lora_trainable

print("=" * 65)
print("OPTION 3: ViT + LoRA (no SupCon)")
print("=" * 65)
print(f"  Total:      {n_lora_total:>12,}  ({n_lora_total/1e6:.2f}M)")
print(f"  Trainable:  {n_lora_trainable:>12,}  ({n_lora_trainable/1e6:.2f}M)")
print(f"  Frozen:     {n_lora_frozen:>12,}  ({n_lora_frozen/1e6:.2f}M)")
print()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("=" * 65)
print("SUMMARY FOR WRITEUP")
print("=" * 65)
print(f"{'Option':<30} {'Total':>12} {'Trainable':>14}")
print("-" * 65)
print(f"{'1. Full ViT (9 classes)':<30} {n_full_total/1e6:>10.2f}M {n_full_trainable/1e6:>12.2f}M")
print(f"{'2. ViT + LoRA + SupCon':<30} {n_supcon_total/1e6:>10.2f}M {n_supcon_trainable/1e6:>12.2f}M")
print(f"{'3. ViT + LoRA (no SupCon)':<30} {n_lora_total/1e6:>10.2f}M {n_lora_trainable/1e6:>12.2f}M")
print("=" * 65)
