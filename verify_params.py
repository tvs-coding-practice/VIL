"""
Cross-check parameter counts for the writeup.
Run with: python verify_params.py
Requires: torch, timm
"""
import torch
from timm.models import create_model
import models

# Config matching user's run
adapt_blocks = list(range(12))
num_classes = 9

# --- 1. Full ViT (no LoRA, no SupCon) ---
model_full = create_model(
    'vit_base_patch16_224_in21k',
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.1,
    adapt_blocks=[],
    use_lora=False,
)
n_full = sum(p.numel() for p in model_full.parameters())
print("=" * 60)
print("1. FULL ViT (no LoRA, 9 classes)")
print("=" * 60)
print(f"   Total parameters:           {n_full:,} ({n_full/1e6:.2f}M)")
print()

# --- 2. ViT + LoRA (no SupCon) ---
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
n_lora_params = sum(p.numel() for n, p in model_lora.named_parameters() if 'lora_' in n)
n_lora_total = sum(p.numel() for p in model_lora.parameters())
n_lora_base = n_lora_total - n_lora_params

print("2. ViT + LoRA (no SupCon)")
print("=" * 60)
print(f"   LoRA parameters:            {n_lora_params:,}")
print(f"   Base ViT (frozen):          {n_lora_base:,} ({n_lora_base/1e6:.2f}M)")
print(f"   Total parameters:          {n_lora_total:,} ({n_lora_total/1e6:.2f}M)")
print()

# --- 3. Trainable breakdown (ViT + LoRA, with SupCon) ---
# Add projection head (same as main.py)
model_lora.init_projection_head(projection_dim=128)
n_proj = sum(p.numel() for p in model_lora.projection_head.parameters())

# Freeze everything, then unfreeze LoRA, head, projection, norms (same as main.py)
for p in model_lora.parameters():
    p.requires_grad = False
for n, p in model_lora.named_parameters():
    if 'lora_' in n:
        p.requires_grad = True
    elif 'head' in n and 'bias' not in n:
        p.requires_grad = True
    elif 'projection_head' in n:
        p.requires_grad = True
    elif 'norm' in n:
        p.requires_grad = True

n_trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
n_frozen = sum(p.numel() for p in model_lora.parameters() if not p.requires_grad)
n_total_supcon = sum(p.numel() for p in model_lora.parameters())

# Breakdown of trainable
n_train_lora = sum(p.numel() for n, p in model_lora.named_parameters() if 'lora_' in n and p.requires_grad)
n_train_head = sum(p.numel() for n, p in model_lora.named_parameters() if 'head' in n and 'projection' not in n and p.requires_grad)
n_train_proj = sum(p.numel() for n, p in model_lora.named_parameters() if 'projection_head' in n and p.requires_grad)
n_train_norm = sum(p.numel() for n, p in model_lora.named_parameters() if 'norm' in n and p.requires_grad)

print("3. ViT + LoRA + SupCon")
print("=" * 60)
print(f"   Trainable - LoRA:           {n_train_lora:,}")
print(f"   Trainable - Head:           {n_train_head:,}")
print(f"   Trainable - Projection:      {n_train_proj:,}")
print(f"   Trainable - Norms:          {n_train_norm:,}")
print(f"   Total trainable:            {n_trainable:,} ({n_trainable/1e6:.2f}M)")
print(f"   Frozen:                     {n_frozen:,} ({n_frozen/1e6:.2f}M)")
print(f"   Total parameters:           {n_total_supcon:,} ({n_total_supcon/1e6:.2f}M)")
print(f"   Check (frozen+trainable):   {n_frozen + n_trainable:,}")
print()

# --- 4. ViT + LoRA without SupCon ---
print("4. ViT + LoRA (no SupCon) - derived")
print("=" * 60)
print(f"   Total (no projection):      {n_lora_total:,} ({n_lora_total/1e6:.2f}M)")
print(f"   Trainable without SupCon:   {n_trainable - n_train_proj:,} ({(n_trainable - n_train_proj)/1e6:.2f}M)")
print()

# --- Summary ---
print("SUMMARY FOR WRITEUP")
print("=" * 60)
print(f"  Full ViT:                    {n_full:,} total ({n_full/1e6:.2f}M) - 100% trainable")
print(f"  ViT+LoRA+SupCon:             {n_total_supcon:,} total ({n_total_supcon/1e6:.2f}M)")
print(f"    - Frozen:                  {n_frozen:,} ({n_frozen/1e6:.2f}M)")
print(f"    - Trainable:               {n_trainable:,} ({n_trainable/1e6:.2f}M)")
print(f"  ViT+LoRA (no SupCon):        {n_lora_total:,} total ({n_lora_total/1e6:.2f}M)")
print(f"    - Remove projection:       -{n_train_proj:,} (-{n_train_proj/1e6:.2f}M)")
