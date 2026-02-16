"""
t-SNE Visualization for SupCon Loss: Proving Domain Alignment

Generates two plots to demonstrate that SupCon Loss aligns features across medical domains:
- Plot A (Baseline): Without SupCon - NIH and CheXpert samples for same disease form separate clusters (Domain Gap)
- Plot B (With SupCon): Same disease from different domains merges into one tight cluster

Usage:
  # First train two models: one without SupCon, one with SupCon (after all 6 tasks)
  # Then run this script with both checkpoint paths:

  python visualize_tsne_supcon.py \
    --data_path /kaggle/working/MedicalCXR_data \
    --checkpoint_baseline ./output/checkpoint/task6_checkpoint.pth \
    --checkpoint_supcon ./output_supcon/checkpoint/task6_checkpoint.pth \
    --output_dir ./tsne_plots \
    --max_samples_per_class 150 \
    --seed 42

  # If you only have one checkpoint (e.g., SupCon model):
  python visualize_tsne_supcon.py \
    --data_path /path/to/data \
    --checkpoint_supcon ./output/checkpoint/task6_checkpoint.pth \
    --output_dir ./tsne_plots
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from timm.models import create_model
from torch.utils.data import DataLoader, ConcatDataset
from datasets import (
    build_transform,
    MEDICAL_CLASS_MAP,
    RemappedSubset,
)
from continual_datasets.continual_datasets import MedicalCXR


class DomainLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset to add domain label to each sample."""
    def __init__(self, base_dataset, domain_name):
        self.base_dataset = base_dataset
        self.domain_name = domain_name

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return x, y, self.domain_name


def build_cross_domain_tsne_dataloader(data_path, max_samples_per_class=200, seed=42):
    """
    Build a dataloader with samples from classes that appear in MULTIPLE domains.
    Focus: Cardiomegaly(2), Pneumothorax(3) from NIH vs CheXpert (same disease, different domains)
    Also: Effusion(6), Infiltration(5), Nodule(7) from NIH vs Brachio
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = build_transform(False, type('Args', (), {'use_supcon': False})())  # Val transform, no SupCon

    dataset_train_list = MedicalCXR(data_path, train=True, transform=transform, mode='vil').data
    dataset_val_list = MedicalCXR(data_path, train=False, transform=transform, mode='vil').data

    domain_idx_map = {'NIH': 0, 'Brachio': 1, 'Chexpert': 2}
    all_datasets = []

    # Tasks with cross-domain overlap (for domain alignment proof)
    # Task 1 (NIH): Cardiomegaly(2), Pneumothorax(3)
    # Task 5 (CheXpert): Cardiomegaly(2), Pneumothorax(3), No_Finding(8)
    # Task 3 (NIH): Effusion(6), Nodule(7)
    # Task 4 (Brachio): Effusion(6), Infiltration(5), Nodule(7)
    cross_domain_tasks = [
        (1, 'NIH', [2, 3]),           # Cardiomegaly, Pneumothorax
        (5, 'Chexpert', [2, 3, 8]),   # Same diseases + No_Finding
        (3, 'NIH', [6, 7]),           # Effusion, Nodule
        (4, 'Brachio', [5, 6, 7]),    # Effusion, Infiltration, Nodule
    ]

    for task_idx, domain_key, global_ids in cross_domain_tasks:
        d_idx = domain_idx_map[domain_key]
        full_val = dataset_val_list[d_idx]
        source_class_to_idx = full_val.class_to_idx

        # Get source indices for these classes (by name)
        id_to_name = {v: k for k, v in MEDICAL_CLASS_MAP.items()}
        target_classes = [id_to_name[gid] for gid in global_ids if gid in id_to_name]
        target_source_indices = [source_class_to_idx[c] for c in target_classes if c in source_class_to_idx]

        if not target_source_indices:
            continue

        val_indices = [i for i, (_, label) in enumerate(full_val.imgs) if label in target_source_indices]
        np.random.shuffle(val_indices)

        # Limit samples per task for manageable t-SNE
        max_for_task = max_samples_per_class * len(global_ids)  # Approx
        val_indices = val_indices[:min(len(val_indices), max_for_task)]

        val_subset = torch.utils.data.Subset(full_val, val_indices)
        val_subset = RemappedSubset(val_subset, source_class_to_idx, MEDICAL_CLASS_MAP)
        wrapped = DomainLabelDataset(val_subset, domain_key)
        all_datasets.append(wrapped)

    if not all_datasets:
        raise ValueError("No cross-domain data found. Check data_path.")

    combined = ConcatDataset(all_datasets)
    loader = DataLoader(combined, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    return loader


@torch.no_grad()
def extract_features(model, dataloader, device, use_projection_head=True):
    """Extract features: projection head output (SupCon) or pooled backbone (baseline)."""
    all_features = []
    all_labels = []
    all_domains = []

    model.eval()
    for batch in dataloader:
        if len(batch) == 3:
            images, labels, domains = batch
        else:
            images, labels = batch
            domains = ['Unknown'] * len(labels)

        images = images.to(device)
        if isinstance(images, list):
            images = images[0]  # SupCon returns [v1, v2]; use first view for eval

        features = model.forward_features(images)

        if use_projection_head and hasattr(model, 'forward_projection') and model.projection_head is not None:
            try:
                feats = model.forward_projection(features)
            except Exception:
                feats = features[:, 0] if features.dim() == 3 else features.mean(dim=1)
        else:
            feats = features[:, 0]  # CLS token

        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        all_domains.extend(domains if isinstance(domains[0], str) else [str(d) for d in domains])

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels, all_domains


def create_model_and_load(args, checkpoint_path, device):
    """Create model matching training config and load checkpoint."""
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        adapt_blocks=args.adapt_blocks,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    if args.use_supcon and hasattr(model, 'init_projection_head'):
        model.init_projection_head(projection_dim=args.projection_dim)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    return model


def plot_tsne(features, labels, domains, title, output_path, class_names=None):
    """Generate t-SNE plot colored by domain (NIH vs Brachio vs Chexpert)."""
    print(f"  Running t-SNE on {features.shape[0]} samples...")
    perplexity = min(30, max(5, len(features) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    emb = tsne.fit_transform(features)

    domain_colors = {'NIH': '#2E86AB', 'Brachio': '#A23B72', 'Chexpert': '#F18F01', 'Unknown': '#95A5A6'}
    domain_markers = {'NIH': 'o', 'Brachio': 's', 'Chexpert': '^', 'Unknown': 'x'}

    fig, ax = plt.subplots(figsize=(10, 8))

    for domain in np.unique(domains):
        mask = np.array(domains) == domain
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=domain_colors.get(domain, '#95A5A6'),
            marker=domain_markers.get(domain, 'o'),
            label=domain,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=12, wrap=True)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for SupCon domain alignment')
    parser.add_argument('--data_path', required=True, help='Path to MedicalCXR data')
    parser.add_argument('--checkpoint_baseline', default='', help='Checkpoint trained WITHOUT SupCon (Plot A)')
    parser.add_argument('--checkpoint_supcon', default='', help='Checkpoint trained WITH SupCon (Plot B)')
    parser.add_argument('--output_dir', default='./tsne_plots', help='Where to save plots')
    parser.add_argument('--max_samples_per_class', type=int, default=150, help='Max samples per class for t-SNE')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default='vit_base_patch16_224_in21k')
    parser.add_argument('--use_lora', action='store_true', help='Model used LoRA')
    parser.add_argument('--use_adapters', action='store_true', help='Model used adapters')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--drop', type=float, default=0.0)
    args = parser.parse_args()

    args.nb_classes = 9
    args.adapt_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    args.use_supcon = False
    if not args.use_adapters and not args.use_lora:
        args.use_lora = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Building cross-domain dataloader (NIH vs CheXpert vs Brachio)...")
    dataloader = build_cross_domain_tsne_dataloader(
        args.data_path,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    if args.checkpoint_baseline:
        print("\n--- Plot A: Baseline (without SupCon) ---")
        args_plot = type('Args', (), dict(vars(args)))()
        args_plot.use_supcon = False
        model = create_model_and_load(args_plot, args.checkpoint_baseline, device)
        features, labels, domains = extract_features(model, dataloader, device, use_projection_head=False)
        plot_tsne(
            features, labels, domains,
            title='Baseline (ICON): Domain Gap - Same disease from NIH vs CheXpert form separate clusters',
            output_path=os.path.join(args.output_dir, 'tsne_baseline.png'),
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.checkpoint_supcon:
        print("\n--- Plot B: With SupCon ---")
        args_plot = type('Args', (), dict(vars(args)))()
        args_plot.use_supcon = True
        model = create_model_and_load(args_plot, args.checkpoint_supcon, device)
        use_proj = hasattr(model, 'projection_head') and model.projection_head is not None
        features, labels, domains = extract_features(model, dataloader, device, use_projection_head=use_proj)
        plot_tsne(
            features, labels, domains,
            title='Proposed (SupCon): Domain alignment - Same disease from different domains merges',
            output_path=os.path.join(args.output_dir, 'tsne_supcon.png'),
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not args.checkpoint_baseline and not args.checkpoint_supcon:
        print("Error: Provide at least --checkpoint_baseline or --checkpoint_supcon")
        return

    print(f"\nDone! Plots saved to {args.output_dir}")
    print("\nInterpretation:")
    print("  - Plot A (Baseline): NIH and CheXpert samples for the same disease (e.g., Pneumothorax) form separate clusters = Domain Gap")
    print("  - Plot B (SupCon): Those clusters should merge into one = SupCon aligns features across domains")


if __name__ == '__main__':
    main()
