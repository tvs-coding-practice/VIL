import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import Engine
import models
import utils
import os

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def set_data_config(args):
    if args.dataset == "iDigits":
        args.class_num = 10
        args.domain_num = 4
    elif args.dataset == "DomainNet":
        args.class_num = 345
        args.domain_num = 6
    elif args.dataset == "CORe50":
        args.class_num = 50
        args.domain_num = 8
    elif args.dataset == "MedicalCXR":
        args.class_num = 9  # Total classes (0 to 8)
        args.domain_num = 3  # NIH, Brachio, Chexpert
    return args

def main(args):
    # utils.init_distributed_mode(args)
    args.distributed = False
    args = set_data_config(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
   
    # -------------------------------------------------------------------------
    # Model Creation
    # -------------------------------------------------------------------------
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        adapt_blocks=args.adapt_blocks,
        # Pass LoRA arguments to the model init
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    model.to(device)
    
    # Enable gradient checkpointing to save memory (trades compute for memory)
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(True)
        print("Gradient checkpointing enabled to save memory")
    
    # Initialize projection head for SupCon if enabled
    if args.use_supcon:
        if hasattr(model, 'init_projection_head'):
            model.init_projection_head(projection_dim=args.projection_dim)
            print(f"Initialized SupCon projection head with dimension {args.projection_dim}")
        else:
            print("Warning: Model does not support projection head. SupCon will be disabled.")
            args.use_supcon = False

    # Initialize Engine
    engine = Engine(model=model, device=device, class_mask=class_mask, domain_list=domain_list, args=args)
    
    # -------------------------------------------------------------------------
    # Ensure head is properly initialized (especially after loading pretrained weights)
    # -------------------------------------------------------------------------
    if hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
        # Check if head bias is all zeros (proper initialization) or has problematic values
        head_bias_mean = model.head.bias.abs().mean().item()
        head_weight_mean = model.head.weight.abs().mean().item()
        
        # If head seems improperly initialized (bias too large or weights too small), re-initialize
        if head_bias_mean > 0.01 or head_weight_mean < 0.001:
            print(f"Re-initializing head: bias_mean={head_bias_mean:.6f}, weight_mean={head_weight_mean:.6f}")
            # Re-initialize head with proper values
            torch.nn.init.trunc_normal_(model.head.weight, std=0.02)
            torch.nn.init.zeros_(model.head.bias)
            print(f"Head re-initialized: new bias_mean={model.head.bias.abs().mean().item():.6f}, "
                  f"new weight_mean={model.head.weight.abs().mean().item():.6f}")
    
    # -------------------------------------------------------------------------
    # Training Strategy: LoRA vs Adapters
    # -------------------------------------------------------------------------
    if args.use_lora:
        # --- Strategy A: LoRA Training ---
        print(f"LoRA Enabled (Rank={args.lora_rank}, Alpha={args.lora_alpha}).")
        print("Freezing backbone. Unfreezing LoRA parameters, HEAD, and LAYER NORMS.")
        
        # 1. Freeze EVERYTHING first
        for p in model.parameters():
            p.requires_grad = False
            
        # 2. Unfreeze specific parts
        trainable_params = []
        for n, p in model.named_parameters():
            if 'lora_' in n: # Matches lora_A and lora_B
                p.requires_grad = True
                trainable_params.append(n)
            elif 'head' in n: # Always train the classifier head
                if 'bias' in n:
                    # Freeze head bias to prevent it from learning class bias
                    # The model should learn from features, not bias
                    p.requires_grad = False
                    print(f"Freezing head bias: {n}")
                else:
                    p.requires_grad = True
                    trainable_params.append(n)
            elif 'projection_head' in n: # Train projection head for SupCon
                p.requires_grad = True
                trainable_params.append(n)
            elif 'norm' in n: # <--- CRITICAL FIX: Train LayerNorms
                p.requires_grad = True
                trainable_params.append(n)
        
        print(f"Total trainable tensors: {len(trainable_params)}")

    else:
        # --- Strategy B: Original Adapter / Partial Fine-tuning ---
        print("Standard Training: Using Adapters/Partial Freezing")
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'adapter' in n:
                p.requires_grad = True
            if 'head' in n:
                p.requires_grad = True

    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)
    
    # Verify head is trainable
    if hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
        head_trainable = any(p.requires_grad for p in model.head.parameters())
        print(f'Head is trainable: {head_trainable}')
        if head_trainable:
            head_params = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            print(f'Head parameters: {head_params}')

    # Create Optimizer and Scheduler
    optimizer = create_optimizer(args, model)
    
    # Verify head is in optimizer
    if hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
        head_in_optimizer = False
        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) == id(model.head.weight) or id(p) == id(model.head.bias):
                    head_in_optimizer = True
                    break
            if head_in_optimizer:
                break
        print(f'Head parameters in optimizer: {head_in_optimizer}')
        
        # Separate head weights (not bias) into its own param group with lower LR
        # Head bias is frozen, so we only need to handle weights
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        if head_params:
            # Remove head from existing param groups and add to new one with lower LR
            base_lr = optimizer.param_groups[0]['lr']
            head_lr = base_lr * 0.1  # 10x smaller LR for head
            
            # Remove head params from existing groups
            for group in optimizer.param_groups:
                group['params'] = [p for p in group['params'] 
                                  if id(p) != id(model.head.weight) and id(p) != id(model.head.bias)]
            
            # Add head as separate param group with lower LR
            optimizer.add_param_group({
                'params': head_params, 
                'lr': head_lr,
                'weight_decay': optimizer.param_groups[0].get('weight_decay', 0.0)
            })
            print(f'Head parameters in separate param group with LR={head_lr:.6f} (10x smaller than base LR={base_lr:.6f})')

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None
            
    print(args)
    
    # -------------------------------------------------------------------------
    # Evaluation Loop
    # -------------------------------------------------------------------------
    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            
            _ = engine.evaluate_till_now(model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        return
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    engine.train_and_evaluate(model, criterion, data_loader, optimizer, 
                              lr_scheduler, device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAE')

    parser.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.0028125, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data_path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='iDigits', type=str, help='dataset name')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--train_mask', default=True, type=bool, help='if using the class mask at training')
    parser.add_argument('--task_inc', action='store_true', default=False, help='if doing task incremental')
    parser.add_argument('--domain_inc', action='store_true', default=False, help='if doing domain incremental')
    parser.add_argument('--versatile_inc', action='store_true', default=False, help='if doing versatile incremental')
    parser.add_argument('--joint_train', default=False, help='if doing joint training')

    # Prompt / Adapter parameters
    parser.add_argument('--adapt_blocks', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--num_freeze_epochs', type=int,default=3)
    parser.add_argument('--eval_only_emas', default=False)

    # Misc parameters
    parser.add_argument('--print_freq', type=int, default=10, help = 'The frequency of printing')
    parser.add_argument('--develop', action='store_true', default=False)
    
     #! IC
    parser.add_argument('--IC', action='store_true', default=False, help='if using incremental classifier')
    parser.add_argument('--d_threshold', action='store_true', default=False, help='if using dynamic thresholding in IC')
    parser.add_argument('--gamma',default=10.0, type=float, help='coefficient in dynamic thresholding')
    parser.add_argument('--thre',default=0, type=float, help='value of static threshold if not using dynamic thresholding')
    parser.add_argument('--alpha',default=1.0, type=float, help='coefficient of knowledge distillation in IC loss')

    #! CAST
    parser.add_argument('--beta',default=0.001, type=float, help='coefficient of cast loss')
    parser.add_argument('--k', default=2, type=int, help='the number of clusters in shift pool')
    parser.add_argument('--use_cast_loss', action='store_true', default=False, help='if using CAST loss')
    parser.add_argument('--norm_cast', action='store_true', default=False, help='if using normalization in cast')
    
    #! SupCon (Supervised Contrastive Loss)
    parser.add_argument('--use_supcon', action='store_true', default=False, help='if using Supervised Contrastive Loss')
    parser.add_argument('--supcon_weight', default=0.5, type=float, help='coefficient of SupCon loss (default: 0.5)')
    parser.add_argument('--supcon_temperature', default=0.07, type=float, help='temperature parameter for SupCon (default: 0.07)')
    parser.add_argument('--projection_dim', default=128, type=int, help='dimension of projection head for SupCon (default: 128)')
    
    # -------------------------------------------------------------------------
    # LoRA PARAMETERS (LoRA is now the default method)
    # -------------------------------------------------------------------------
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA instead of Adapters')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank of LoRA matrices (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Scaling factor for LoRA (default: 16)')
    parser.add_argument('--use_adapters', action='store_true', default=False, help='Use legacy Adapters instead of LoRA (deprecated)')

    args = parser.parse_args()
    
    # LoRA is enabled by default unless --use_adapters is explicitly set
    if not args.use_adapters and not args.use_lora:
        args.use_lora = True  # Default to LoRA
    elif args.use_adapters:
        args.use_lora = False  # Explicitly disable LoRA if adapters are requested

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    main(args)

    sys.exit(0)