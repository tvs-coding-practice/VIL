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
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import Engine
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

    # 2. Reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 3. Create Data Loaders
    print(f"Loading data for {args.dataset}...")
    data_loader, class_mask = build_continual_dataloader(args)

    # 4. Create Model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.class_num,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    model.to(device)

    # 5. Create Engine & Optimizer
    engine = Engine(args, model)
    optimizer = create_optimizer(args, model)
    criterion = torch.nn.CrossEntropyLoss()

    # 6. EMA Model Setup
    ema_model = None
    if args.model_ema:
        # Assuming ManualEMA is in engine.py as per your uploads
        from engine import ManualEMA
        ema_model = ManualEMA(model, args.model_ema_decay, device)

    # -------------------------------------------------------------------------
    # RESUME & CHECKPOINT LOGIC
    # -------------------------------------------------------------------------
    start_task = 0
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    # Directory to save checkpoints
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Search for the latest existing checkpoint (Iterate backwards)
    print("Checking for existing checkpoints...")
    for task_id in range(args.num_tasks - 1, -1, -1):
        ckpt_path = os.path.join(ckpt_dir, f'task{task_id}_checkpoint.pth')

        if os.path.exists(ckpt_path):
            print(f"\n>>> Found checkpoint for Task {task_id}. Resuming from Task {task_id + 1}...")
            checkpoint = torch.load(ckpt_path, map_location=device)

            # A. Load Model & Optimizer
            model.load_state_dict(checkpoint['model'])
            # Note: We usually re-create the optimizer per task, but loading state is okay if supported
            # optimizer.load_state_dict(checkpoint['optimizer'])

            if ema_model is not None and 'ema_model' in checkpoint:
                # Check if stored state is the inner model or wrapper
                # If ManualEMA wrapper has .ema_model, load into that
                if hasattr(ema_model, 'ema_model'):
                    ema_model.ema_model.load_state_dict(checkpoint['ema_model'])
                else:
                    ema_model.load_state_dict(checkpoint['ema_model'])

            # B. Restore Experiment Metrics (Acc Matrix)
            if 'acc_matrix' in checkpoint:
                acc_matrix = checkpoint['acc_matrix']

            # C. Restore Engine State (Critical for Distillation/History)
            if 'engine_state' in checkpoint:
                es = checkpoint['engine_state']
                # Restore critical engine variables so it "remembers" previous domains/classes
                if hasattr(engine, 'current_task'): engine.current_task = es.get('current_task', task_id)
                if hasattr(engine, 'class_group_train_count'): engine.class_group_train_count = es.get(
                    'class_group_train_count', [])
                if hasattr(engine, 'class_group_list'): engine.class_group_list = es.get('class_group_list', [])
                if hasattr(engine, 'acc_per_label'): engine.acc_per_label = es.get('acc_per_label', None)
                # Restore specialized sets if they exist
                if hasattr(engine, 'added_classes_in_cur_task'): engine.added_classes_in_cur_task = es.get(
                    'added_classes_in_cur_task', set())
            else:
                # Fallback for old checkpoints: manually sync engine task counter
                engine.current_task = task_id

            # D. Print Past Results (So logs look complete)
            print(f"\n=======================================================")
            print(f"   Restored History (Tasks 0 to {task_id})")
            print(f"=======================================================")
            for t in range(task_id + 1):
                acc = acc_matrix[t, t]
                print(f"Task {t} : Best Accuracy = {acc:.2f}%")
            print(f"=======================================================\n")

            start_task = task_id + 1
            break

    if start_task >= args.num_tasks:
        print("All tasks completed! Exiting.")
        return

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    start_time = time.time()

    for task_id in range(start_task, args.num_tasks):
        print(f"\n[Main] Starting Task {task_id} (Classes: {class_mask[task_id]})...")

        # Run Training for this Task
        model, optimizer = engine.train_and_evaluate(
            model, criterion, data_loader, optimizer, device,
            task_id, class_mask, acc_matrix, ema_model, args
        )

        # Save Checkpoint immediately after task finishes
        if args.output_dir:
            print(f"[Main] Saving checkpoint for Task {task_id}...")

            # Capture Engine State for safer resuming
            engine_state = {
                'current_task': engine.current_task,
                'class_group_train_count': getattr(engine, 'class_group_train_count', []),
                'class_group_list': getattr(engine, 'class_group_list', []),
                'acc_per_label': getattr(engine, 'acc_per_label', None),
                'added_classes_in_cur_task': getattr(engine, 'added_classes_in_cur_task', set()),
            }

            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs,
                'args': args,
                'acc_matrix': acc_matrix,
                'class_mask': class_mask,
                'engine_state': engine_state
            }

            if ema_model is not None:
                # Save the underlying model inside ManualEMA
                if hasattr(ema_model, 'ema_model'):
                    save_dict['ema_model'] = ema_model.ema_model.state_dict()
                else:
                    save_dict['ema_model'] = ema_model.state_dict()

            torch.save(save_dict, os.path.join(ckpt_dir, f'task{task_id}_checkpoint.pth'))

            # Also save the matrix separately for quick plotting
            np.save(os.path.join(args.output_dir, 'acc_matrix.npy'), acc_matrix)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

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