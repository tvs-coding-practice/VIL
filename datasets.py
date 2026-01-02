import random
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils

# --- Configuration for Medical VIL ---
MEDICAL_CLASS_MAP = {
    'Atelectasis': 0,
    'Emphysema': 1,
    'Cardiomegaly': 2,
    'Pneumothorax': 3,
    'Edema': 4,
    'Infiltration': 5,
    'Effusion': 6,
    'Nodule': 7,
    'No_Finding': 8
}

# 6-Task Plan: NIH (CIL) -> Brachio (DIL) -> Chexpert (VIL)
MEDICAL_TASKS_CONFIG = [
    ('NIH', ['Atelectasis', 'Emphysema'], False),  # Task 1
    ('NIH', ['Cardiomegaly', 'Pneumothorax'], False),  # Task 2
    ('NIH', ['Edema', 'Infiltration'], True),  # Task 3 (Weighted: Edema < Infiltration)
    ('NIH', ['Effusion', 'Nodule'], False),  # Task 4
    ('Brachio', ['Effusion', 'Infiltration', 'Nodule'], True),  # Task 5 (Weighted: Effusion < others)
    ('Chexpert', ['Cardiomegaly', 'Pneumothorax', 'No_Finding'], False)  # Task 6
]


class TwoCropTransform:
    """Generate two different augmented views of the same image"""
    def __init__(self, transform, strong_transform=None):
        self.transform = transform
        self.strong_transform = strong_transform if strong_transform is not None else transform

    def __call__(self, x):
        # View 1: Standard augmentation
        v1 = self.transform(x)
        # View 2: Strong augmentation (matches your manual logic)
        v2 = self.strong_transform(x)
        return [v1, v2]


class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list()
    domain_list = list()

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    # -----------------------------------------------
    # 1. MEDICAL VIL IMPLEMENTATION
    # -----------------------------------------------
    if args.dataset == 'MedicalCXR':
        # Load the MedicalCXR dataset wrapper (returns list of ImageFolders [NIH, Brachio, Chexpert])
        # Note: We rely on the MedicalCXR class defined in continual_datasets.py
        dataset_train_list = MedicalCXR(args.data_path, train=True, transform=transform_train, mode='vil').data
        dataset_val_list = MedicalCXR(args.data_path, train=False, transform=transform_val, mode='vil').data

        # Build the specific 6-task scenario
        splited_dataset, class_mask, domain_list, samplers = build_medical_vil_scenario(
            dataset_train_list, dataset_val_list, args
        )

        args.nb_classes = len(MEDICAL_CLASS_MAP)  # Total global classes (9)

        # Create DataLoaders
        for i in range(len(splited_dataset)):
            dataset_train, dataset_val = splited_dataset[i]
            current_sampler = samplers[i]

            if current_sampler is not None:
                sampler_train = current_sampler
                shuffle_train = False
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                shuffle_train = False

            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            dataloader.append({'train': data_loader_train, 'val': data_loader_val})

        return dataloader, class_mask, domain_list

    # -----------------------------------------------
    # 2. STANDARD IMPLEMENTATION (Existing Code)
    # -----------------------------------------------
    if args.task_inc:
        mode = 'til'
    elif args.domain_inc:
        mode = 'dil'
    elif args.versatile_inc:
        mode = 'vil'
    elif args.joint_train:
        mode = 'joint'
    else:
        mode = 'cil'

    if mode in ['til', 'cil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])

            splited_dataset = list()
            for i in range(args.num_tasks):
                t = [train[i+args.num_tasks*j] for j in range(len(dataset_list))]
                v = [val[i+args.num_tasks*j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            args.nb_classes = len(splited_dataset[0][1].datasets[0].dataset.classes)
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0]
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = len(dataset_val.classes)

    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset.append((dataset_train, dataset_val))
            
            args.nb_classes = len(dataset_val.classes)
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val.classes)
            else:
                splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val[0].classes)
    
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                train.append(dataset_train)
                val.append(dataset_val)
                args.nb_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]

            class_mask = None
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.nb_classes = len(dataset_val.classes)
            class_mask = None
            
    else:
        raise ValueError(f'Invalid mode: {mode}')
                

    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)

    for i in range(len(splited_dataset)):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask, domain_list

def get_dataset(dataset, transform_train, transform_val, mode, args,):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)
    # MedicalCXR is handled directly in the main function to avoid recursion or type mismatch with .data list,
    # but could be added here if needed.
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val


def build_medical_vil_scenario(dataset_train_list, dataset_val_list, args):
    """
    Constructs the specific 6-task plan using the loaded domain datasets.
    dataset_train_list: [ImageFolder(NIH), ImageFolder(Brachio), ImageFolder(Chexpert)]
    """
    splited_dataset = []
    class_mask = []
    domain_list = []
    samplers = []

    # Map domain names to list indices based on the order defined in MedicalCXR class
    domain_idx_map = {'NIH': 0, 'Brachio': 1, 'Chexpert': 2}

    for task_idx, (domain_key, target_classes, use_weighted) in enumerate(MEDICAL_TASKS_CONFIG):
        # 1. Select the correct domain dataset
        d_idx = domain_idx_map[domain_key]
        full_train = dataset_train_list[d_idx]
        full_val = dataset_val_list[d_idx]

        # 2. Map Target Class Names to Source Dataset Indices
        source_class_to_idx = full_train.class_to_idx

        target_source_indices = []
        global_ids = []

        for cls_name in target_classes:
            if cls_name not in source_class_to_idx:
                raise ValueError(
                    f"Class '{cls_name}' not found in {domain_key}. Available: {list(source_class_to_idx.keys())}")

            target_source_indices.append(source_class_to_idx[cls_name])
            global_ids.append(MEDICAL_CLASS_MAP[cls_name])

        class_mask.append(global_ids)
        domain_list.append(domain_key)

        # 3. Create Subsets & Remap Labels
        # Train
        train_indices = [i for i, (_, label) in enumerate(full_train.imgs) if label in target_source_indices]
        train_subset = Subset(full_train, train_indices)
        train_subset = RemappedSubset(train_subset, source_class_to_idx, MEDICAL_CLASS_MAP)

        # Val
        val_indices = [i for i, (_, label) in enumerate(full_val.imgs) if label in target_source_indices]
        val_subset = Subset(full_val, val_indices)
        val_subset = RemappedSubset(val_subset, source_class_to_idx, MEDICAL_CLASS_MAP)

        splited_dataset.append((train_subset, val_subset))

        # 4. Handle Weighted Sampling
        if use_weighted:
            # We access the original targets via indices to calculate weights
            subset_targets = [full_train.targets[i] for i in train_indices]
            class_counts = {}
            for t in subset_targets:
                class_counts[t] = class_counts.get(t, 0) + 1

            weights = []
            for t in subset_targets:
                weights.append(1.0 / class_counts[t])

            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, len(weights))
            samplers.append(sampler)
        else:
            samplers.append(None)

    return splited_dataset, class_mask, domain_list, samplers


class RemappedSubset(torch.utils.data.Dataset):
    """
    Wraps a Subset to remap targets from Source-Folder-IDs to Global-Experiment-IDs.
    """

    def __init__(self, subset, source_map, global_map):
        self.subset = subset
        self.idx_to_name = {v: k for k, v in source_map.items()}
        self.global_map = global_map

    def __getitem__(self, index):
        x, y_source = self.subset[index]
        class_name = self.idx_to_name[y_source]
        y_global = self.global_map[class_name]
        return x, y_global

    def __len__(self):
        return len(self.subset)


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks):
        train_split_indices = list()
        test_split_indices = list()
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask

def build_vil_scenario(splited_dataset, args):
    datasets = list()
    class_mask = list()
    domain_list = list()

    for i in range(len(splited_dataset)):
        dataset, mask = split_single_dataset(splited_dataset[i][0], splited_dataset[i][1], args)
        datasets.append(dataset)
        class_mask.append(mask)
        for _ in range(len(dataset)):
            domain_list.append(f'D{i}')

    splited_dataset = sum(datasets, [])
    class_mask = sum(class_mask, [])

    args.num_tasks = len(splited_dataset)

    zipped = list(zip(splited_dataset, class_mask, domain_list))
    random.shuffle(zipped)
    splited_dataset, class_mask, domain_list = zip(*zipped)

    return splited_dataset, class_mask, domain_list, args


def build_transform(is_train, args):
    # Base transform (Standard)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # If using SupCon, we need to return two views
    if is_train and args.use_supcon:
        # Define the 'Strong' augmentation you were doing manually in engine.py
        # Note: Gaussian noise is usually done on tensors, but standard transforms are cleaner.
        strong_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Stronger jitter
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)), # Replaces your manual erasing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return TwoCropTransform(transform, strong_transform)

    return transform
