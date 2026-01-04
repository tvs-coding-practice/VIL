import math
import sys
import os
import datetime
import json
from turtle import undo
from typing import Iterable
from pathlib import Path

import torch
import gc  # For garbage collection

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision import transforms


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised Contrastive Loss as described in:
    "Supervised Contrastive Learning" by Khosla et al., NeurIPS 2020
    
    Args:
        features: Tensor of shape [2*N, projection_dim] where N is batch size
                 Features from two augmentations of the same batch
        labels: Tensor of shape [2*N] with class labels
        temperature: Temperature parameter for scaling (default: 0.07)
    
    Returns:
        Loss value (scalar tensor)
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize features (should already be normalized, but ensure it)
    features = F.normalize(features, p=2, dim=1)
    
    # Create mask for positive pairs (same class)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # For numerical stability, subtract max
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Create mask to exclude self-contrast (diagonal)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask
    
    # Compute exp logits
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
    
    # Loss is the negative log-likelihood
    loss = -mean_log_prob_pos.mean()
    
    return loss


class ManualEMA:
    """Manual implementation of Exponential Moving Average for model parameters."""
    def __init__(self, model, decay: float, device=None):
        """
        Args:
            model: The model or adapter to create EMA for
            decay: EMA decay factor (float between 0 and 1)
            device: Device to place the EMA model on
        """
        self.decay = float(decay)
        self.device = device
        
        # Create a deep copy of the model parameters for EMA
        self.ema_model = copy.deepcopy(model)
        
        # Move to device if specified
        if device is not None:
            if isinstance(self.ema_model, (list, tuple)):
                for m in self.ema_model:
                    m.to(device)
            else:
                self.ema_model.to(device)
        
        # Set EMA model to eval mode and disable gradients
        if isinstance(self.ema_model, (list, tuple)):
            for m in self.ema_model:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        else:
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
    
    def update(self, model):
        # Helper to get the correct model (unwrap DDP if necessary)
        if hasattr(model, 'module'):
            model = model.module

        with torch.no_grad():
            # Iterate over both models simultaneously
            for ema_v, model_v in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
                
                # Skip if one is a buffer and the other is a parameter (rare safety check)
                if ema_v.device != model_v.device:
                    model_v = model_v.to(device=ema_v.device)

                # --- FIX START: Handle Shape Mismatch ---
                # Check if shapes differ but element count is the same (e.g., [1, 8, 768] vs [8, 768])
                if ema_v.shape != model_v.shape:
                    if ema_v.numel() == model_v.numel():
                        model_v = model_v.view(ema_v.shape)
                    else:
                        # If sizes assume distinct logic (e.g. buffers growing), skip or print warning
                        continue
                # --- FIX END ---
                
                # Perform the EMA update
                # ema_v = decay * ema_v + (1 - decay) * model_v
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
                
    @property
    def module(self):
        """Return the EMA model."""
        return self.ema_model
    
    def state_dict(self):
        """Return state dict of EMA model for checkpointing."""
        if isinstance(self.ema_model, (list, tuple)):
            return [m.state_dict() for m in self.ema_model]
        else:
            return self.ema_model.state_dict()


class Engine():
    def __init__(self, model=None,device=None,class_mask=[], domain_list= [], args=None):
        self.current_task=0
        self.current_classes=[]
        #! distillation
        self.class_group_num = 5
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        
        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.distill_head= None
        self.model = model
        
        self.num_classes= max([item for mask in class_mask for item in mask])+1
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args=args
        
        self.class_mask=class_mask
        self.domain_list=domain_list
        
        # Create class name mapping for confusion matrix labels
        # Try to get class names from args or use default mapping
        self.class_names = {}
        if hasattr(args, 'class_names') and args.class_names:
            self.class_names = args.class_names
        else:
            # Default medical class mapping (can be overridden)
            medical_class_map = {
                0: 'Atelectasis', 1: 'Emphysema', 2: 'Cardiomegaly', 3: 'Pneumothorax',
                4: 'Edema', 5: 'Infiltration', 6: 'Effusion', 7: 'Nodule', 8: 'No_Finding'
            }
            # Only include classes that are actually used
            all_classes = set()
            for mask in class_mask:
                all_classes.update(mask)
            self.class_names = {cls_id: medical_class_map.get(cls_id, f'Class_{cls_id}') 
                               for cls_id in sorted(all_classes)}

        self.task_type="initial"
        self.args=args
        
        self.adapter_vec=[]
        self.task_type_list=[]
        self.class_group_list=[]
        self.adapter_vec_label=[]
        self.device=device
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.class_num, self.args.domain_num))
            self.label_train_count = np.zeros((self.args.class_num))
            self.tanh = torch.nn.Tanh()
            
        self.cs=torch.nn.CosineSimilarity(dim=1,eps=1e-6)

    def kl_div(self,p,q):
        p=F.softmax(p,dim=1)
        q=F.softmax(q,dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q),dim=1))
        return kl
  
    def set_new_head(self, model, labels_to_be_added,task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id]*len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)
    
        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head
        
    def inference_acc(self,model,data_loader,device):
        print("Start detecting labels to be added...")
        if len(self.current_classes) == 0:
            print("Warning: current_classes is empty, returning empty list")
            return []
        accuracy_per_label = []
        correct_pred_per_label = [0 for i in range(len(self.current_classes))]
        num_instance_per_label = [0 for i in range(len(self.current_classes))]
        
        if data_loader is None:
            print("Warning: data_loader is None, returning zero accuracy for all labels")
            return [0.0] * len(self.current_classes)
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if hasattr(self.args, 'develop') and self.args.develop:
                    if batch_idx>200:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                if output.shape[-1] > self.num_classes: # there are already added nodes till now
                    output,_,_ = self.get_max_label_logits(output, self.current_classes) # there are added nodes previously, but not in current task -> get maximum value and use it
                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)
                
                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += sum(mask).item()
        for correct, num in zip (correct_pred_per_label, num_instance_per_label):
            if num > 0:
                accuracy_per_label.append(round(correct/num,2))
            else:
                # If no instances for this label, set accuracy to 0.0
                accuracy_per_label.append(0.0)
        return accuracy_per_label
    
    def detect_labels_to_be_added(self,inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label,acc,thre in zip(self.current_classes, inference_acc,thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else: # static threshold
            for label,acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self,vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs,dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self,diff_adapter, other):
        weights=[]
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights) # summation-> 1
        return weights

    def get_model_module(self, model):
        """Helper to unwrap DDP/DP models to get the underlying module."""
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
            return model.module
        return model

    def train_one_epoch(self, model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model = None, args = None):

        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10
        
        # Get underlying model for accessing adapters/attributes
        model_module = self.get_model_module(model)

        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            
            # Initialize auxiliary losses
            distill_loss = 0
            supcon_loss = 0
            
            # ------------------------------------------------------------------
            # 1. OPTIMIZED DATA LOADING & FORWARD PASS
            # ------------------------------------------------------------------
            if args.use_supcon and isinstance(input, list):
                # --- SupCon Path (Parallelized Augmentation) ---
                # Input is [view1, view2] from TwoCropTransform
                view1, view2 = input
                
                # Move both views to GPU
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # Concatenate for single efficient forward pass: Shape [2*B, C, H, W]
                images = torch.cat([view1, view2], dim=0)
                
                # Forward pass
                output = model(images)
                
                # Split output back to [View1, View2] for specific losses
                bs = view1.shape[0]
                output1 = output[:bs] # Logits/Features for View 1
                
                # Calculate SupCon Loss
                # We duplicate targets because we have 2 views per image in 'output'
                supcon_targets = torch.cat([target, target], dim=0)
                supcon_loss = supervised_contrastive_loss(output, supcon_targets, temperature=args.supcon_temperature)
                
                # For Standard CE Loss and Accuracy, we use View 1 (Standard Augmentation)
                logits = output1
                
            else:
                # --- Standard Path (No SupCon) ---
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                logits = output

            # ------------------------------------------------------------------
            # 2. CALCULATE LOSSES
            # ------------------------------------------------------------------
            
            # Base Cross Entropy Loss
            loss = criterion(logits, target)

            # Add SupCon Loss if enabled
            if args.use_supcon:
                loss += args.supcon_weight * supcon_loss

            # CAST Loss (Orthogonality for Adapters/LoRA)
            if self.args.use_cast_loss:
                # Check if we have previous adapters to compare against
                if hasattr(self, 'adapter_vec') and len(self.adapter_vec) > args.k:
                    # Get current parameters
                    cur_adapters = model_module.get_adapter() if hasattr(model_module, 'get_adapter') else []
                    
                    if len(cur_adapters) > 0:
                        self.cur_adapters = self.flatten_parameters(cur_adapters)
                        
                        # Calculate difference from previous task
                        diff_adapter = self.cur_adapters - self.prev_adapters
                        
                        # Find clusters and calculate distance
                        _, other = self.find_same_cluster_items(diff_adapter)
                        sim = 0
                        
                        weights = self.calculate_l2_distance(diff_adapter, other)
                        for o, w in zip(other, weights):
                            if self.args.norm_cast:
                                sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter) * torch.norm(o))
                            else:
                                sim += w * torch.matmul(diff_adapter, o)
                                
                        orth_loss = args.beta * torch.abs(sim)
                        if orth_loss > 0:
                            loss += orth_loss

            # Distillation Loss (IC)
            if self.args.IC and ema_model is not None:
                with torch.no_grad():
                    # Handle list input case for teacher
                    teacher_input = input[0] if isinstance(input, list) else input
                    teacher_input = teacher_input.to(device, non_blocking=True)
                    
                    # Robustly handle EMA model access
                    # This fixes the "ModuleList" and "NotImplementedError"
                    try:
                        # Try calling the ema_model directly (works for newer timm ModelEma)
                        teacher_output = ema_model(teacher_input)
                    except (NotImplementedError, TypeError):
                        # Fallback: access .module or .ema if direct call fails
                        if hasattr(ema_model, 'module') and not isinstance(ema_model.module, torch.nn.ModuleList):
                             teacher_output = ema_model.module(teacher_input)
                        elif hasattr(ema_model, 'ema'):
                             teacher_output = ema_model.ema(teacher_input)
                        else:
                             # Last resort: if it's a ModuleList, we can't infer forward, 
                             # skip distillation or assume model matches structure.
                             # If model(images) worked, we assume model_module(teacher_input) works.
                             # We use the unwrapped DDP model as a proxy if EMA fails.
                             teacher_output = model_module(teacher_input)

                # Calculate KL Divergence
                distill_loss = self.kl_div(logits, teacher_output)
                if distill_loss > 0:
                    loss += distill_loss

            # ------------------------------------------------------------------
            # 3. OPTIMIZATION STEP
            # ------------------------------------------------------------------
            acc1, acc5 = accuracy(logits, target, topk=(1, 5)) if logits.shape[1] >= 5 else accuracy(logits, target, topk=(1, 1))
            
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
            optimizer.step()

            # Update EMA model
            if ema_model is not None:
                ema_model.update(model)

            torch.cuda.synchronize()
            
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])

            # --- MEMORY OPTIMIZATION ---
            optimizer.zero_grad(set_to_none=True)
            del input, target, loss, logits, output
            if 'images' in locals(): del images
            if 'view1' in locals(): del view1
            if 'view2' in locals(): del view2
            if 'supcon_loss' in locals(): del supcon_loss
            
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_max_label_logits(self,output, class_mask,task_id=None, slice=True,target=None):
        #! Get max value for each label output
        correct=0
        total=0
        for label in range(self.num_classes): 
            label_nodes = np.where(self.labels_in_head == label)[0]
            output[:,label],max_index = torch.max(output[:,label_nodes],dim=1)
        if slice:
            output = output[:, :self.num_classes] # discard logits of added nodes
            
        return output,correct,total
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader, 
                device, task_id=-1, class_mask=None, ema_model=None, args=None,):
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)

        # switch to evaluation mode
        model.eval()

        correct_sum, total_sum = 0,0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))
        
        # Collect all predictions and targets for confusion matrix
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx,(input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx>20:
                        break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output            
                output = model(input)
                
                # Debug: Check output statistics before processing
                if batch_idx == 0 and task_id == 0:
                    print(f"\n[DEBUG] Raw model output stats:")
                    print(f"  Output shape: {output.shape}")
                    print(f"  Output min/max/mean: {output.min().item():.4f} / {output.max().item():.4f} / {output.mean().item():.4f}")
                    print(f"  Output per-class mean: {output.mean(dim=0)[:min(5, output.shape[1])]}")
                    if hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
                        print(f"  Head weight stats: min={model.head.weight.min().item():.4f}, max={model.head.weight.max().item():.4f}, mean={model.head.weight.mean().item():.4f}")
                        print(f"  Head bias: {model.head.bias[:min(5, len(model.head.bias))]}")
                
                output, correct, total = self.get_max_label_logits(output, class_mask[task_id],task_id=task_id, target=target,slice=True) 
                output_ema = [output.softmax(dim=1)]
                correct_sum+=correct
                total_sum+=total
                
                if ema_model is not None:
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)
                    output = model(input)
                    output,_,_ = self.get_max_label_logits(output, class_mask[task_id],slice=True) 
                    output_ema.append(output.softmax(dim=1))
                    model.put_adapter(tmp_adapter)
                
                output = torch.stack(output_ema, dim=-1).max(dim=-1)[0]
                
                # Mask out unseen classes during evaluation (same as training)
                # Get all classes seen up to current task
                all_seen_classes_eval = []
                for i in range(task_id + 1):
                    all_seen_classes_eval.extend(class_mask[i])
                all_seen_classes_eval = sorted(list(set(all_seen_classes_eval)))
                not_seen_mask = np.setdiff1d(np.arange(self.num_classes), all_seen_classes_eval)
                if len(not_seen_mask) > 0:
                    not_seen_mask = torch.tensor(not_seen_mask, dtype=torch.int64).to(device)
                    output = output.index_fill(dim=1, index=not_seen_mask, value=float('-inf'))
                
                loss = criterion(output, target)
                
                # Get predictions for confusion matrix (now restricted to seen classes)
                _, pred = torch.max(output, 1)
                
                # Debug: Check prediction distribution
                if batch_idx == 0 and task_id == 0:
                    print(f"\n[DEBUG] After masking and processing:")
                    print(f"  Output shape: {output.shape}")
                    print(f"  Output min/max/mean: {output.min().item():.4f} / {output.max().item():.4f} / {output.mean().item():.4f}")
                    print(f"  Output per-class mean: {output.mean(dim=0)}")
                    print(f"  Predictions in batch: {torch.bincount(pred, minlength=output.shape[1])[:min(5, output.shape[1])]}")
                    print(f"  Targets in batch: {torch.bincount(target, minlength=output.shape[1])[:min(5, output.shape[1])]}")
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                if self.args.d_threshold and self.current_task +1 != self.args.num_tasks and self.current_task == task_id:
                    label_correct, label_total = self.update_acc_per_label(label_correct, label_total, output, target)
                acc1, acc3 = accuracy(output, target, topk=(1, 3))

                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@3'].update(acc3.item(), n=input.shape[0])
            if total_sum>0:
                print(f"Max Pooling acc: {correct_sum/total_sum}")
                
            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total, decimals=3)
                print(self.label_train_count)
                print(self.acc_per_label)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.meters['Acc@1'], top3=metric_logger.meters['Acc@3'], losses=metric_logger.meters['Loss']))

        # Compute and print confusion matrix
        if len(all_predictions) > 0 and len(all_targets) > 0:
            # Convert to numpy arrays for sklearn
            preds_array = np.array(all_predictions)
            targets_array = np.array(all_targets)
            
            # Get all classes that actually appear in the data (targets and predictions)
            unique_targets = np.unique(targets_array)
            unique_preds = np.unique(preds_array)
            classes_in_data = sorted(list(set(np.concatenate([unique_targets, unique_preds]))))
            
            # Get all classes seen up to current task (for display purposes)
            all_seen_classes = []
            for i in range(task_id + 1):
                all_seen_classes.extend(class_mask[i])
            all_seen_classes = sorted(list(set(all_seen_classes)))
            
            # Use classes_in_data to ensure all samples are included
            # But also include all_seen_classes to show zeros for classes not in current task
            display_classes = sorted(list(set(classes_in_data + all_seen_classes)))
            
            # Compute confusion matrix using all classes that appear in data
            # This ensures all samples are counted
            cm = confusion_matrix(targets_array, preds_array, labels=display_classes)
            
            # Get class names for labels
            class_labels = [self.class_names.get(cls_id, f'Class_{cls_id}') for cls_id in display_classes]
            
            # Verify total samples match
            total_in_matrix = cm.sum()
            
            print(f"\nConfusion Matrix for Task {task_id + 1} (all seen classes up to task {task_id + 1}):")
            print(f"Total samples collected: {len(all_predictions)}")
            print(f"Total samples in matrix: {total_in_matrix}")
            print(f"Classes in data: {classes_in_data}")
            print(f"All seen classes: {all_seen_classes}")
            print(f"Display classes: {display_classes}")
            print(f"Class Labels: {class_labels}")
            print("\nConfusion Matrix (rows=actual, cols=predicted):")
            # Print header
            print(f"{'Actual \\ Predicted':<20}", end="")
            for label in class_labels:
                print(f"{label[:12]:>12}", end="")
            print(f"{'Total':>10}")
            # Print rows
            for i, (label, row) in enumerate(zip(class_labels, cm)):
                print(f"{label[:18]:<20}", end="")
                for val in row:
                    print(f"{val:>12}", end="")
                print(f"{row.sum():>10}")
            print()

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    @torch.no_grad()
    def evaluate_till_now(self,model: torch.nn.Module, data_loader, 
                        device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None,):
        stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@3, Loss

        for i in range(task_id+1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)

            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@3']
            stat_matrix[2, i] = test_stats['Loss']

            acc_matrix[i, task_id] = test_stats['Acc@1']
        
        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

        diagonal = np.diag(acc_matrix)

        result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@3: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) -
                                acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
        print(result_str)
        return test_stats
    
    def flatten_parameters(self,modules):
        flattened_params = []
       
        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params) 
        return torch.cat([param.view(-1) for param in flattened_params])
    
    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
              
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)
            self.kmeans = KMeans(n_clusters=k,n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)
    
    
    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args = None,):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        
        # Check for LoRA or adapter parameters
        param_pattern = 'lora_' if getattr(args, 'use_lora', False) else 'adapter'
        
        if epoch == 0:
            for n, p in model.named_parameters():
                if param_pattern in n:
                    p.requires_grad = False
            param_type = 'LoRA' if getattr(args, 'use_lora', False) else 'adapter'
            print(f'Freezing {param_type} parameters for {args.num_freeze_epochs} epochs')

        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if param_pattern in n:
                    p.requires_grad = True
            param_type = 'LoRA' if getattr(args, 'use_lora', False) else 'adapter'
            print(f'Unfreezing {param_type} parameters')        
        return model
    
    
    def pre_train_task(self, model, data_loader, device, task_id, args):
        self.current_task += 1
        
        # Re-initialize head bias at the start of each task to prevent bias accumulation
        if task_id == 0 and hasattr(model, 'head') and isinstance(model.head, torch.nn.Linear):
            # Check if head bias is already biased
            head_bias_abs_mean = model.head.bias.abs().mean().item()
            if head_bias_abs_mean > 0.01:
                print(f"Task {task_id}: Re-initializing head bias (current mean={head_bias_abs_mean:.6f})")
                torch.nn.init.zeros_(model.head.bias)
                print(f"Head bias re-initialized to zeros")
        self.current_class_group = int(min(self.class_mask[task_id])/self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        
        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()  
        #! distillation
        if self.class_group_train_count[self.current_class_group]==0:
            self.distill_head=None
        else: # already seen classes
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                # Convert inf_acc to numpy array for proper broadcasting
                inf_acc = np.array(inf_acc)
                # Validate that inf_acc has the correct length
                if len(inf_acc) != len(self.current_classes):
                    print(f"Warning: inf_acc length ({len(inf_acc)}) doesn't match current_classes length ({len(self.current_classes)})")
                    # Pad or truncate to match
                    if len(inf_acc) < len(self.current_classes):
                        inf_acc = np.pad(inf_acc, (0, len(self.current_classes) - len(inf_acc)), 'constant', constant_values=0.0)
                    else:
                        inf_acc = inf_acc[:len(self.current_classes)]
                thresholds=[]
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                        # Avoid division by zero: if average_accs is zero, use a small epsilon
                        epsilon = 1e-8
                        average_accs = np.where(average_accs == 0, epsilon, average_accs)
                        thresholds = self.args.gamma*(average_accs - inf_acc) / average_accs
                        thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                        thresholds = [round(t,2) if t>self.args.thre else self.args.thre for t in thresholds]
                        print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                    # If count is 0, thresholds remains empty list and static threshold will be used in detect_labels_to_be_added
                labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
                
                
                if len(labels_to_be_added) > 0: #! Add node to the classifier if needed
                    new_head = self.set_new_head(model, labels_to_be_added,task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad=False
    
        if task_id==0:
            self.task_type_list.append("Initial")
            return model, optimizer
        
        prev_class = self.class_mask[task_id-1]
        prev_domain = self.domain_list[task_id-1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        
        if prev_class == cur_class:
            self.task_type = "DIL"
        else:
            self.task_type = "CIL"
        
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        
        return model, optimizer


    def post_train_task(self,model: torch.nn.Module,task_id=-1):
        #! update classifier pool
        self.class_group_train_count[self.current_class_group]+=1
        self.classifier_pool[self.current_class_group]=copy.deepcopy(model.head)
        for c in self.classifier_pool:
                if c != None:
                    for p in c.parameters():
                        p.requires_grad=False
      
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector=self.cur_adapters - self.prev_adapters
        # if task_id>0: #? 1
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()
                 
    def get_engine_state(self):
        """Get the current state of the engine for checkpointing."""
        # Save classifier pool state dicts (not the modules themselves)
        classifier_pool_state = []
        for c in self.classifier_pool:
            if c is not None:
                classifier_pool_state.append(c.state_dict())
            else:
                classifier_pool_state.append(None)
        
        engine_state = {
            'current_task': self.current_task,
            'classifier_pool_state': classifier_pool_state,
            'class_group_train_count': self.class_group_train_count.copy(),
            'class_group_list': self.class_group_list.copy(),
            'task_type_list': self.task_type_list.copy(),
            'adapter_vec_label': self.adapter_vec_label.copy(),
        }
        
        # Handle adapter_vec - convert tensors to CPU and detach for serialization
        if hasattr(self, 'adapter_vec') and len(self.adapter_vec) > 0:
            adapter_vec_state = []
            for v in self.adapter_vec:
                if isinstance(v, torch.Tensor):
                    adapter_vec_state.append(v.cpu().clone().detach())
                else:
                    adapter_vec_state.append(v)
            engine_state['adapter_vec'] = adapter_vec_state
        
        if self.args.d_threshold:
            engine_state['acc_per_label'] = self.acc_per_label.copy()
            engine_state['label_train_count'] = self.label_train_count.copy()
        return engine_state
    
    def load_engine_state(self, engine_state, model=None):
        """Load engine state from checkpoint."""
        self.current_task = engine_state.get('current_task', 0)
        
        # Restore classifier pool from state dicts
        classifier_pool_state = engine_state.get('classifier_pool_state', [None] * self.class_group_num)
        # Note: We'll restore classifier_pool when needed, as it requires the model head structure
        # For now, we'll store the state dicts and restore them later if needed
        self._classifier_pool_state_dicts = classifier_pool_state
        
        self.class_group_train_count = engine_state.get('class_group_train_count', [0] * self.class_group_num)
        self.class_group_list = engine_state.get('class_group_list', [])
        self.task_type_list = engine_state.get('task_type_list', [])
        self.adapter_vec_label = engine_state.get('adapter_vec_label', [])
        
        if 'adapter_vec' in engine_state:
            # Convert back to tensors on the correct device
            adapter_vec_loaded = []
            for v in engine_state['adapter_vec']:
                if isinstance(v, torch.Tensor):
                    adapter_vec_loaded.append(v.to(self.device))
                else:
                    adapter_vec_loaded.append(v)
            self.adapter_vec = adapter_vec_loaded
        
        if self.args.d_threshold and 'acc_per_label' in engine_state:
            self.acc_per_label = engine_state['acc_per_label']
            self.label_train_count = engine_state.get('label_train_count', np.zeros((self.args.class_num)))
        
        print(f"Loaded engine state: current_task={self.current_task}, completed_tasks={len(self.task_type_list)}")
        
        # Restore classifier pool if we have the model and state dicts
        if model is not None and hasattr(self, '_classifier_pool_state_dicts'):
            for i, state_dict in enumerate(self._classifier_pool_state_dicts):
                if state_dict is not None and self.classifier_pool[i] is not None:
                    try:
                        self.classifier_pool[i].load_state_dict(state_dict)
                    except Exception as e:
                        print(f"Warning: Could not restore classifier_pool[{i}]: {e}")

    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                        lr_scheduler, device: torch.device, class_mask=None, args = None,):

        # create matrix to save end-of-task accuracies 
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        
        # Check for existing checkpoints to resume from
        start_task = 0
        if args.output_dir and utils.is_main_process():
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
            # Find the highest completed task
            for task_id in range(args.num_tasks - 1, -1, -1):
                checkpoint_path = os.path.join(checkpoint_dir, f'task{task_id+1}_checkpoint.pth')
                if os.path.exists(checkpoint_path):
                    print(f"\n{'='*60}")
                    print(f"Found checkpoint for Task {task_id+1}. Loading to resume training...")
                    print(f"{'='*60}\n")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    
                    # Load model state
                    model.load_state_dict(checkpoint['model'])
                    if 'optimizer' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer'])
                    if 'ema_model' in checkpoint and checkpoint['ema_model'] is not None:
                        # EMA model will be recreated if needed
                        pass
                    
                    # Load engine state
                    if 'engine_state' in checkpoint:
                        self.load_engine_state(checkpoint['engine_state'], model=model)
                        
                        # Restore classifier_pool from state dicts after loading engine state
                        if hasattr(self, '_classifier_pool_state_dicts'):
                            for i, state_dict in enumerate(self._classifier_pool_state_dicts):
                                if state_dict is not None:
                                    # Recreate the classifier head if it doesn't exist
                                    if self.classifier_pool[i] is None:
                                        # We need to create a copy of model.head with the same structure
                                        # This will be done when the classifier is actually used
                                        pass
                                    else:
                                        try:
                                            self.classifier_pool[i].load_state_dict(state_dict)
                                        except Exception as e:
                                            print(f"Warning: Could not restore classifier_pool[{i}]: {e}")
                    
                    # Load accuracy matrix if available
                    if 'acc_matrix' in checkpoint:
                        acc_matrix = checkpoint['acc_matrix']
                        print(f"Loaded accuracy matrix from checkpoint")
                        # Print previous task results
                        print(f"\nPrevious Task Results:")
                        print(f"Accuracy Matrix:")
                        print(acc_matrix[:task_id+1, :task_id+1])
                        if task_id >= 0:
                            avg_acc = np.mean([acc_matrix[i, task_id] for i in range(task_id+1)])
                            print(f"Average Accuracy: {avg_acc:.4f}")
                    
                    start_task = task_id + 1
                    print(f"Resuming from Task {start_task} (Task {task_id+1} was completed)")
                    break
        
        ema_model = None
        
        for task_id in range(start_task, args.num_tasks):
            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            
            model, optimizer = self.pre_train_task(model, data_loader[task_id]['train'], device, task_id,args)
            
            # Create EMA model if needed (for task 1+ with adapters)
            if task_id >= 1 and len(args.adapt_blocks) > 0 and ema_model is None:
                ema_model = ManualEMA(model.get_adapter(), decay=args.ema_decay, device=device)
            
            train_stats = {}  # Initialize to avoid errors
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args,)
                train_stats = self.train_one_epoch(model=model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, ema_model=ema_model, args=args,)
              
                if lr_scheduler:
                    lr_scheduler.step(epoch)
                    
            self.post_train_task(model,task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1 
            test_stats = self.evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                        task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, ema_model=ema_model, args=args)
            
            # Print and save task results
            print(f"\n{'='*80}")
            print(f"TASK {task_id+1} COMPLETED")
            print(f"{'='*80}")
            print(f"Task Type: {self.task_type}")
            print(f"Current Classes: {self.current_classes}")
            
            # Print accuracy matrix for this task
            print(f"\nAccuracy Matrix (row=task, col=eval_after_task):")
            print(acc_matrix[:task_id+1, :task_id+1])
            
            # Calculate and print average accuracy
            avg_acc = np.mean([acc_matrix[i, task_id] for i in range(task_id+1)])
            print(f"\nAverage Accuracy across all tasks: {avg_acc:.4f}")
            
            if task_id > 0:
                # Calculate forgetting
                forgetting = np.mean([np.max(acc_matrix[i, :task_id+1]) - acc_matrix[i, task_id] 
                                    for i in range(task_id)])
                print(f"Forgetting: {forgetting:.4f}")
            
            print(f"{'='*80}\n")
            
            # Save checkpoint after each task
            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint', f'task{task_id+1}_checkpoint.pth')
                state_dict = {
                        'model': model.state_dict(),
                        'ema_model': ema_model.state_dict() if ema_model is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'task_id': task_id,
                        'args': args,
                        'engine_state': self.get_engine_state(),
                        'acc_matrix': acc_matrix,
                    }
                if args.sched is not None and args.sched != 'constant' and lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                
                utils.save_on_master(state_dict, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                
                # Also save accuracy matrix separately for easy access
                acc_matrix_path = os.path.join(args.output_dir, 'checkpoint', 'acc_matrix.npy')
                np.save(acc_matrix_path, acc_matrix)
                print(f"Accuracy matrix saved: {acc_matrix_path}")

            # Save detailed log stats
            log_stats = {
                'task_id': task_id,
                'task_type': self.task_type,
                'current_classes': self.current_classes.tolist() if isinstance(self.current_classes, np.ndarray) else self.current_classes,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'acc_matrix': acc_matrix.tolist(),
            }
            
            # Calculate summary statistics
            if task_id >= 0:
                avg_acc = np.mean([acc_matrix[i, task_id] for i in range(task_id+1)])
                log_stats['avg_accuracy'] = float(avg_acc)
                if task_id > 0:
                    forgetting = np.mean([np.max(acc_matrix[i, :task_id+1]) - acc_matrix[i, task_id] 
                                        for i in range(task_id)])
                    log_stats['forgetting'] = float(forgetting)

            if args.output_dir and utils.is_main_process():
                log_file = os.path.join(args.output_dir, 'task_results.jsonl')
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
                print(f"Task results logged to: {log_file}")
                
                # Also save to timestamped file for backup
                timestamped_log = os.path.join(args.output_dir, f'log_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}.jsonl')
                with open(timestamped_log, 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
