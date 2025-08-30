"""
Knowledge distillation trainer for Nepali student model.
Optimized for M4 Max MacBook Pro with comprehensive monitoring.
"""

import os
import time
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from datasets import Dataset as HFDataset
import wandb
from tqdm import tqdm
import psutil

from ..model.distillation_loss import DistillationLoss
from ..model.tokenizer import NepaliTokenizer

import logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: float
    loss: float
    kd_loss: float
    task_loss: float
    learning_rate: float
    grad_norm: float
    memory_used: float
    tokens_per_second: float


class NepaliDistillationDataset(Dataset):
    """PyTorch dataset for knowledge distillation."""
    
    def __init__(self, hf_dataset: HFDataset, tokenizer: NepaliTokenizer, max_length: int = 1024):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize input and target
        input_text = item.get('input', item.get('text', ''))
        target_text = item.get('target', item.get('text', ''))
        teacher_output = item.get('teacher_output', '')
        
        # Encode texts
        input_encoding = self.tokenizer.encode(input_text, max_length=self.max_length, return_tensors='pt')
        target_encoding = self.tokenizer.encode(target_text, max_length=self.max_length, return_tensors='pt')
        
        # For now, use target encoding as teacher logits placeholder
        # In real implementation, this would be actual teacher model logits
        teacher_encoding = self.tokenizer.encode(teacher_output or target_text, max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'teacher_logits': teacher_encoding['input_ids'].squeeze(0)  # Placeholder
        }


class DistillationTrainer:
    """Knowledge distillation trainer optimized for M4 Max."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_dataset: HFDataset,
        eval_dataset: HFDataset,
        tokenizer: NepaliTokenizer,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        
        # Create PyTorch datasets
        max_length = config['data'].get('max_length', 1024)
        self.train_torch_dataset = NepaliDistillationDataset(train_dataset, tokenizer, max_length)
        self.eval_torch_dataset = NepaliDistillationDataset(eval_dataset, tokenizer, max_length)
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = DistillationLoss(config['distillation'])
        
        # Training state
        self.global_step = 0
        self.best_metric = float('inf')  # Assuming loss as metric (lower is better)
        self.epochs_trained = 0
        self.start_time = time.time()
        
        # Monitoring
        self.setup_logging()
        self.metrics_history = []
        
        # Create dataloaders
        self.train_dataloader = self._create_dataloader(self.train_torch_dataset, is_training=True)
        self.eval_dataloader = self._create_dataloader(self.eval_torch_dataset, is_training=False)
        
        # Setup output directories
        self.output_dir = Path(config['checkpointing']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # M4 Max optimizations
        self._setup_mps_optimizations()
        
        logger.info(f"Initialized trainer with {len(self.train_dataset)} train samples, {len(self.eval_dataset)} eval samples")
    
    def setup_logging(self):
        """Setup experiment tracking and logging."""
        
        # Setup Weights & Biases if configured
        if self.config['training'].get('use_wandb', True):
            try:
                wandb.init(
                    project="nepali-distillation",
                    config=self.config,
                    name=f"nepali-student-{int(time.time())}",
                    tags=["distillation", "nepali", "m4-max"]
                )
                logger.info("✅ Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize W&B: {e}")
        
        # Setup file logging
        if self.config.get('logging', {}).get('log_to_file', True):
            log_file = Path(self.config.get('logging', {}).get('log_file', 'outputs/logs/training.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add file handler to logger
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def _setup_optimizer(self):
        """Setup optimizer with layer-wise learning rate decay."""
        
        # Parameter groups with different learning rates
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config['training'].get('weight_decay', 0.01),
                'lr': self.config['training'].get('learning_rate', 5e-4)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': self.config['training'].get('learning_rate', 5e-4)
            }
        ]
        
        # Use AdamW with optimizations for M4 Max
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            eps=1e-8,
            betas=(0.9, 0.999),
            fused=True if hasattr(torch.optim.AdamW, 'fused') else False  # Use fused optimizer if available
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        
        num_training_steps = len(self.train_dataloader) * self.config['training'].get('num_epochs', 3)
        warmup_steps = self.config['training'].get('warmup_steps', 1000)
        
        scheduler_type = self.config['training'].get('scheduler_type', 'linear')
        
        if scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        
        return scheduler
    
    def _create_dataloader(self, dataset: Dataset, is_training: bool = True) -> DataLoader:
        """Create optimized DataLoader for M4 Max."""
        
        def collate_fn(batch):
            """Custom collate function with padding."""
            
            # Pad sequences to the same length in the batch
            max_length = max(len(item['input_ids']) for item in batch)
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            batch_teacher_logits = []
            
            for item in batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                labels = item['labels']
                teacher_logits = item['teacher_logits']
                
                # Pad sequences
                pad_length = max_length - len(input_ids)
                
                if pad_length > 0:
                    pad_token_id = self.tokenizer.token_to_id.get(self.tokenizer.special_tokens['pad_token'], 0)
                    
                    input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
                    labels = torch.cat([labels, torch.full((pad_length,), -100)])  # -100 is ignored in loss
                    teacher_logits = torch.cat([teacher_logits, torch.full((pad_length,), pad_token_id)])
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
                batch_teacher_logits.append(teacher_logits)
            
            return {
                'input_ids': torch.stack(batch_input_ids),
                'attention_mask': torch.stack(batch_attention_mask),
                'labels': torch.stack(batch_labels),
                'teacher_logits': torch.stack(batch_teacher_logits)
            }
        
        batch_size = (self.config['data'].get('batch_size', 8) if is_training 
                     else self.config['data'].get('eval_batch_size', 16))
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            collate_fn=collate_fn,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=True,
            drop_last=is_training,
            persistent_workers=True if self.config['data'].get('num_workers', 4) > 0 else False
        )
    
    def _setup_mps_optimizations(self):
        """Setup Metal Performance Shaders optimizations."""
        
        if self.device.type == 'mps':
            logger.info("🔧 Applying M4 Max MPS optimizations...")
            
            # Enable MPS optimizations
            torch.backends.mps.enable_sdp_kernel(True)
            
            # Optimize for M4 Max memory bandwidth
            if hasattr(torch.backends.mps, 'set_cache_mode'):
                torch.backends.mps.set_cache_mode('high_bandwidth')
            
            # Enable mixed precision for MPS
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            logger.info("✅ M4 Max optimizations enabled")
        else:
            self.use_amp = False
            self.scaler = None
    
    def train(self):
        """Main training loop with comprehensive monitoring."""
        
        logger.info("🚀 Starting knowledge distillation training...")
        logger.info(f"📊 Model parameters: {self.model.count_parameters():,}")
        logger.info(f"💾 Training on device: {self.device}")
        
        self.model.train()
        self.model.to(self.device)
        
        num_epochs = self.config['training'].get('num_epochs', 3)
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 4)
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        
        total_steps = len(self.train_dataloader) * num_epochs
        
        logger.info(f"🎯 Training plan: {num_epochs} epochs, {len(self.train_dataloader)} steps/epoch, {total_steps} total steps")
        
        for epoch in range(num_epochs):
            logger.info(f"📚 Starting Epoch {epoch + 1}/{num_epochs}")
            
            epoch_metrics = self._train_epoch(
                epoch=epoch,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm
            )
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            
            # Combine metrics
            combined_metrics = {**epoch_metrics, **eval_metrics}
            self.log_metrics(combined_metrics, step_type='epoch')
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1)
            
            # Check if best model
            eval_loss = eval_metrics.get('eval_loss', float('inf'))
            if eval_loss < self.best_metric:
                self.best_metric = eval_loss
                self.save_model("best_model")
                logger.info(f"💾 Saved new best model with eval_loss: {eval_loss:.4f}")
            
            self.epochs_trained += 1
        
        # Final model save
        self.save_model("final_model")
        
        # Training summary
        total_time = time.time() - self.start_time
        self._log_training_summary(total_time)
        
        logger.info("🎉 Training completed successfully!")
    
    def _train_epoch(self, epoch: int, gradient_accumulation_steps: int, max_grad_norm: float) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        epoch_loss = 0.0
        epoch_kd_loss = 0.0
        epoch_task_loss = 0.0
        epoch_steps = 0
        tokens_processed = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True
        )
        
        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp and self.scaler:
                with torch.amp.autocast(device_type='cpu' if self.device.type == 'mps' else str(self.device)):
                    loss, loss_components = self._forward_step(batch)
                    loss = loss / gradient_accumulation_steps
            else:
                loss, loss_components = self._forward_step(batch)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update metrics
            tokens_processed += batch['input_ids'].numel()
            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_kd_loss += loss_components.get('kd_loss', 0)
            epoch_task_loss += loss_components.get('task_loss', 0)
            epoch_steps += 1
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                
                # Clip gradients
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
