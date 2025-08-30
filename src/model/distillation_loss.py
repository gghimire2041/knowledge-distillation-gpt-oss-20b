"""
Knowledge distillation loss functions for Nepali student model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining soft targets (teacher) and hard targets (ground truth).
    
    The loss combines:
    1. KL divergence between student and teacher (soft targets)
    2. Cross-entropy between student and ground truth (hard targets)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Distillation parameters
        self.temperature = config.get('temperature', 4.0)
        self.alpha = config.get('alpha', 0.7)  # Weight for knowledge distillation loss
        self.beta = config.get('beta', 0.3)    # Weight for task-specific loss
        
        # Loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Validate weights sum to 1
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            logger.warning(f"Loss weights don't sum to 1.0: alpha={self.alpha}, beta={self.beta}")
        
        logger.info(f"Initialized DistillationLoss: T={self.temperature}, α={self.alpha}, β={self.beta}")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate combined distillation loss.
        
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
            labels: Ground truth labels [batch_size, seq_len] (optional)
        
        Returns:
            Combined loss tensor
        """
        
        # Ensure tensors are on the same device
        if student_logits.device != teacher_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)
        
        # Knowledge distillation loss (KL divergence)
        kd_loss = self._compute_kd_loss(student_logits, teacher_logits)
        
        # Task-specific loss (cross-entropy with ground truth)
        if labels is not None and self.beta > 0:
            task_loss = self._compute_task_loss(student_logits, labels)
        else:
            task_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + self.beta * task_loss
        
        return total_loss
    
    def _compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss using KL divergence."""
        
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Compute KL divergence
        kd_loss = self.kl_loss(student_soft, teacher_soft)
        
        # Scale by temperature squared (following Hinton et al.)
        kd_loss = kd_loss * (self.temperature ** 2)
        
        return kd_loss
    
    def _compute_task_loss(self, student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute task-specific cross-entropy loss."""
        
        # Reshape for cross-entropy computation
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Flatten logits and labels
        flat_logits = student_logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)
        
        # Compute cross-entropy loss
        task_loss = self.ce_loss(flat_logits, flat_labels)
        
        return task_loss
    
    def get_loss_components(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual loss components for monitoring.
        
        Returns:
            Dictionary with individual loss values
        """
        
        with torch.no_grad():
            kd_loss = self._compute_kd_loss(student_logits, teacher_logits)
            
            if labels is not None:
                task_loss = self._compute_task_loss(student_logits, labels)
            else:
                task_loss = torch.tensor(0.0, device=student_logits.device)
            
            total_loss = self.alpha * kd_loss + self.beta * task_loss
        
        return {
            'kd_loss': kd_loss,
            'task_loss': task_loss,
            'total_loss': total_loss,
            'weighted_kd_loss': self.alpha * kd_loss,
            'weighted_task_loss': self.beta * task_loss
        }


class AdaptiveDistillationLoss(DistillationLoss):
    """
    Adaptive distillation loss that adjusts temperature and weights during training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Adaptive parameters
        self.min_temperature = config.get('min_temperature', 2.0)
        self.max_temperature = config.get('max_temperature', 8.0)
        self.temperature_decay = config.get('temperature_decay', 0.999)
        
        # Weight adaptation
        self.adaptive_weights = config.get('adaptive_weights', False)
        self.weight_adjustment_rate = config.get('weight_adjustment_rate', 0.01)
        
        # Training step counter
        self.step = 0
        
        logger.info(f"Initialized AdaptiveDistillationLoss: adaptive_weights={self.adaptive_weights}")
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with adaptive temperature and weights."""
        
        # Update step counter
        self.step += 1
        
        # Adapt temperature (cool down during training)
        if self.step % 100 == 0:  # Update every 100 steps
            self.temperature = max(
                self.min_temperature,
                self.temperature * self.temperature_decay
            )
        
        # Adapt weights if enabled
        if self.adaptive_weights and self.step % 500 == 0:
            self._adapt_weights(student_logits, teacher_logits, labels)
        
        return super().forward(student_logits, teacher_logits, labels)
    
    def _adapt_weights(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor]
    ):
        """Adapt loss weights based on relative performance."""
        
        with torch.no_grad():
            kd_loss = self._compute_kd_loss(student_logits, teacher_logits)
            
            if labels is not None:
                task_loss = self._compute_task_loss(student_logits, labels)
                
                # Adjust weights based on relative loss magnitudes
                total_loss = kd_loss + task_loss
                if total_loss > 0:
                    kd_weight = kd_loss / total_loss
                    task_weight = task_loss / total_loss
                    
                    # Smooth adjustment
                    self.alpha = self.alpha * (1 - self.weight_adjustment_rate) + kd_weight * self.weight_adjustment_rate
                    self.beta = self.beta * (1 - self.weight_adjustment_rate) + task_weight * self.weight_adjustment_rate
                    
                    # Ensure weights sum to 1
                    total_weight = self.alpha + self.beta
                    self.alpha /= total_weight
                    self.beta /= total_weight


class MultiTaskDistillationLoss(nn.Module):
    """
    Multi-task distillation loss for training on multiple Nepali language tasks simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Task configurations
        self.tasks = config.get('tasks', {})
        self.task_weights = config.get('task_weights', {})
        
        # Individual loss functions for each task
        self.loss_functions = nn.ModuleDict()
        
        for task_name, task_config in self.tasks.items():
            if task_config.get('type') == 'language_modeling':
                self.loss_functions[task_name] = DistillationLoss(task_config)
            elif task_config.get('type') == 'classification':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif task_config.get('type') == 'sequence_labeling':
                self.loss_functions[task_name] = nn.CrossEntropyLoss(ignore_index=-100)
            else:
                logger.warning(f"Unknown task type: {task_config.get('type')} for task {task_name}")
        
        logger.info(f"Initialized MultiTaskDistillationLoss with {len(self.tasks)} tasks")
    
    def forward(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Calculate multi-task loss.
        
        Args:
            outputs: Dictionary containing outputs for each task
                    Format: {
                        'task_name': {
                            'student_logits': torch.Tensor,
                            'teacher_logits': torch.Tensor,
                            'labels': torch.Tensor
                        }
                    }
        
        Returns:
            Combined multi-task loss
        """
        
        total_loss = 0.0
        task_losses = {}
        
        for task_name, task_outputs in outputs.items():
            if task_name not in self.loss_functions:
                continue
            
            loss_fn = self.loss_functions[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            if isinstance(loss_fn, DistillationLoss):
                task_loss = loss_fn(
                    student_logits=task_outputs['student_logits'],
                    teacher_logits=task_outputs['teacher_logits'],
                    labels=task_outputs.get('labels')
                )
            else:
                # Standard loss function
                task_loss = loss_fn(
                    task_outputs['student_logits'].view(-1, task_outputs['student_logits'].size(-1)),
                    task_outputs['labels'].view(-1)
                )
            
            weighted_loss = weight * task_loss
            total_loss += weighted_loss
            task_losses[task_name] = task_loss.detach()
        
        # Store individual task losses for monitoring
        if hasattr(self, '_last_task_losses'):
            self._last_task_losses = task_losses
        
        return total_loss
    
    def get_task_losses(self) -> Dict[str, float]:
        """Get the last computed task losses for monitoring."""
        return getattr(self, '_last_task_losses', {})


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better representations by contrasting 
    student and teacher hidden states.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.temperature = config.get('contrastive_temperature', 0.1)
        self.margin = config.get('margin', 1.0)
        self.similarity_function = config.get('similarity_function', 'cosine')
        
    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate contrastive loss between student and teacher hidden states.
        
        Args:
            student_hidden: Hidden states from student [batch_size, seq_len, hidden_size]
            teacher_hidden: Hidden states from teacher [batch_size, seq_len, hidden_size]
            labels: Optional labels for supervised contrastive learning
        
        Returns:
            Contrastive loss
        """
        
        # Flatten hidden states
        student_flat = student_hidden.view(-1, student_hidden.size(-1))  # [batch*seq, hidden]
        teacher_flat = teacher_hidden.view(-1, teacher_hidden.size(-1))   # [batch*seq, hidden]
        
        # Compute similarity
        if self.similarity_function == 'cosine':
            similarity = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        elif self.similarity_function == 'dot':
            similarity = torch.sum(student_flat * teacher_flat, dim=-1)
        else:
            # L2 distance
            similarity = -torch.norm(student_flat - teacher_flat, dim=-1)
        
        # Contrastive loss - maximize similarity between student and teacher
        loss = torch.mean(1 - similarity)  # Loss decreases as similarity increases
        
        return loss


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create appropriate loss function based on configuration.
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Configured loss function
    """
    
    loss_type = config.get('type', 'distillation')
    
    if loss_type == 'distillation':
        return DistillationLoss(config)
    elif loss_type == 'adaptive_distillation':
        return AdaptiveDistillationLoss(config)
    elif loss_type == 'multitask_distillation':
        return MultiTaskDistillationLoss(config)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
