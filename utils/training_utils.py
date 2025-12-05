"""
Training utilities: loss computation, checkpointing, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Optional
from transformers import AutoTokenizer


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    trigger_loss_weight: float = 1.0,
    llm_loss_weight: float = 1.0,
    temporal_loss_weight: float = 0.5,
    tokenizer: Optional[AutoTokenizer] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-task loss for video trigger model.
    
    Args:
        outputs: Model outputs with 'trigger_probs', 'llm_output', etc.
        targets: Ground truth with 'trigger_mask', 'descriptions', etc.
        trigger_loss_weight: Weight for trigger detection loss
        llm_loss_weight: Weight for LLM generation loss
        temporal_loss_weight: Weight for temporal consistency loss
        tokenizer: Tokenizer for encoding descriptions
    Returns:
        Dictionary with individual losses and total loss
    """
    losses = {}
    
    # Trigger detection loss
    if 'trigger_probs' in outputs and 'trigger_mask' in targets:
        trigger_probs = outputs['trigger_probs']
        trigger_mask = targets['trigger_mask']
        
        # Reshape for loss computation
        if trigger_probs.dim() == 3:
            # (B, T, num_classes)
            B, T, num_classes = trigger_probs.shape
            trigger_probs = trigger_probs.view(B * T, num_classes)
            trigger_mask = trigger_mask.view(B * T)
        
        if num_classes == 2:
            # Binary classification
            trigger_loss = F.cross_entropy(trigger_probs, trigger_mask, reduction='mean')
        else:
            # Multi-class
            trigger_loss = F.cross_entropy(trigger_probs, trigger_mask, reduction='mean')
        
        losses['trigger_loss'] = trigger_loss
    
    # LLM generation loss
    if 'llm_output' in outputs and 'descriptions' in targets:
        llm_output = outputs['llm_output']
        if isinstance(llm_output, dict):
            logits = llm_output.get('logits')
        else:
            logits = llm_output
        
        descriptions = targets['descriptions']
        
        if logits is not None and tokenizer is not None:
            # Tokenize descriptions
            try:
                encoded = tokenizer(
                    descriptions,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(logits.device)
                attention_mask = encoded['attention_mask'].to(logits.device)
                
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Flatten
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Compute loss only on non-padding tokens
                llm_loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id else -100,
                    reduction='mean'
                )
                losses['llm_loss'] = llm_loss
            except Exception as e:
                # Fallback: use a simple loss
                losses['llm_loss'] = torch.tensor(0.0, device=logits.device)
        else:
            losses['llm_loss'] = torch.tensor(0.0)
    
    # Temporal consistency loss (encourage smooth predictions)
    if 'trigger_probs' in outputs:
        trigger_probs = outputs['trigger_probs']
        if trigger_probs.dim() == 3:
            # Compute temporal smoothness
            B, T, num_classes = trigger_probs.shape
            if T > 1:
                # Difference between consecutive frames
                diff = trigger_probs[:, 1:, :] - trigger_probs[:, :-1, :]
                temporal_loss = torch.mean(torch.abs(diff))
                losses['temporal_loss'] = temporal_loss
            else:
                losses['temporal_loss'] = torch.tensor(0.0, device=trigger_probs.device)
        else:
            losses['temporal_loss'] = torch.tensor(0.0)
    
    # Compute total loss
    total_loss = (
        trigger_loss_weight * losses.get('trigger_loss', torch.tensor(0.0)) +
        llm_loss_weight * losses.get('llm_loss', torch.tensor(0.0)) +
        temporal_loss_weight * losses.get('temporal_loss', torch.tensor(0.0))
    )
    
    losses['total_loss'] = total_loss
    
    return losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: str,
    filename: Optional[str] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # Extract LLM model name from model if available
    llm_model_name = None
    if hasattr(model, 'temporal_llm') and hasattr(model.temporal_llm, 'llm_model_name'):
        llm_model_name = model.temporal_llm.llm_model_name
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Add model configuration metadata if available
    if llm_model_name is not None:
        checkpoint['llm_model_name'] = llm_model_name
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
    Returns:
        Epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check for LLM model name mismatch if metadata is available
    checkpoint_llm_name = checkpoint.get('llm_model_name', None)
    if checkpoint_llm_name is not None and hasattr(model, 'temporal_llm') and hasattr(model.temporal_llm, 'llm_model_name'):
        current_llm_name = model.temporal_llm.llm_model_name
        if checkpoint_llm_name != current_llm_name:
            print(f"⚠️  WARNING: LLM model mismatch detected during training resume!")
            print(f"   Checkpoint was saved with: {checkpoint_llm_name}")
            print(f"   Current model uses: {current_llm_name}")
            print("   This may cause loading errors if architectures differ.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
    
    return epoch

