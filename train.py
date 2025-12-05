"""
Training script for Video Trigger Model
Implements the multi-stage training loop as described in the blueprint
"""

import os
# Set tokenizers parallelism before any tokenizer imports to avoid warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from typing import Optional
from tqdm import tqdm
from transformers import AutoTokenizer

from models import VideoTriggerModel
from utils import ActivityNetDataset, collate_fn, compute_loss, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train Video Trigger Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    tokenizer: AutoTokenizer,
    writer: Optional[SummaryWriter] = None,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    trigger_loss_sum = 0.0
    llm_loss_sum = 0.0
    temporal_loss_sum = 0.0
    
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        timestamps = batch['timestamps'].to(device)  # (B, T)
        attention_mask = batch['attention_mask'].to(device)  # (B, T)
        trigger_mask = batch['trigger_mask'].to(device)  # (B, T)
        
        # Forward pass with mixed precision if enabled
        # Use torch.amp.autocast for better compatibility across devices
        if use_amp:
            # Determine device type for autocast
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=device_type):
                outputs = model(frames, timestamps=timestamps, return_triggers=True)
                
                # Prepare targets
                targets = {
                    'trigger_mask': trigger_mask,
                    'descriptions': batch['descriptions']
                }
                
                # Compute loss
                losses = compute_loss(
                    outputs,
                    targets,
                    trigger_loss_weight=config['training']['trigger_loss_weight'],
                    llm_loss_weight=config['training']['llm_loss_weight'],
                    temporal_loss_weight=config['training']['temporal_loss_weight'],
                    tokenizer=tokenizer
                )
                
                loss = losses['total_loss']
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
        else:
            # Standard forward pass
            outputs = model(frames, timestamps=timestamps, return_triggers=True)
            
            # Prepare targets
            targets = {
                'trigger_mask': trigger_mask,
                'descriptions': batch['descriptions']
            }
            
            # Compute loss
            losses = compute_loss(
                outputs,
                targets,
                trigger_loss_weight=config['training']['trigger_loss_weight'],
                llm_loss_weight=config['training']['llm_loss_weight'],
                temporal_loss_weight=config['training']['temporal_loss_weight'],
                tokenizer=tokenizer
            )
            
            loss = losses['total_loss']
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass with mixed precision
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                # Gradient clipping with scaler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate losses (multiply by gradient_accumulation_steps to get true loss)
        total_loss += loss.item() * gradient_accumulation_steps
        trigger_loss_sum += losses.get('trigger_loss', torch.tensor(0.0)).item()
        llm_loss_sum += losses.get('llm_loss', torch.tensor(0.0)).item()
        temporal_loss_sum += losses.get('temporal_loss', torch.tensor(0.0)).item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'trigger': f'{losses.get("trigger_loss", torch.tensor(0.0)).item():.4f}',
            'llm': f'{losses.get("llm_loss", torch.tensor(0.0)).item():.4f}'
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/TriggerLoss', losses.get('trigger_loss', torch.tensor(0.0)).item(), global_step)
            writer.add_scalar('Train/LLMLoss', losses.get('llm_loss', torch.tensor(0.0)).item(), global_step)
            writer.add_scalar('Train/TemporalLoss', losses.get('temporal_loss', torch.tensor(0.0)).item(), global_step)
    
    avg_loss = total_loss / len(dataloader)
    avg_trigger_loss = trigger_loss_sum / len(dataloader)
    avg_llm_loss = llm_loss_sum / len(dataloader)
    avg_temporal_loss = temporal_loss_sum / len(dataloader)
    
    return {
        'total_loss': avg_loss,
        'trigger_loss': avg_trigger_loss,
        'llm_loss': avg_llm_loss,
        'temporal_loss': avg_temporal_loss
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    tokenizer: AutoTokenizer,
    use_amp: bool = False
):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct_triggers = 0
    total_triggers = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            frames = batch['frames'].to(device)
            timestamps = batch['timestamps'].to(device)
            trigger_mask = batch['trigger_mask'].to(device)
            
            with torch.no_grad():
                # Validation
                if use_amp:
                    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type=device_type):
                        outputs = model(frames, timestamps=timestamps, return_triggers=True)
                else:
                    outputs = model(frames, timestamps=timestamps, return_triggers=True)
            
            targets = {
                'trigger_mask': trigger_mask,
                'descriptions': batch['descriptions']
            }
            
            losses = compute_loss(
                outputs,
                targets,
                trigger_loss_weight=config['training']['trigger_loss_weight'],
                llm_loss_weight=config['training']['llm_loss_weight'],
                temporal_loss_weight=config['training']['temporal_loss_weight'],
                tokenizer=tokenizer
            )
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
            
            # Compute trigger accuracy
            if 'trigger_probs' in outputs:
                trigger_probs = outputs['trigger_probs']
                if trigger_probs.dim() == 3:
                    pred_triggers = trigger_probs.argmax(dim=-1)
                    correct_triggers += (pred_triggers == trigger_mask).sum().item()
                    total_triggers += trigger_mask.numel()
    
    # Handle empty validation set
    if num_batches == 0:
        print("Warning: Validation set is empty. Skipping validation.")
        return {
            'loss': float('inf'),
            'trigger_accuracy': 0.0
        }
    
    avg_loss = total_loss / num_batches
    accuracy = correct_triggers / total_triggers if total_triggers > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'trigger_accuracy': accuracy
    }


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['llm']['model_name'],
        trust_remote_code=True  # Needed for Gemma models
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Create model
    model = VideoTriggerModel(
        visual_encoder_type=config['model']['visual_encoder']['type'],
        visual_encoder_pretrained=config['model']['visual_encoder']['pretrained'],
        visual_feature_dim=config['model']['visual_encoder']['feature_dim'],
        trigger_input_dim=config['model']['trigger_detector']['input_dim'],
        trigger_hidden_dim=config['model']['trigger_detector']['hidden_dim'],
        trigger_num_classes=config['model']['trigger_detector']['num_classes'],
        trigger_threshold=config['model']['trigger_detector']['threshold'],
        time_aware_input_dim=config['model']['time_aware_encoder']['input_dim'],
        time_aware_hidden_dim=config['model']['time_aware_encoder']['hidden_dim'],
        time_aware_num_layers=config['model']['time_aware_encoder']['num_layers'],
        time_aware_num_heads=config['model']['time_aware_encoder']['num_heads'],
        time_aware_encoder_type=config['model']['time_aware_encoder']['type'],
        llm_model_name=config['model']['llm'].get('model_name', 'google/gemma-3-1b-it'),
        llm_max_length=config['model']['llm']['max_length'],
        use_temporal_lstm=config['model']['llm'].get('use_temporal_lstm', True),
        temporal_lstm_hidden=config['model']['llm'].get('temporal_lstm_hidden', 512),
        temporal_lstm_layers=config['model']['llm'].get('temporal_lstm_layers', 2),
        llm_dtype=config['model']['llm'].get('dtype', 'float32'),
        use_gradient_checkpointing=config['model']['llm'].get('use_gradient_checkpointing', True),
        use_llava=config['model']['llm'].get('use_llava', True),
        llava_model_name=config['model']['llm'].get('llava_model_name', 'llava-hf/llava-1.5-7b-hf'),
        clip_window_size=config['data']['clip_window_size'],
        clip_overlap=config['data']['clip_overlap']
    ).to(device)
    
    # Apply freezing configuration
    freeze_config = config['model'].get('freeze', {})
    model.freeze_components(
        freeze_visual_backbone=freeze_config.get('visual_backbone', True),
        freeze_visual_projection=freeze_config.get('visual_projection', False),
        freeze_trigger_detector=freeze_config.get('trigger_detector', False),
        freeze_time_aware_encoder=freeze_config.get('time_aware_encoder', False),
        freeze_llm=freeze_config.get('llm_base', True),
        freeze_temporal_lstm=freeze_config.get('temporal_lstm', False),
        freeze_llm_projections=freeze_config.get('llm_projections', False)
    )
    
    # Print parameter counts
    print("\n=== Model Parameter Counts ===")
    param_counts = model.count_parameters(trainable_only=True)
    total_params = param_counts['total']
    total_params_m = total_params / 1e6
    
    print(f"Trainable Parameters:")
    print(f"  Visual Encoder: {param_counts['visual_encoder']:,} ({param_counts['visual_encoder']/1e6:.2f}M)")
    print(f"  Trigger Detector: {param_counts['trigger_detector']:,} ({param_counts['trigger_detector']/1e6:.2f}M)")
    print(f"  Time-aware Encoder: {param_counts['time_aware_encoder']:,} ({param_counts['time_aware_encoder']/1e6:.2f}M)")
    print(f"  Temporal LLM: {param_counts['temporal_llm']:,} ({param_counts['temporal_llm']/1e6:.2f}M)")
    print(f"  Total Trainable: {total_params:,} ({total_params_m:.2f}M)")
    
    # Count frozen parameters
    param_counts_all = model.count_parameters(trainable_only=False)
    frozen_params = param_counts_all['total'] - total_params
    frozen_params_m = frozen_params / 1e6
    print(f"\nFrozen Parameters: {frozen_params:,} ({frozen_params_m:.2f}M)")
    print(f"Total Parameters: {param_counts_all['total']:,} ({param_counts_all['total']/1e6:.2f}M)")
    print(f"Trainable Ratio: {total_params/param_counts_all['total']*100:.2f}%")
    print("=" * 40 + "\n")
    
    # Create datasets - try FiftyOne first, fall back to direct JSON loading
    use_fiftyone = config['data'].get('use_fiftyone', False)  # Default to False to avoid MongoDB requirement
    auto_split = config['data'].get('auto_split', True)  # Automatically split train/val if validation is empty
    val_ratio = config['data'].get('val_ratio', 0.2)  # Validation split ratio
    split_seed = config['data'].get('split_seed', 42)  # Random seed for reproducible splits
    
    # Set random seed for consistent train/val splits
    import random
    random.seed(split_seed)
    import numpy as np
    np.random.seed(split_seed)
    
    print("\n=== Loading Training Dataset ===")
    train_dataset = ActivityNetDataset(
        split="train",
        frame_sampling_rate=config['data']['frame_sampling_rate'],
        clip_window_size=config['data']['clip_window_size'],
        video_dir=config['data'].get('train_video_dir'),
        annotations_path=config['data'].get('train_annotations'),
        use_fiftyone=use_fiftyone,
        auto_split=auto_split,
        val_ratio=val_ratio,
        split_seed=split_seed,
        train_annotations_path=config['data'].get('train_annotations'),
        train_video_dir=config['data'].get('train_video_dir'),
        max_frames=config['data'].get('max_frames', None)
    )
    print(f"Training dataset: {len(train_dataset)} samples\n")
    
    print("=== Loading Validation Dataset ===")
    val_dataset = ActivityNetDataset(
        split="validation",
        frame_sampling_rate=config['data']['frame_sampling_rate'],
        clip_window_size=config['data']['clip_window_size'],
        video_dir=config['data'].get('val_video_dir'),
        annotations_path=config['data'].get('val_annotations'),
        use_fiftyone=use_fiftyone,
        auto_split=auto_split,
        val_ratio=val_ratio,
        split_seed=split_seed,
        train_annotations_path=config['data'].get('train_annotations'),
        train_video_dir=config['data'].get('train_video_dir'),
        max_frames=config['data'].get('max_frames', None)
    )
    print(f"Validation dataset: {len(val_dataset)} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Mixed precision training setup
    use_amp = config['training'].get('use_mixed_precision', True)
    
    # Disable AMP if CUDA is not available to avoid "User provided device_type of 'cuda'" warning
    if device.type != 'cuda' and use_amp:
        print("⚠️  CUDA not available, disabling mixed precision (AMP) to avoid errors.")
        use_amp = False
    
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    if use_amp:
        print(f"✓ Mixed precision training enabled (AMP)")
    
    # Use standard torch.amp.autocast for newer PyTorch versions (cpu or cuda)
    # For older versions or specific needs, torch.cuda.amp.autocast is used
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    # Optimizer
    # Ensure learning_rate and weight_decay are floats (YAML may parse scientific notation as strings)
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = config['training']['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, tokenizer, writer,
            scaler=scaler, use_amp=use_amp
        )
        
        # Update learning rate
        if epoch < warmup_steps:
            scheduler.step()
        
        # Log epoch metrics
        print(f"Epoch {epoch} - Train Loss: {train_metrics['total_loss']:.4f}")
        if writer:
            writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
            writer.add_scalar('Epoch/TrainTriggerLoss', train_metrics['trigger_loss'], epoch)
            writer.add_scalar('Epoch/TrainLLMLoss', train_metrics['llm_loss'], epoch)
        
        # Validate
        if (epoch + 1) % config['training']['eval_every'] == 0:
            if len(val_dataset) > 0:
                val_metrics = validate(model, val_loader, device, config, tokenizer, use_amp=use_amp)
                print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                      f"Trigger Accuracy: {val_metrics['trigger_accuracy']:.4f}")
                if writer:
                    writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
                    writer.add_scalar('Epoch/ValTriggerAccuracy', val_metrics['trigger_accuracy'], epoch)
            else:
                print(f"Epoch {epoch} - Validation skipped (empty validation set)")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics['total_loss'],
                config['training']['save_dir']
            )
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, config['training']['num_epochs'] - 1,
        train_metrics['total_loss'], config['training']['save_dir'],
        filename='final_checkpoint.pt'
    )
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()

