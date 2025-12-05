"""
Inference script for Video Trigger Model
Processes video and generates analysis only when triggers are detected
"""

import torch
import yaml
import argparse
import json
import os
from tqdm import tqdm

from models import VideoTriggerModel
from utils import VideoProcessor, FrameSampler, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Video Trigger Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default='outputs/analysis.json',
                        help='Path to output JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Trigger detection threshold (overrides config)')
    parser.add_argument('--strict-loading', action='store_true',
                        help='Use strict checkpoint loading (fails if architectures mismatch)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Clear GPU cache if using CUDA to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load checkpoint FIRST to check if LLaVA weights are present
    print(f"Loading checkpoint from {args.checkpoint} to CPU...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint_state = checkpoint['model_state_dict']
    
    # Print checkpoint architecture information
    print("\n" + "="*60)
    print("CHECKPOINT ARCHITECTURE INFORMATION")
    print("="*60)
    
    # Get checkpoint metadata
    epoch = checkpoint.get('epoch', 'Unknown')
    loss = checkpoint.get('loss', 'Unknown')
    checkpoint_llm_name = checkpoint.get('llm_model_name', None)
    
    print(f"Epoch: {epoch}")
    print(f"Loss: {loss}")
    if checkpoint_llm_name:
        print(f"LLM Model (from checkpoint): {checkpoint_llm_name}")
    
    # Analyze checkpoint keys to determine architecture
    all_keys = list(checkpoint_state.keys())
    total_params = sum(v.numel() if isinstance(v, torch.Tensor) else 0 for v in checkpoint_state.values())
    
    # Component detection
    has_visual_encoder = any('visual_encoder' in key for key in all_keys)
    has_trigger_detector = any('trigger_detector' in key for key in all_keys)
    has_time_aware = any('time_aware_encoder' in key for key in all_keys)
    has_llava = any('temporal_llm.model.' in key for key in all_keys)
    has_temporal_llm = any('temporal_llm.llm.' in key for key in all_keys) and not has_llava
    has_temporal_lstm = any('temporal_llm.temporal_lstm' in key for key in all_keys)
    
    # Visual encoder type
    visual_encoder_type = "Unknown"
    if any('visual_encoder.backbone.0' in key for key in all_keys):
        visual_encoder_type = "ResNet (detected from structure)"
    elif any('visual_encoder.backbone' in key for key in all_keys):
        visual_encoder_type = "CNN-based"
    
    # LLM type detection
    llm_type = "Unknown"
    if has_llava:
        llm_type = "LLaVA (Vision-Language Model)"
        llava_keys = [k for k in all_keys if 'temporal_llm.model.' in k]
        print(f"  LLaVA keys found: {len(llava_keys)}")
    elif has_temporal_llm:
        if any('temporal_llm.llm.model.layers' in key for key in all_keys):
            llm_type = "LLaMA-style (Gemma/LLaMA)"
        elif any('temporal_llm.llm.h.' in key for key in all_keys):
            llm_type = "GPT-2 style"
        else:
            llm_type = "Text-only LLM"
    
    print(f"\nModel Components:")
    print(f"  Visual Encoder: {'✓' if has_visual_encoder else '✗'} ({visual_encoder_type})")
    print(f"  Trigger Detector: {'✓' if has_trigger_detector else '✗'}")
    print(f"  Time-aware Encoder: {'✓' if has_time_aware else '✗'}")
    print(f"  Temporal LSTM: {'✓' if has_temporal_lstm else '✗'}")
    print(f"  LLM Type: {llm_type}")
    
    # Count parameters by component
    component_params = {
        'visual_encoder': 0,
        'trigger_detector': 0,
        'time_aware_encoder': 0,
        'temporal_llm': 0,
        'other': 0
    }
    
    for key, value in checkpoint_state.items():
        if isinstance(value, torch.Tensor):
            param_count = value.numel()
            if 'visual_encoder' in key:
                component_params['visual_encoder'] += param_count
            elif 'trigger_detector' in key:
                component_params['trigger_detector'] += param_count
            elif 'time_aware_encoder' in key:
                component_params['time_aware_encoder'] += param_count
            elif 'temporal_llm' in key:
                component_params['temporal_llm'] += param_count
            else:
                component_params['other'] += param_count
    
    print(f"\nParameter Counts:")
    for component, count in component_params.items():
        if count > 0:
            count_m = count / 1e6
            percentage = (count / total_params * 100) if total_params > 0 else 0
            print(f"  {component.replace('_', ' ').title()}: {count:,} ({count_m:.2f}M) - {percentage:.1f}%")
    
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Total State Dict Keys: {len(all_keys)}")
    print("="*60 + "\n")
    
    # Check if LLaVA weights are in checkpoint (if frozen during training, they should be)
    has_llava_in_checkpoint = has_llava
    use_llava = config['model']['llm'].get('use_llava', True)
    skip_llava_loading = has_llava_in_checkpoint and use_llava
    
    if has_llava_in_checkpoint:
        print("✓ LLaVA weights found in checkpoint - will skip HuggingFace model loading")
        print("  (Only processor will be loaded, model weights will come from checkpoint)")
    elif use_llava:
        print("⚠️  LLaVA weights not in checkpoint - will load from HuggingFace")
        print("  (This may happen if LLaVA was not included in checkpoint saving)")
    
    # Create model (with skip_llava_loading flag if checkpoint has weights)
    print("Creating model architecture...")
    model = VideoTriggerModel(
        visual_encoder_type=config['model']['visual_encoder']['type'],
        visual_encoder_pretrained=False,  # Don't load ImageNet weights - checkpoint has trained weights
        visual_feature_dim=config['model']['visual_encoder']['feature_dim'],
        trigger_input_dim=config['model']['trigger_detector']['input_dim'],
        trigger_hidden_dim=config['model']['trigger_detector']['hidden_dim'],
        trigger_num_classes=config['model']['trigger_detector']['num_classes'],
        trigger_threshold=args.threshold or config['inference']['trigger_threshold'],
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
        use_llava=use_llava,
        llava_model_name=config['model']['llm'].get('llava_model_name', 'llava-hf/llava-1.5-7b-hf'),
        skip_llava_loading=skip_llava_loading,  # Skip HuggingFace loading if checkpoint has weights
        clip_window_size=config['data']['clip_window_size'],
        clip_overlap=config['data']['clip_overlap']
    )
    
    # Move model to device after creation
    model = model.to(device)
    
    # If we skipped LLaVA loading, ensure model structure exists before loading checkpoint
    # (We need the model structure to load checkpoint weights into)
    if skip_llava_loading and hasattr(model, 'temporal_llm') and hasattr(model.temporal_llm, 'model') and model.temporal_llm.model is None:
        print("Loading LLaVA model structure (required to load checkpoint weights)...")
        model.temporal_llm.ensure_model_loaded()
        model = model.to(device)  # Move to device again after loading
    
    # Print model architecture using PyTorch's native printing
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("="*60)
    
    # Print detailed parameter information
    print("\nMODEL PARAMETER SUMMARY")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)")
    
    # Print parameter counts by component if available
    if hasattr(model, 'count_parameters'):
        param_counts = model.count_parameters(trainable_only=False)
        print("\nParameters by Component:")
        for component, count in param_counts.items():
            if component != 'total' and count > 0:
                count_m = count / 1e6
                percentage = (count / total_params * 100) if total_params > 0 else 0
                print(f"  {component.replace('_', ' ').title()}: {count:,} ({count_m:.2f}M) - {percentage:.1f}%")
    print("="*60 + "\n")
    
    # Get LLM model names for comparison
    checkpoint_llm_name = checkpoint.get('llm_model_name', None)
    current_llm_name = config['model']['llm']['model_name']
    
    # Check if checkpoint has incompatible LLM architecture
    has_gpt2_llm = any('temporal_llm.llm.h.' in key or 'temporal_llm.llm.wte' in key for key in checkpoint_state.keys())
    has_llama_llm = any('temporal_llm.llm.model.layers.' in key or 'temporal_llm.llm.model.embed_tokens' in key for key in checkpoint_state.keys())
    
    is_llama_model = 'llama' in current_llm_name.lower() or 'gemma' in current_llm_name.lower() or 'gemini' in current_llm_name.lower()
    
    # If strict loading is requested, try that first
    if args.strict_loading:
        try:
            model.load_state_dict(checkpoint_state, strict=True)
            print(f"✓ Loaded checkpoint from {args.checkpoint} (strict mode)")
        except RuntimeError as e:
            print(f"❌ Strict loading failed: {e}")
            print("   Use without --strict-loading to load compatible parts only.")
            raise
    
    # Check LLM model name match if metadata is available
    llm_names_match = False
    if checkpoint_llm_name is not None:
        llm_names_match = checkpoint_llm_name == current_llm_name
        if not llm_names_match:
            print(f"⚠️  WARNING: LLM model mismatch detected!")
            print(f"   Checkpoint was saved with: {checkpoint_llm_name}")
            print(f"   Current config uses: {current_llm_name}")
            print("   This may cause size mismatches in LLM weights.")
    
    if has_gpt2_llm and is_llama_model:
        print("⚠️  WARNING: Checkpoint was saved with GPT-2, but current config uses LLaMA/Gemma.")
        print("   Loading compatible parts only (visual encoder, trigger detector, time-aware encoder, temporal LSTM).")
        print("   LLM weights will be initialized from pretrained model (not from checkpoint).")
        print("   Consider retraining with the new LLM or use GPT-2 for inference.")
        
        # Filter out incompatible LLM keys
        compatible_state = {}
        incompatible_keys = []
        for key, value in checkpoint_state.items():
            if 'temporal_llm.llm.' in key and ('h.' in key or 'wte' in key or 'wpe' in key or 'ln_f' in key):
                incompatible_keys.append(key)
            elif 'temporal_llm.feature_projection' in key or 'temporal_llm.output_projection' in key:
                # These might have size mismatches, skip them
                incompatible_keys.append(key)
            else:
                compatible_state[key] = value
        
        # Load compatible parts
        missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
        
        if incompatible_keys:
            print(f"   Skipped {len(incompatible_keys)} incompatible LLM-related keys")
        if missing_keys:
            print(f"   Missing keys (will use default initialization): {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"     - {key}")
            else:
                for key in missing_keys[:5]:
                    print(f"     - {key}")
                print(f"     ... and {len(missing_keys) - 5} more")
    elif has_llama_llm:
        # Both checkpoint and current model are LLaMA-style
        if checkpoint_llm_name is not None and llm_names_match:
            # Same model name - try to load everything with size checking
            print(f"✓ Checkpoint LLM model matches current config: {current_llm_name}")
            print("   Attempting to load all weights with size verification...")
            
            compatible_state = {}
            incompatible_keys = []
            size_mismatches = []
            model_state_dict = dict(model.named_parameters())
            
            for key, checkpoint_value in checkpoint_state.items():
                if key in model_state_dict:
                    current_param = model_state_dict[key]
                    if current_param.shape == checkpoint_value.shape:
                        compatible_state[key] = checkpoint_value
                    else:
                        # Size mismatch - record it
                        incompatible_keys.append(key)
                        size_mismatches.append((key, checkpoint_value.shape, current_param.shape))
                else:
                    # Key doesn't exist in current model
                    incompatible_keys.append(key)
            
            # Load compatible parts
            try:
                missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
                
                if incompatible_keys:
                    print(f"   ⚠️  Found {len(incompatible_keys)} keys with size mismatches:")
                    # Show first few mismatches
                    for key, ckpt_shape, curr_shape in size_mismatches[:5]:
                        print(f"     - {key}: checkpoint {ckpt_shape} vs current {curr_shape}")
                    if len(size_mismatches) > 5:
                        print(f"     ... and {len(size_mismatches) - 5} more mismatches")
                    print("   These weights will be initialized from pretrained model.")
                
                if missing_keys:
                    print(f"   Missing keys (will use default initialization): {len(missing_keys)}")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            print(f"     - {key}")
                    else:
                        for key in missing_keys[:5]:
                            print(f"     - {key}")
                        print(f"     ... and {len(missing_keys) - 5} more")
            except RuntimeError as e:
                print(f"   Error loading compatible parts: {e}")
                print("   This may indicate additional incompatibilities. Continuing with pretrained weights...")
        else:
            # Different model or no metadata - be more cautious
            if checkpoint_llm_name is not None:
                print(f"⚠️  WARNING: Checkpoint was saved with {checkpoint_llm_name}, but current config uses {current_llm_name}.")
            else:
                print("⚠️  WARNING: Checkpoint was saved with a LLaMA-style model (metadata not available).")
            print("   Checking for size mismatches and loading compatible parts only...")
            
            # Pre-check all keys for size compatibility before attempting to load
            compatible_state = {}
            incompatible_keys = []
            size_mismatches = []
            model_state_dict = dict(model.named_parameters())
            
            for key, checkpoint_value in checkpoint_state.items():
                if key in model_state_dict:
                    current_param = model_state_dict[key]
                    if current_param.shape == checkpoint_value.shape:
                        compatible_state[key] = checkpoint_value
                    else:
                        # Size mismatch - record it
                        incompatible_keys.append(key)
                        size_mismatches.append((key, checkpoint_value.shape, current_param.shape))
                else:
                    # Key doesn't exist in current model
                    incompatible_keys.append(key)
            
            # Load compatible parts
            try:
                missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
                
                if incompatible_keys:
                    print(f"   Skipped {len(incompatible_keys)} incompatible keys (size mismatches or missing)")
                    if size_mismatches:
                        print("   Size mismatches found:")
                        for key, ckpt_shape, curr_shape in size_mismatches[:5]:
                            print(f"     - {key}: checkpoint {ckpt_shape} vs current {curr_shape}")
                        if len(size_mismatches) > 5:
                            print(f"     ... and {len(size_mismatches) - 5} more")
                    print("   LLM weights will be initialized from pretrained model (not from checkpoint).")
                    if checkpoint_llm_name is not None:
                        print(f"   Suggestion: Use --config with llm.model_name: {checkpoint_llm_name} to match the checkpoint.")
                if missing_keys:
                    print(f"   Missing keys (will use default initialization): {len(missing_keys)}")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            print(f"     - {key}")
                    else:
                        for key in missing_keys[:5]:
                            print(f"     - {key}")
                        print(f"     ... and {len(missing_keys) - 5} more")
            except RuntimeError as e:
                print(f"   Error loading compatible parts: {e}")
                print("   This may indicate additional incompatibilities. Continuing with pretrained weights...")
    else:
        # Try normal loading first
        try:
            model.load_state_dict(checkpoint_state, strict=True)
            print(f"✓ Loaded checkpoint from {args.checkpoint}")
        except RuntimeError as e:
            print(f"⚠️  Warning: Could not load checkpoint strictly: {e}")
            print("   Attempting to load compatible parts only...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
            if missing_keys:
                print(f"   Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"   Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    # Initialize video processors
    frame_sampler = FrameSampler(fps=config['data']['frame_sampling_rate'])
    video_processor = VideoProcessor()
    
    print(f"Processing video: {args.video}")
    
    # Sample frames from video
    frames, timestamps = frame_sampler.sample_frames(args.video)
    print(f"Sampled {len(frames)} frames")
    
    # Process frames to tensors
    frame_tensors = video_processor.process_frames(frames)  # (T, C, H, W)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    
    # Move tensors to device
    frame_tensors = frame_tensors.to(device)
    timestamps_tensor = timestamps_tensor.to(device)
    
    # Determine threshold
    threshold = args.threshold if args.threshold is not None else config['inference']['trigger_threshold']
    
    # Get LLaVA image size from config (default: 336x336, optimal for LLaVA)
    llava_image_size = config['inference'].get('llava_image_size', [336, 336])
    if isinstance(llava_image_size, list):
        llava_image_size = tuple(llava_image_size)
    elif isinstance(llava_image_size, int):
        llava_image_size = (llava_image_size, llava_image_size)
    
    # Get prompts from config
    llava_prompt = config['inference'].get('llava_prompt', None)
    llm_prompt = config['inference'].get('llm_prompt', None)
    
    # Run inference
    print(f"Running inference with trigger threshold: {threshold:.3f}...")
    print(f"LLaVA will process frames at {llava_image_size[0]}x{llava_image_size[1]} resolution (optimal for scene description)")
    if llava_prompt:
        print(f"Using custom LLaVA prompt from config")
    if llm_prompt:
        print(f"Using custom LLM prompt from config")
    results = model.infer_triggered_analysis(
        frame_tensors,
        timestamps=timestamps_tensor,
        trigger_threshold=threshold,
        video_path=args.video,  # Pass video path for high-res frame extraction
        llava_image_size=llava_image_size,
        llava_prompt=llava_prompt,
        llm_prompt=llm_prompt
    )
    
    print(f"Detected {len(results)} triggers")
    
    if len(results) == 0:
        print(f"\n⚠️  No triggers detected. Try lowering the threshold:")
        print(f"   python inference.py ... --threshold 0.2")
        print(f"   or edit config.yaml: inference.trigger_threshold: 0.2")
    
    # Generate video summary from all analyses
    video_summary = None
    if len(results) > 0 and config['inference'].get('generate_summary', True):
        print("\n" + "="*60)
        print("GENERATING VIDEO SUMMARY")
        print("="*60)
        
        # Collect all analysis texts
        analysis_texts = [result['analysis'] for result in results if result.get('analysis')]
        
        if len(analysis_texts) > 0:
            try:
                # Use LLaVA to summarize all analyses
                if hasattr(model, 'temporal_llm') and hasattr(model.temporal_llm, 'summarize_text'):
                    summary_prompt = config['inference'].get('summary_prompt', None)
                    video_summary = model.temporal_llm.summarize_text(
                        texts=analysis_texts,
                        prompt=summary_prompt,
                        max_new_tokens=config['inference'].get('summary_max_tokens', 300),
                        temperature=0.7,
                        do_sample=False
                    )
                    print("✓ Video summary generated successfully")
                else:
                    # Fallback: simple concatenation if summarizer not available
                    video_summary = " ".join(analysis_texts)
                    print("⚠️  Summarizer not available, using concatenated analyses")
            except Exception as e:
                print(f"⚠️  Error generating summary: {e}")
                # Fallback to simple concatenation
                video_summary = " ".join(analysis_texts)
        else:
            video_summary = "No analyses available for summarization."
    
    # Format output
    output_data = {
        'video_path': args.video,
        'num_frames_analyzed': len(frames),
        'num_triggers_detected': len(results),
        'triggers': results
    }
    
    # Add summary if generated
    if video_summary is not None:
        output_data['video_summary'] = video_summary
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("DETAILED ANALYSIS BY TRIGGER")
    print("="*60)
    for i, result in enumerate(results):
        print(f"\nTrigger {i+1}:")
        print(f"  Timestamp: {result['timestamp']:.2f}s")
        print(f"  Frame Index: {result['frame_index']}")
        print(f"  Confidence: {result['trigger_confidence']:.4f}")
        print(f"  Analysis: {result['analysis']}")
    
    # Print video summary if available
    if video_summary is not None:
        print("\n" + "="*60)
        print("VIDEO SUMMARY")
        print("="*60)
        print(video_summary)
        print("="*60)


if __name__ == '__main__':
    main()

