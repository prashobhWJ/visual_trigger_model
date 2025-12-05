"""
Complete Video Trigger Model
Combines all stages: Visual Encoder, Trigger Detector, Time-aware Encoder, and Temporal LLM
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np

from .visual_encoder import VisualEncoder
from .trigger_detector import TriggerDetector
from .time_aware_encoder import TimeAwareEncoder
from .temporal_llm import TemporalLLM
from .temporal_llava import TemporalLLaVA


class VideoTriggerModel(nn.Module):
    """
    Complete multi-stage video analysis model with trigger detection and LLM-based reasoning.
    """
    
    def __init__(
        self,
        # Visual encoder config
        visual_encoder_type: str = "resnet50",
        visual_encoder_pretrained: bool = True,
        visual_feature_dim: int = 512,
        
        # Trigger detector config
        trigger_input_dim: int = 512,
        trigger_hidden_dim: int = 256,
        trigger_num_classes: int = 2,
        trigger_threshold: float = 0.5,
        
        # Time-aware encoder config
        time_aware_input_dim: int = 512,
        time_aware_hidden_dim: int = 768,
        time_aware_num_layers: int = 4,
        time_aware_num_heads: int = 8,
        time_aware_encoder_type: str = "transformer",
        
        # LLM config
        llm_model_name: str = "google/gemma-3-1b-it",
        llm_max_length: int = 512,
        use_temporal_lstm: bool = True,
        temporal_lstm_hidden: int = 512,
        temporal_lstm_layers: int = 2,
        llm_dtype: str = "float32",
        use_gradient_checkpointing: bool = True,
        use_llava: bool = True,  # Use LLaVA for vision-language understanding
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",  # LLaVA model name
        skip_llava_loading: bool = False,  # Skip loading LLaVA from HuggingFace (for inference when checkpoint has weights)
        
        # Clip extraction config
        clip_window_size: int = 16,
        clip_overlap: int = 4
    ):
        """
        Initialize the complete video trigger model.
        """
        super().__init__()
        
        # Stage 1: Visual Encoder
        self.visual_encoder = VisualEncoder(
            encoder_type=visual_encoder_type,
            pretrained=visual_encoder_pretrained,
            feature_dim=visual_feature_dim
        )
        
        # Stage 1: Trigger Detector
        self.trigger_detector = TriggerDetector(
            input_dim=trigger_input_dim,
            hidden_dim=trigger_hidden_dim,
            num_classes=trigger_num_classes,
            threshold=trigger_threshold
        )
        
        # Stage 2: Time-aware Encoder
        self.time_aware_encoder = TimeAwareEncoder(
            input_dim=time_aware_input_dim,
            hidden_dim=time_aware_hidden_dim,
            num_layers=time_aware_num_layers,
            num_heads=time_aware_num_heads,
            encoder_type=time_aware_encoder_type,
            use_timestamps=True
        )
        
        # Stage 3: Temporal LLM or LLaVA
        self.use_llava = use_llava
        if use_llava:
            # Use LLaVA for vision-language understanding
            self.temporal_llm = TemporalLLaVA(
                llava_model_name=llava_model_name,
                feature_dim=time_aware_hidden_dim,  # Not used by LLaVA but kept for compatibility
                max_length=llm_max_length,
                freeze_llm=True,  # Freeze LLaVA by default
                llm_dtype=llm_dtype,
                use_gradient_checkpointing=use_gradient_checkpointing,
                skip_loading=skip_llava_loading  # Skip HuggingFace loading if checkpoint has weights
            )
        else:
            # Use traditional TemporalLLM (text-only LLM)
            self.temporal_llm = TemporalLLM(
                llm_model_name=llm_model_name,
                feature_dim=time_aware_hidden_dim,
                max_length=llm_max_length,
                use_temporal_lstm=use_temporal_lstm,
                temporal_lstm_hidden=temporal_lstm_hidden,
                temporal_lstm_layers=temporal_lstm_layers,
                llm_dtype=llm_dtype,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
        
        # Clip extraction parameters
        self.clip_window_size = clip_window_size
        self.clip_overlap = clip_overlap
    
    def freeze_components(
        self,
        freeze_visual_backbone: bool = True,
        freeze_visual_projection: bool = False,
        freeze_trigger_detector: bool = False,
        freeze_time_aware_encoder: bool = False,
        freeze_llm: bool = True,
        freeze_temporal_lstm: bool = False,
        freeze_llm_projections: bool = False
    ):
        """
        Freeze/unfreeze model components for efficient training.
        
        Args:
            freeze_visual_backbone: Freeze pretrained visual encoder backbone
            freeze_visual_projection: Freeze visual encoder projection head
            freeze_trigger_detector: Freeze trigger detector (usually keep trainable)
            freeze_time_aware_encoder: Freeze time-aware encoder
            freeze_llm: Freeze base LLM model
            freeze_temporal_lstm: Freeze temporal LSTM
            freeze_llm_projections: Freeze LLM projection layers
        """
        # Visual encoder
        if freeze_visual_backbone:
            self.visual_encoder.freeze_backbone()
        else:
            self.visual_encoder.unfreeze_backbone()
        
        if freeze_visual_projection:
            self.visual_encoder.freeze_projection()
        else:
            self.visual_encoder.unfreeze_projection()
        
        # Trigger detector (usually keep trainable)
        if freeze_trigger_detector:
            for param in self.trigger_detector.parameters():
                param.requires_grad = False
        else:
            for param in self.trigger_detector.parameters():
                param.requires_grad = True
        
        # Time-aware encoder
        if freeze_time_aware_encoder:
            for param in self.time_aware_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.time_aware_encoder.parameters():
                param.requires_grad = True
        
        # Temporal LLM or LLaVA
        if freeze_llm:
            self.temporal_llm.freeze_llm()
        else:
            self.temporal_llm.unfreeze_llm()
        
        if not self.use_llava:
            # Only TemporalLLM has temporal LSTM
            if freeze_temporal_lstm:
                self.temporal_llm.freeze_temporal_lstm()
            else:
                self.temporal_llm.unfreeze_temporal_lstm()
        
        if freeze_llm_projections:
            self.temporal_llm.freeze_projections()
        else:
            self.temporal_llm.unfreeze_projections()
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self, trainable_only: bool = True):
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
        Returns:
            Dictionary with parameter counts by component
        """
        counts = {}
        
        if trainable_only:
            counts['visual_encoder'] = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
            counts['trigger_detector'] = sum(p.numel() for p in self.trigger_detector.parameters() if p.requires_grad)
            counts['time_aware_encoder'] = sum(p.numel() for p in self.time_aware_encoder.parameters() if p.requires_grad)
            counts['temporal_llm'] = sum(p.numel() for p in self.temporal_llm.parameters() if p.requires_grad)
            counts['total'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            counts['visual_encoder'] = sum(p.numel() for p in self.visual_encoder.parameters())
            counts['trigger_detector'] = sum(p.numel() for p in self.trigger_detector.parameters())
            counts['time_aware_encoder'] = sum(p.numel() for p in self.time_aware_encoder.parameters())
            counts['temporal_llm'] = sum(p.numel() for p in self.temporal_llm.parameters())
            counts['total'] = sum(p.numel() for p in self.parameters())
        
        return counts
    
    def forward(
        self,
        frames: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_triggers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            frames: Tensor of shape (B, T, C, H, W) - video frames
            timestamps: Optional tensor of shape (B, T) - timestamps for each frame
            return_triggers: Whether to return trigger detection results
        Returns:
            Dictionary with 'llm_output' and optionally 'triggers', 'trigger_probs'
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        
        # Stage 1: Extract visual features
        visual_features = self.visual_encoder(frames)  # (B, T, visual_feature_dim)
        
        # Stage 1: Detect triggers
        trigger_probs = self.trigger_detector(visual_features)  # (B, T, num_classes)
        # Clamp trigger probabilities: values below 0.1 become 0.1, ensuring range [0.1, 1.0]
        # This helps with triggering even when computed values are very small
        trigger_probs = torch.clamp(trigger_probs, min=0.1, max=1.0)
        
        # Stage 2: Encode with temporal awareness
        encoded_features = self.time_aware_encoder(
            visual_features,
            timestamps=timestamps
        )  # (B, T, time_aware_hidden_dim)
        
        # Stage 3: LLM processing
        # OPTION 1: Skip LLaVA during training (it's frozen and doesn't provide logits)
        # OPTION 3: During inference, only process frames where triggers are detected
        if self.use_llava:
            if self.training:
                # Skip LLaVA during training - it's frozen and doesn't provide logits anyway
                # This dramatically speeds up training (from ~13s/iter to <1s/iter)
                llm_output = {
                    'generated_text': None,
                    'logits': None,
                    'hidden_states': None,
                    'llava_output': None
                }
            else:
                # During inference: OPTION 3 - Only process frames with detected triggers
                # Use the trained trigger detector to decide if LLaVA analysis is needed
                # The trigger detector uses trained weights from checkpoint to detect events
                if trigger_probs.dim() == 3:
                    # (B, T, num_classes) - extract trigger class probability
                    if trigger_probs.shape[2] == 2:
                        # Binary classification: class 1 is trigger, class 0 is no-trigger
                        # Extract probability of trigger class (class 1)
                        trigger_confidences = trigger_probs[:, :, 1]  # (B, T) - probability of trigger
                    else:
                        # Multi-class: use max probability as confidence
                        trigger_confidences = trigger_probs.max(dim=-1)[0]  # (B, T)
                    
                    # Check if any frame in each batch exceeds threshold
                    # This uses the trained trigger detector's threshold from config
                    has_triggers = (trigger_confidences > self.trigger_detector.threshold).any(dim=1)  # (B,)
                else:
                    # Binary case - assume shape is (B, T) with probabilities
                    has_triggers = (trigger_probs > self.trigger_detector.threshold).any(dim=1)  # (B,)
                
                if has_triggers.any():
                    # OPTION 3: Only process batches that have triggers detected by trained model
                    # The trigger detector (with trained checkpoint weights) has determined
                    # that these frames contain events worth analyzing with LLaVA
                    llm_output = self.temporal_llm(frames)  # LLaVA processes frames directly
                else:
                    # No triggers detected by trained trigger detector
                    # Skip LLaVA processing entirely to save computation time
                    llm_output = {
                        'generated_text': [],
                        'logits': None,
                        'hidden_states': None,
                        'llava_output': []
                    }
        else:
            # Traditional LLM processes encoded features
            llm_output = self.temporal_llm(encoded_features)  # Dict with 'logits', 'hidden_states'
        
        result = {
            'llm_output': llm_output,
            'visual_features': visual_features,
            'encoded_features': encoded_features
        }
        
        if return_triggers:
            result['trigger_probs'] = trigger_probs
            result['triggers'] = trigger_probs.argmax(dim=-1) if trigger_probs.dim() > 2 else trigger_probs
        
        return result
    
    def extract_clip_features(
        self,
        frames: torch.Tensor,
        center_idx: int,
        timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features for a clip around a trigger point.
        
        Args:
            frames: Tensor of shape (T, C, H, W) - all frames
            center_idx: Index of the trigger frame
            timestamps: Optional tensor of shape (T,) - timestamps
        Returns:
            clip_features: Encoded features for the clip
            clip_timestamps: Timestamps for the clip
        """
        # Calculate clip boundaries
        half_window = self.clip_window_size // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(frames.shape[0], center_idx + half_window)
        
        # Extract clip frames
        clip_frames = frames[start_idx:end_idx]  # (clip_len, C, H, W)
        
        # Pad if necessary
        if clip_frames.shape[0] < self.clip_window_size:
            padding = self.clip_window_size - clip_frames.shape[0]
            clip_frames = torch.cat([
                clip_frames,
                clip_frames[-1:].repeat(padding, 1, 1, 1)
            ], dim=0)
        
        # Extract timestamps if provided
        if timestamps is not None:
            clip_timestamps = timestamps[start_idx:end_idx]
            if len(clip_timestamps) < self.clip_window_size:
                padding = self.clip_window_size - len(clip_timestamps)
                last_ts = timestamps[-1] if len(timestamps) > 0 else 0.0
                clip_timestamps = torch.cat([
                    clip_timestamps,
                    torch.full((padding,), last_ts, device=frames.device)
                ])
        else:
            clip_timestamps = torch.arange(
                start_idx, start_idx + self.clip_window_size,
                dtype=torch.float32,
                device=frames.device
            )
        
        # Add batch dimension
        clip_frames = clip_frames.unsqueeze(0)  # (1, clip_len, C, H, W)
        
        # Extract visual features
        visual_features = self.visual_encoder(clip_frames)  # (1, clip_len, visual_feature_dim)
        
        # Encode with temporal awareness
        clip_timestamps_batch = clip_timestamps.unsqueeze(0)  # (1, clip_len)
        encoded_features = self.time_aware_encoder(
            visual_features,
            timestamps=clip_timestamps_batch
        )  # (1, clip_len, time_aware_hidden_dim)
        
        # Remove batch dimension
        encoded_features = encoded_features.squeeze(0)  # (clip_len, time_aware_hidden_dim)
        
        return encoded_features, clip_timestamps
    
    def infer_triggered_analysis(
        self,
        frames: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        trigger_threshold: Optional[float] = None,
        video_path: Optional[str] = None,
        llava_image_size: Tuple[int, int] = (336, 336),
        max_frames: Optional[int] = None,
        llava_prompt: Optional[str] = None,
        llm_prompt: Optional[str] = None
    ) -> List[Dict]:
        """
        Inference method that only processes frames where triggers are detected.
        
        Args:
            frames: Tensor of shape (T, C, H, W) - video frames (low-res for trigger detection)
            timestamps: Optional tensor of shape (T,) - timestamps
            trigger_threshold: Detection threshold
            video_path: Optional path to video file (for extracting high-res frames for LLaVA)
            llava_image_size: Image size for LLaVA (default: 336x336, optimal for LLaVA)
            max_frames: Optional maximum number of frames to analyze with LLaVA (samples evenly if exceeded)
            llava_prompt: Optional prompt for LLaVA analysis (if None, uses default from temporal_llava)
            llm_prompt: Optional prompt for traditional LLM analysis (if None, uses default from temporal_llm)
        Returns:
            List of analysis results, each with 'timestamp', 'trigger_confidence', 'analysis'
        """
        self.eval()
        results = []
        
        with torch.no_grad():
            # Process frames in batches for efficiency
            batch_size = 32
            num_frames = frames.shape[0]
            
            # Sample frames at lower rate for trigger detection (to reduce computation)
            # Note: This is a SECOND sampling step - frames were already sampled in inference.py
            # If you're seeing triggers every 2 seconds, this might be too aggressive
            # Consider using all frames or a smaller sample_rate
            sample_rate = 1  # Use all frames for trigger detection (was 3, which caused double sampling)
            # If you want to reduce computation, set sample_rate > 1, but be aware this reduces temporal resolution
            sampled_indices = list(range(0, num_frames, sample_rate))
            sampled_frames = frames[sampled_indices]
            
            print(f"\nTrigger Detection:")
            print(f"  Total frames available: {num_frames}")
            print(f"  Sampling every {sample_rate} frame(s) for trigger detection")
            print(f"  Frames to analyze: {len(sampled_indices)}")
            if sample_rate > 1:
                print(f"  ⚠️  WARNING: Using sample_rate={sample_rate} means analyzing every {sample_rate} frames")
                print(f"     This reduces temporal resolution and may miss triggers between sampled frames")
            
            if timestamps is not None:
                sampled_timestamps = timestamps[sampled_indices]
            else:
                # Create timestamps tensor on the same device as frames
                sampled_timestamps = torch.tensor(
                    sampled_indices, 
                    dtype=torch.float32,
                    device=frames.device
                )
            
            # Add batch dimension
            sampled_frames = sampled_frames.unsqueeze(0)  # (1, T_sampled, C, H, W)
            
            # Stage 1: Extract features and detect triggers
            # YES, ResNet visual encoder IS being used here for feature extraction
            print(f"  Using ResNet visual encoder to extract features from {len(sampled_indices)} sampled frames...")
            visual_features = self.visual_encoder(sampled_frames)  # (1, T_sampled, feature_dim)
            print(f"  Extracted visual features shape: {visual_features.shape}")
            
            # YES, Trigger detector IS being used here to detect triggers
            print(f"  Running trigger detector on visual features...")
            trigger_probs = self.trigger_detector(visual_features)  # (1, T_sampled, num_classes)
            print(f"  Trigger probabilities shape: {trigger_probs.shape}")
            
            # Find triggered frames
            if trigger_probs.dim() == 3:
                if trigger_probs.shape[2] == 2:
                    # Binary classification
                    trigger_confidences = trigger_probs[0, :, 1]  # Probability of trigger
                else:
                    # Multi-class: use max
                    trigger_confidences = trigger_probs[0].max(dim=-1)[0]
            else:
                trigger_confidences = trigger_probs[0]
            
            threshold = trigger_threshold if trigger_threshold is not None else self.trigger_detector.threshold
            
            # Debug: Print trigger confidences
            confidences_list = trigger_confidences.cpu().tolist()
            print(f"\n  Trigger Detection Results (threshold={threshold:.3f}):")
            print(f"  Total sampled frames: {len(confidences_list)}")
            print(f"  Confidence range: [{min(confidences_list):.4f}, {max(confidences_list):.4f}]")
            print(f"  Mean confidence: {sum(confidences_list)/len(confidences_list):.4f}")
            
            # Show first few confidences with timestamps
            print(f"  First 10 frame confidences:")
            for i in range(min(10, len(confidences_list))):
                frame_idx = sampled_indices[i] if i < len(sampled_indices) else i
                timestamp = sampled_timestamps[i].item() if timestamps is not None else frame_idx / 3.0
                print(f"    Frame {frame_idx} (t={timestamp:.2f}s): confidence={confidences_list[i]:.4f}")
            
            triggered_mask = trigger_confidences > threshold
            
            # Convert to list for reliable iteration
            if isinstance(triggered_mask, torch.Tensor):
                triggered_mask = triggered_mask.cpu().tolist()
            
            # Process each triggered frame
            # Group consecutive triggers to avoid processing the same clip multiple times
            triggered_indices = [i for i, is_triggered in enumerate(triggered_mask) if is_triggered]
            
            if len(triggered_indices) == 0:
                print("No triggers detected above threshold")
                return results
            
            print(f"Found {len(triggered_indices)} triggered frames, processing...")
            
            # Sample triggers if max_frames is specified and we have too many
            if max_frames is not None and len(triggered_indices) > max_frames:
                print(f"  Sampling {max_frames} triggers evenly from {len(triggered_indices)} detected triggers")
                # Use linspace to sample evenly across all triggers
                sampled_positions = torch.linspace(0, len(triggered_indices) - 1, max_frames).long()
                triggered_indices = [triggered_indices[i] for i in sampled_positions]
                print(f"  Sampled indices: {triggered_indices[:5]}... (showing first 5)")
            
            # Process triggers, grouping nearby ones to avoid duplicate processing
            processed_clips = set()  # Track processed clip centers to avoid duplicates
            # Convert min_trigger_gap from sampled frame indices to original frame indices
            min_trigger_gap = 5 * sample_rate  # Minimum frames (in original frame space) between triggers to process separately
            
            for idx, i in enumerate(triggered_indices):
                original_idx = sampled_indices[i]
                timestamp = sampled_timestamps[i].item()
                confidence = trigger_confidences[i].item()
                
                # Skip if we've already processed a clip very close to this one
                skip = False
                for processed_idx in processed_clips:
                    if abs(original_idx - processed_idx) < min_trigger_gap:
                        skip = True
                        break
                
                if skip:
                    print(f"  Skipping trigger at frame {original_idx} (too close to previous trigger)")
                    continue
                
                try:
                    if self.use_llava:
                        # LLaVA needs higher resolution frames for better scene understanding
                        # Extract frames at higher resolution directly from video
                        if video_path is not None:
                            # Extract high-resolution frames directly from video
                            import cv2
                            from PIL import Image
                            
                            cap = cv2.VideoCapture(video_path)
                            if cap.isOpened():
                                video_fps = cap.get(cv2.CAP_PROP_FPS)
                                
                                # Calculate frame numbers for the clip
                                half_window = self.clip_window_size // 2
                                center_frame_num = int(timestamp * video_fps)
                                start_frame_num = max(0, center_frame_num - half_window)
                                
                                # Extract frames at original resolution, then resize to LLaVA optimal size
                                clip_frames_high_res = []
                                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
                                
                                for _ in range(self.clip_window_size):
                                    ret, frame = cap.read()
                                    if not ret:
                                        # Pad with last frame if video ends
                                        if clip_frames_high_res:
                                            # Use last extracted frame
                                            last_frame_array = clip_frames_high_res[-1]
                                            # Convert back to format that can be processed
                                            frame_rgb = last_frame_array
                                            img = Image.fromarray(frame_rgb)
                                            img_resized = img.resize(llava_image_size, Image.Resampling.LANCZOS)
                                            clip_frames_high_res.append(np.array(img_resized))
                                            continue
                                        else:
                                            # No frames extracted yet, break
                                            break
                                    
                                    # Convert BGR to RGB
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    
                                    # Resize to LLaVA optimal size (336x336) instead of 224x224
                                    img = Image.fromarray(frame_rgb)
                                    img_resized = img.resize(llava_image_size, Image.Resampling.LANCZOS)
                                    clip_frames_high_res.append(np.array(img_resized))
                                
                                cap.release()
                                
                                # Convert to tensor: (clip_len, H, W, C) -> (clip_len, C, H, W)
                                clip_frames_list = []
                                for frame_array in clip_frames_high_res:
                                    frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1).float() / 255.0
                                    clip_frames_list.append(frame_tensor)
                                
                                # Pad if necessary
                                while len(clip_frames_list) < self.clip_window_size:
                                    if clip_frames_list:
                                        clip_frames_list.append(clip_frames_list[-1].clone())
                                    else:
                                        # Create black frame as last resort
                                        clip_frames_list.append(torch.zeros(3, llava_image_size[0], llava_image_size[1]))
                                
                                clip_frames = torch.stack(clip_frames_list)  # (clip_len, C, H, W)
                                
                                print(f"  Extracted {len(clip_frames_high_res)} high-res frames ({llava_image_size[0]}x{llava_image_size[1]}) for LLaVA")
                            else:
                                # Fallback: use existing frames but upsample
                                print(f"  Warning: Could not open video {video_path}, using upsampled frames")
                                half_window = self.clip_window_size // 2
                                start_idx = max(0, original_idx - half_window)
                                end_idx = min(frames.shape[0], original_idx + half_window)
                                clip_frames = frames[start_idx:end_idx]  # (clip_len, C, H, W) - low res
                                
                                # Upsample to higher resolution for LLaVA
                                import torch.nn.functional as F
                                clip_frames = F.interpolate(
                                    clip_frames.unsqueeze(0), 
                                    size=llava_image_size, 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze(0)
                                
                                # Pad if necessary
                                if clip_frames.shape[0] < self.clip_window_size:
                                    padding = self.clip_window_size - clip_frames.shape[0]
                                    clip_frames = torch.cat([
                                        clip_frames,
                                        clip_frames[-1:].repeat(padding, 1, 1, 1)
                                    ], dim=0)
                        else:
                            # No video path provided - use existing frames but upsample
                            half_window = self.clip_window_size // 2
                            start_idx = max(0, original_idx - half_window)
                            end_idx = min(frames.shape[0], original_idx + half_window)
                            clip_frames = frames[start_idx:end_idx]  # (clip_len, C, H, W) - low res
                            
                            # Upsample to higher resolution for LLaVA
                            import torch.nn.functional as F
                            clip_frames = F.interpolate(
                                clip_frames.unsqueeze(0), 
                                size=llava_image_size, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze(0)
                            
                            # Pad if necessary
                            if clip_frames.shape[0] < self.clip_window_size:
                                padding = self.clip_window_size - clip_frames.shape[0]
                                clip_frames = torch.cat([
                                    clip_frames,
                                    clip_frames[-1:].repeat(padding, 1, 1, 1)
                                ], dim=0)
                        
                        # Add batch dimension
                        clip_frames = clip_frames.unsqueeze(0)  # (1, clip_len, C, H, W)
                        
                        # Generate analysis with LLaVA using configurable prompt
                        # If no prompt provided, temporal_llm will use its default
                        analysis_texts = self.temporal_llm.generate(
                            clip_frames,
                            prompt=llava_prompt,
                            max_new_tokens=256,
                            temperature=0.7,
                            do_sample=False
                        )
                        
                        # Extract the middle frame (trigger frame) for display
                        # clip_frames is (1, clip_len, C, H, W), get middle frame
                        middle_frame_idx = clip_frames.shape[1] // 2
                        trigger_frame = clip_frames[0, middle_frame_idx]  # (C, H, W)
                        
                        # Convert to PIL Image for display
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        # np is already imported at module level
                        
                        # Denormalize if needed and convert to uint8
                        frame_img = trigger_frame.clone().detach().cpu()
                        if frame_img.min() < 0 or frame_img.max() <= 1.0:
                            # Denormalize or scale to [0, 255]
                            if frame_img.min() < 0:
                                # Denormalize
                                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                                frame_img = frame_img * std + mean
                            frame_img = frame_img.clamp(0, 1) * 255.0
                        frame_img = frame_img.byte()
                        
                        # Convert CHW to HWC and to numpy
                        frame_img_np = frame_img.permute(1, 2, 0).numpy().astype(np.uint8)
                        pil_image = Image.fromarray(frame_img_np, mode='RGB')
                        
                        # Resize to reasonable display size (max 512px width)
                        max_display_width = 512
                        if pil_image.width > max_display_width:
                            aspect_ratio = pil_image.height / pil_image.width
                            new_height = int(max_display_width * aspect_ratio)
                            pil_image = pil_image.resize((max_display_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Convert to base64 for HTML embedding
                        buffered = BytesIO()
                        pil_image.save(buffered, format="JPEG", quality=85)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        frame_image_data = f"data:image/jpeg;base64,{img_base64}"
                    else:
                        # Traditional LLM uses encoded features
                        clip_features, clip_timestamps = self.extract_clip_features(
                            frames,
                            original_idx,
                            timestamps
                        )
                        
                        # Add batch dimension for LLM
                        clip_features = clip_features.unsqueeze(0)  # (1, clip_len, hidden_dim)
                        
                        # Generate analysis with configurable prompt
                        # If no prompt provided, temporal_llm will use its default
                        analysis_texts = self.temporal_llm.generate(
                            clip_features,
                            max_new_tokens=150,
                            temperature=0.8,
                            top_k=50,
                            top_p=0.95,
                            prompt=llm_prompt,
                            max_words=100,
                            repetition_penalty=1.3
                        )
                        
                        # Extract frame image from original frames for display
                        # Get the trigger frame from the original frames tensor
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        # np is already imported at module level
                        
                        try:
                            # Get frame at trigger index
                            if original_idx < frames.shape[0]:
                                trigger_frame = frames[original_idx]  # (C, H, W)
                                
                                # Denormalize if needed and convert to uint8
                                frame_img = trigger_frame.clone().detach().cpu()
                                if frame_img.min() < 0 or frame_img.max() <= 1.0:
                                    # Denormalize or scale to [0, 255]
                                    if frame_img.min() < 0:
                                        # Denormalize
                                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                                        frame_img = frame_img * std + mean
                                    frame_img = frame_img.clamp(0, 1) * 255.0
                                frame_img = frame_img.byte()
                                
                                # Convert CHW to HWC and to numpy
                                frame_img_np = frame_img.permute(1, 2, 0).numpy().astype(np.uint8)
                                pil_image = Image.fromarray(frame_img_np, mode='RGB')
                                
                                # Resize to reasonable display size (max 512px width)
                                max_display_width = 512
                                if pil_image.width > max_display_width:
                                    aspect_ratio = pil_image.height / pil_image.width
                                    new_height = int(max_display_width * aspect_ratio)
                                    pil_image = pil_image.resize((max_display_width, new_height), Image.Resampling.LANCZOS)
                                
                                # Convert to base64 for HTML embedding
                                buffered = BytesIO()
                                pil_image.save(buffered, format="JPEG", quality=85)
                                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                                frame_image_data = f"data:image/jpeg;base64,{img_base64}"
                            else:
                                frame_image_data = None
                        except Exception as e:
                            print(f"  Warning: Could not extract frame image: {e}")
                            frame_image_data = None
                    
                    # Get frame image (set in LLaVA branch or traditional LLM branch)
                    frame_image = frame_image_data if 'frame_image_data' in locals() else None
                    
                    results.append({
                        'timestamp': timestamp,
                        'frame_index': original_idx,
                        'trigger_confidence': confidence,
                        'analysis': analysis_texts[0] if analysis_texts else "",
                        'frame_image': frame_image
                    })
                    
                    # Clear frame_image_data for next iteration
                    if 'frame_image_data' in locals():
                        del frame_image_data
                    
                    processed_clips.add(original_idx)
                    print(f"  Processed trigger {len(results)}/{len(triggered_indices)} at frame {original_idx} (t={timestamp:.2f}s)")
                    
                except Exception as e:
                    print(f"  Warning: Failed to process trigger at frame {original_idx}: {e}")
                    # Continue processing other triggers even if one fails
                    import traceback
                    traceback.print_exc()
                    continue
        
        return results

