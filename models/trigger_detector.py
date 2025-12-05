"""
Stage 1: Trigger Detector
Detects events or triggers in video frames (e.g., object appearance, action start, scene change)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class TriggerDetector(nn.Module):
    """
    Detects triggers/events in video frames based on extracted features.
    Acts as a filter to decide when to involve a more complex LLM for deep analysis.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        threshold: float = 0.5,
        use_temporal_context: bool = True,
        temporal_window: int = 3
    ):
        """
        Args:
            input_dim: Dimension of input features from visual encoder
            hidden_dim: Hidden dimension for the detector network
            num_classes: Number of trigger classes (2 for binary, more for multi-class)
            threshold: Threshold for trigger detection
            use_temporal_context: Whether to use temporal context from previous frames
            temporal_window: Number of previous frames to consider for temporal context
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.threshold = threshold
        self.use_temporal_context = use_temporal_context
        self.temporal_window = temporal_window
        
        # Feature processing
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Temporal context processing (if enabled)
        if use_temporal_context:
            self.temporal_encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            lstm_output_dim = hidden_dim
        else:
            lstm_output_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # For binary classification, add sigmoid
        if num_classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=-1)
    
    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor of shape (B, T, input_dim) for video sequences
                     or (B, input_dim) for single frames
            return_logits: Whether to return raw logits in addition to probabilities
        Returns:
            probs: Trigger probabilities of shape (B, T, num_classes) or (B, num_classes)
            logits: Raw logits (if return_logits=True)
        """
        # Handle both single frames and sequences
        is_sequence = features.dim() == 3
        
        if not is_sequence:
            # Single frame: add temporal dimension
            features = features.unsqueeze(1)  # (B, 1, input_dim)
        
        batch_size, seq_len, _ = features.shape
        
        # Project features
        projected = self.feature_projection(features)  # (B, T, hidden_dim)
        
        # Apply temporal context if enabled
        if self.use_temporal_context and seq_len > 1:
            temporal_out, _ = self.temporal_encoder(projected)
            classifier_input = temporal_out
        else:
            classifier_input = projected
        
        # Classification
        logits = self.classifier(classifier_input)  # (B, T, num_classes)
        probs = self.activation(logits)
        # Clamp probabilities: values below 0.1 become 0.1, ensuring range [0.1, 1.0]
        # This helps with triggering even when computed values are very small
        probs = torch.clamp(probs, min=0.1, max=1.0)
        
        # Remove temporal dimension if input was single frame
        if not is_sequence:
            probs = probs.squeeze(1)
            logits = logits.squeeze(1)
        
        if return_logits:
            return probs, logits
        return probs
    
    def detect(
        self,
        features: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Detect if a trigger is present in the features.
        Args:
            features: Tensor of shape (input_dim,) or (B, input_dim)
            threshold: Detection threshold (uses self.threshold if None)
        Returns:
            is_trigger: Boolean indicating if trigger detected
            confidence: Confidence score
        """
        if threshold is None:
            threshold = self.threshold
        
        self.eval()
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            probs = self.forward(features)
            
            if self.num_classes == 2:
                # Binary classification: trigger probability is second class
                trigger_prob = probs[:, 1] if probs.dim() > 1 else probs[1]
            else:
                # Multi-class: use max probability
                trigger_prob = probs.max(dim=-1)[0]
            
            is_trigger = (trigger_prob > threshold).item() if trigger_prob.numel() == 1 else (trigger_prob > threshold)
            confidence = trigger_prob.item() if trigger_prob.numel() == 1 else trigger_prob
            
            return is_trigger, confidence

