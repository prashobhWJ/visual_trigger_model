"""
Stage 1: Lightweight Visual Encoder
Processes video frames at modest frame rate (2-5 FPS) to extract features
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from typing import Tuple, Optional


class VisualEncoder(nn.Module):
    """
    Lightweight visual encoder for processing video frames.
    Uses CNN backbone (ResNet, EfficientNet) to extract frame features.
    """
    
    def __init__(
        self,
        encoder_type: str = "resnet50",
        pretrained: bool = True,
        feature_dim: int = 512,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            encoder_type: Type of encoder backbone ('resnet18', 'resnet50', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension
            input_size: Input image size (H, W)
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        # Load backbone
        if encoder_type.startswith("resnet"):
            if encoder_type == "resnet18":
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet18(weights=weights)
                backbone_dim = 512
            elif encoder_type == "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                backbone = models.resnet50(weights=weights)
                backbone_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet type: {encoder_type}")
            
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif encoder_type == "efficientnet_b0":
            from torchvision.models import efficientnet_b0
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            backbone_dim = 1280
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # Projection head to desired feature dimension
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Track if backbone is frozen
        self._backbone_frozen = False
        self._projection_frozen = False
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape (B, C, H, W) or (B, T, C, H, W) for video clips
        Returns:
            features: Tensor of shape (B, feature_dim) or (B, T, feature_dim)
        """
        # Handle both single frames and video clips
        if frames.dim() == 4:
            # Single frame: (B, C, H, W)
            batch_size = frames.size(0)
            features = self.backbone(frames)
            features = self.projection(features)
            return features
        elif frames.dim() == 5:
            # Video clip: (B, T, C, H, W)
            batch_size, num_frames = frames.size(0), frames.size(1)
            frames = frames.view(batch_size * num_frames, *frames.shape[2:])
            features = self.backbone(frames)
            features = self.projection(features)
            features = features.view(batch_size, num_frames, self.feature_dim)
            return features
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {frames.dim()}D")
    
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame (convenience method for inference).
        Args:
            frame: Tensor of shape (C, H, W)
        Returns:
            features: Tensor of shape (feature_dim,)
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)  # Add batch dimension
        features = self.forward(frame)
        return features.squeeze(0)  # Remove batch dimension
    
    def freeze_backbone(self):
        """Freeze the pretrained backbone, keep projection trainable"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
    
    def freeze_projection(self):
        """Freeze the projection head"""
        for param in self.projection.parameters():
            param.requires_grad = False
        self._projection_frozen = True
    
    def unfreeze_projection(self):
        """Unfreeze the projection head"""
        for param in self.projection.parameters():
            param.requires_grad = True
        self._projection_frozen = False
    
    def freeze_all(self):
        """Freeze both backbone and projection"""
        self.freeze_backbone()
        self.freeze_projection()
    
    def unfreeze_all(self):
        """Unfreeze both backbone and projection"""
        self.unfreeze_backbone()
        self.unfreeze_projection()

