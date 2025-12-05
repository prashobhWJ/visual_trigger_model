"""
Stage 2: Time-aware Encoder
Encodes selected frames with temporal awareness using timestamps and spatiotemporal context
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for timestamps"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, d_model)
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        max_len = self.pe.size(1)
        
        # Handle sequences longer than max_len by extending positional encoding
        if seq_len > max_len:
            # Extend positional encoding by repeating the last position
            pe = self.pe[:, :max_len, :]  # Get all available positions
            last_pe = self.pe[:, -1:, :]  # (1, 1, d_model) - last position
            repeat_count = seq_len - max_len
            extended_pe = last_pe.repeat(1, repeat_count, 1)
            pe = torch.cat([pe, extended_pe], dim=1)
        else:
            pe = self.pe[:, :seq_len, :]
        
        x = x + pe
        return x


class TimeAwareEncoder(nn.Module):
    """
    Encodes video frames with temporal awareness.
    Integrates timestamps and local spatiotemporal context.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        use_timestamps: bool = True,
        encoder_type: str = "transformer",
        dropout: float = 0.1,
        max_seq_length: int = 500
    ):
        """
        Args:
            input_dim: Dimension of input frame features
            hidden_dim: Hidden dimension for encoder
            num_layers: Number of encoder layers
            num_heads: Number of attention heads (for transformer)
            use_timestamps: Whether to incorporate timestamp information
            encoder_type: Type of encoder ('transformer' or 'lstm')
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_timestamps = use_timestamps
        self.encoder_type = encoder_type
        
        # Project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Timestamp encoding
        if use_timestamps:
            # Learnable timestamp embedding
            self.timestamp_embedding = nn.Linear(1, hidden_dim)
            # Or use positional encoding (use larger default to handle variable sequence lengths)
            # max_seq_length is a hint, but PositionalEncoding can handle longer sequences
            self.pos_encoding = PositionalEncoding(hidden_dim, max(max_seq_length, 500))
        
        if encoder_type == "transformer":
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
        elif encoder_type == "lstm":
            # Bidirectional LSTM encoder
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim // 2,  # Will be bidirectional
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        frame_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            frame_features: Tensor of shape (B, T, input_dim) - features for selected frames
            timestamps: Optional tensor of shape (B, T) - timestamps for each frame
            mask: Optional attention mask of shape (B, T) - 1 for valid, 0 for padding
        Returns:
            encoded_features: Tensor of shape (B, T, hidden_dim) with temporal awareness
        """
        batch_size, seq_len, _ = frame_features.shape
        
        # Project to hidden dimension
        x = self.input_projection(frame_features)  # (B, T, hidden_dim)
        
        # Incorporate timestamp information
        if self.use_timestamps:
            if timestamps is not None:
                # Normalize timestamps (assuming they're in seconds)
                # Add learnable timestamp embedding
                timestamp_emb = self.timestamp_embedding(timestamps.unsqueeze(-1))  # (B, T, hidden_dim)
                x = x + timestamp_emb
            
            # Add positional encoding
            x = self.pos_encoding(x)
        
        x = self.dropout(x)
        x = self.layer_norm(x)
        
        # Encode with transformer or LSTM
        if self.encoder_type == "transformer":
            # Create attention mask if provided
            if mask is not None:
                # Convert mask to attention mask (0 for valid, -inf for padding)
                attn_mask = mask == 0
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
                attn_mask = attn_mask.expand(batch_size, 1, seq_len, seq_len)
                attn_mask = attn_mask.reshape(batch_size * 1, seq_len, seq_len)
                attn_mask = attn_mask.masked_fill(attn_mask, float('-inf'))
            else:
                attn_mask = None
            
            encoded = self.encoder(x, src_key_padding_mask=mask if mask is not None else None)
            
        else:  # LSTM
            encoded, _ = self.encoder(x)
        
        return encoded
    
    def encode_clip(
        self,
        frame_features: torch.Tensor,
        timestamps: torch.Tensor,
        center_timestamp: float
    ) -> torch.Tensor:
        """
        Encode a clip of frames around a trigger timestamp.
        Args:
            frame_features: Tensor of shape (T, input_dim) - features for clip frames
            timestamps: Tensor of shape (T,) - timestamps for each frame
            center_timestamp: Center timestamp of the trigger
        Returns:
            encoded_features: Tensor of shape (T, hidden_dim)
        """
        # Normalize timestamps relative to center
        relative_timestamps = timestamps - center_timestamp
        
        # Add batch dimension
        frame_features = frame_features.unsqueeze(0)  # (1, T, input_dim)
        relative_timestamps = relative_timestamps.unsqueeze(0)  # (1, T)
        
        encoded = self.forward(frame_features, relative_timestamps)
        
        # Remove batch dimension
        return encoded.squeeze(0)  # (T, hidden_dim)

