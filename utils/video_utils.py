"""
Video processing utilities for frame extraction and sampling
"""

import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class FrameSampler:
    """Samples frames from video at specified rate"""
    
    def __init__(self, fps: int = 3):
        """
        Args:
            fps: Frames per second to sample (2-5 FPS for lightweight analysis)
        """
        self.fps = fps
    
    def sample_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Sample frames from video.
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample
        Returns:
            frames: List of frame arrays
            timestamps: List of timestamps in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        # Calculate frame interval
        frame_interval = max(1, int(video_fps / self.fps))
        
        frames = []
        timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at specified rate
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamp = frame_count / video_fps
                timestamps.append(timestamp)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return frames, timestamps


class VideoProcessor:
    """Processes video frames for model input"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Args:
            image_size: Target image size (H, W)
            normalize: Whether to normalize to [0, 1] and apply ImageNet stats
        """
        self.image_size = image_size
        
        if normalize:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
    
    def process_frames(
        self,
        frames: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Process list of frames to tensor.
        Args:
            frames: List of frame arrays (H, W, C) in RGB
        Returns:
            Tensor of shape (T, C, H, W)
        """
        processed = []
        for frame in frames:
            tensor = self.transform(frame)
            processed.append(tensor)
        
        return torch.stack(processed)  # (T, C, H, W)
    
    def process_single_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a single frame.
        Args:
            frame: Frame array (H, W, C) in RGB
        Returns:
            Tensor of shape (C, H, W)
        """
        return self.transform(frame)
    
    def extract_clip_frames(
        self,
        video_path: str,
        center_timestamp: float,
        window_size: int = 16,
        fps: int = 30
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract frames around a specific timestamp.
        Args:
            video_path: Path to video file
            center_timestamp: Center timestamp in seconds
            window_size: Number of frames to extract
            fps: Video frame rate
        Returns:
            frames: List of frame arrays
            timestamps: List of timestamps
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        half_window = window_size // 2
        
        # Calculate frame indices
        center_frame_idx = int(center_timestamp * video_fps)
        start_frame_idx = max(0, center_frame_idx - half_window)
        end_frame_idx = start_frame_idx + window_size
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        frames = []
        timestamps = []
        
        for i in range(window_size):
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame if video ends
                if frames:
                    frame = frames[-1]
                else:
                    break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamp = (start_frame_idx + i) / video_fps
            timestamps.append(timestamp)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < window_size:
            if frames:
                frames.append(frames[-1])
                timestamps.append(timestamps[-1] + (1.0 / video_fps))
            else:
                # Create black frame if no frames extracted
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                timestamps.append(0.0)
        
        return frames, timestamps

