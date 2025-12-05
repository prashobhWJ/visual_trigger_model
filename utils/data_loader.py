"""
Data loading utilities for video datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import fiftyone as fo

from .video_utils import VideoProcessor, FrameSampler


def is_english_text(text: str, min_english_ratio: float = 0.7) -> bool:
    """
    Check if text is primarily English.
    
    Args:
        text: Text to check
        min_english_ratio: Minimum ratio of ASCII/English characters (default: 0.7)
    
    Returns:
        True if text appears to be English, False otherwise
    """
    if not text or not text.strip():
        return True  # Empty text is acceptable
    
    # Count ASCII characters (basic English letters, numbers, punctuation)
    ascii_count = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text.replace(' ', ''))  # Exclude spaces from count
    
    if total_chars == 0:
        return True
    
    english_ratio = ascii_count / total_chars
    return english_ratio >= min_english_ratio


def clean_description(description: str) -> str:
    """
    Clean and validate description text.
    Removes non-English characters and normalizes whitespace.
    
    Args:
        description: Raw description text
    
    Returns:
        Cleaned description or empty string if not English
    """
    if not description:
        return ''
    
    # Remove non-ASCII characters (keep only English)
    cleaned = ''.join(c if ord(c) < 128 else ' ' for c in description)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned


class VideoDataset(Dataset):
    """
    Dataset for video trigger detection and analysis.
    Expects annotations in JSON format:
    {
        "videos": [
            {
                "video_path": "path/to/video.mp4",
                "triggers": [
                    {"timestamp": 5.2, "label": 1, "description": "..."},
                    ...
                ],
                "description": "Overall video description"
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        video_dir: str,
        annotations_path: str,
        frame_sampling_rate: int = 3,
        clip_window_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: Optional[int] = None
    ):
        """
        Args:
            video_dir: Directory containing video files
            annotations_path: Path to JSON annotations file
            frame_sampling_rate: FPS for frame sampling
            clip_window_size: Number of frames in clip around trigger
            image_size: Target image size
            max_frames: Maximum frames per video
        """
        self.video_dir = video_dir
        self.frame_sampling_rate = frame_sampling_rate
        self.clip_window_size = clip_window_size
        self.max_frames = max_frames
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Initialize processors
        self.frame_sampler = FrameSampler(fps=frame_sampling_rate)
        self.video_processor = VideoProcessor(image_size=image_size)
        
        # Build dataset index
        self.samples = []
        for video_info in self.annotations.get('videos', []):
            video_path = os.path.join(video_dir, video_info['video_path'])
            if not os.path.exists(video_path):
                continue
            
            triggers = video_info.get('triggers', [])
            description = video_info.get('description', '')
            
            # Create samples for each trigger
            for trigger in triggers:
                self.samples.append({
                    'video_path': video_path,
                    'trigger_timestamp': trigger['timestamp'],
                    'trigger_label': trigger.get('label', 1),
                    'trigger_description': trigger.get('description', ''),
                    'video_description': description
                })
            
            # Also create negative samples (no trigger)
            if not triggers:
                self.samples.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': description
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'frames': (T, C, H, W) - sampled frames
                - 'timestamps': (T,) - frame timestamps
                - 'trigger_label': scalar - trigger label
                - 'trigger_timestamp': scalar - trigger timestamp (if exists)
                - 'description': str - ground truth description
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        # Sample frames from video
        frames, timestamps = self.frame_sampler.sample_frames(
            video_path,
            max_frames=self.max_frames
        )
        
        # Process frames to tensors
        frame_tensors = self.video_processor.process_frames(frames)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # Get trigger information
        trigger_label = sample['trigger_label']
        trigger_timestamp = sample.get('trigger_timestamp')
        
        # Create trigger mask (1 if frame is near trigger, 0 otherwise)
        trigger_mask = torch.zeros(len(timestamps), dtype=torch.long)
        if trigger_timestamp is not None:
            # Mark frames within 1 second of trigger
            time_diff = torch.abs(timestamps_tensor - trigger_timestamp)
            trigger_mask[time_diff < 1.0] = 1
        
        # Get ground truth description
        description = sample.get('trigger_description', '') or sample.get('video_description', '')
        
        return {
            'frames': frame_tensors,  # (T, C, H, W)
            'timestamps': timestamps_tensor,  # (T,)
            'trigger_label': torch.tensor(trigger_label, dtype=torch.long),
            'trigger_mask': trigger_mask,  # (T,)
            'trigger_timestamp': torch.tensor(trigger_timestamp if trigger_timestamp else -1.0, dtype=torch.float32),
            'description': description,
            'video_path': video_path
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching variable-length video sequences.
    Pads sequences to the same length.
    """
    # Find maximum sequence length
    max_len = max(item['frames'].shape[0] for item in batch)
    batch_size = len(batch)
    
    # Get dimensions
    C, H, W = batch[0]['frames'].shape[1:]
    
    # Initialize padded tensors
    padded_frames = torch.zeros(batch_size, max_len, C, H, W)
    padded_timestamps = torch.zeros(batch_size, max_len)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_trigger_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    trigger_labels = []
    trigger_timestamps = []
    descriptions = []
    video_paths = []
    
    for i, item in enumerate(batch):
        seq_len = item['frames'].shape[0]
        
        # Pad frames
        padded_frames[i, :seq_len] = item['frames']
        padded_timestamps[i, :seq_len] = item['timestamps']
        attention_mask[i, :seq_len] = True
        padded_trigger_mask[i, :seq_len] = item['trigger_mask']
        
        trigger_labels.append(item['trigger_label'])
        trigger_timestamps.append(item['trigger_timestamp'])
        descriptions.append(item['description'])
        video_paths.append(item['video_path'])
    
        return {
            'frames': padded_frames,
            'timestamps': padded_timestamps,
            'attention_mask': attention_mask,
            'trigger_mask': padded_trigger_mask,
            'trigger_labels': torch.stack(trigger_labels),
            'trigger_timestamps': torch.stack(trigger_timestamps),
            'descriptions': descriptions,
            'video_paths': video_paths
        }


def create_train_val_split(
    all_samples: List[Dict],
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train and validation sets.
    
    Args:
        all_samples: List of all samples to split
        val_ratio: Ratio of samples to use for validation (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_samples, val_samples)
    """
    if len(all_samples) == 0:
        return [], []
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle samples
    shuffled_samples = all_samples.copy()
    random.shuffle(shuffled_samples)
    
    # Calculate split point
    val_size = int(len(shuffled_samples) * val_ratio)
    val_samples = shuffled_samples[:val_size]
    train_samples = shuffled_samples[val_size:]
    
    return train_samples, val_samples


class ActivityNetDataset(Dataset):
    """
    Dataset for ActivityNet-200.
    Supports loading from FiftyOne (if available) or directly from JSON annotations.
    Can automatically create train/val splits when validation set is empty.
    """
    
    def __init__(
        self,
        split: str = "validation",
        frame_sampling_rate: int = 3,
        clip_window_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: Optional[int] = None,
        dataset_name: Optional[str] = None,
        video_dir: Optional[str] = None,
        annotations_path: Optional[str] = None,
        use_fiftyone: bool = True,
        auto_split: bool = True,
        val_ratio: float = 0.2,
        split_seed: int = 42,
        train_annotations_path: Optional[str] = None,
        train_video_dir: Optional[str] = None
    ):
        """
        Args:
            split: Dataset split ("train" or "validation")
            frame_sampling_rate: FPS for frame sampling
            clip_window_size: Number of frames in clip around trigger
            image_size: Target image size
            max_frames: Maximum frames per video
            dataset_name: Optional custom name for the FiftyOne dataset
            video_dir: Directory containing video files (for direct JSON loading)
            annotations_path: Path to ActivityNet JSON annotations file (for direct JSON loading)
            use_fiftyone: Whether to try using FiftyOne first (default: True)
            auto_split: If True and validation set is empty, automatically split from training set
            val_ratio: Ratio for validation split when auto_split is True (default: 0.2)
            split_seed: Random seed for reproducible train/val splits (default: 42)
            train_annotations_path: Path to training annotations (for auto_split)
            train_video_dir: Path to training videos (for auto_split)
        """
        self.split = split
        self.frame_sampling_rate = frame_sampling_rate
        self.clip_window_size = clip_window_size
        self.max_frames = max_frames
        
        # Initialize processors
        self.frame_sampler = FrameSampler(fps=frame_sampling_rate)
        self.video_processor = VideoProcessor(image_size=image_size)
        
        # Try to load from FiftyOne first, fall back to direct JSON loading
        self.samples = []
        loaded_from_fiftyone = False
        
        if use_fiftyone:
            try:
                self._load_from_fiftyone(dataset_name)
                loaded_from_fiftyone = True
                print(f"Loaded {len(self.samples)} samples from FiftyOne ActivityNet-200 {self.split} split")
                # Validate and clean descriptions after loading from FiftyOne
                self._validate_and_clean_descriptions()
            except Exception as e:
                print(f"Warning: Failed to load from FiftyOne: {e}")
                print("Falling back to direct JSON loading...")
        
        # If FiftyOne failed or not requested, load from JSON
        if not loaded_from_fiftyone:
            if annotations_path and video_dir:
                self._load_from_json(annotations_path, video_dir)
                
                # Validate and clean English descriptions after loading
                self._validate_and_clean_descriptions()
            else:
                # Try to infer paths from split
                if split == "train":
                    default_video_dir = "data/train/videos"
                    default_annotations = "data/train/annotations.json"
                else:
                    default_video_dir = "data/val/videos"
                    default_annotations = "data/val/annotations.json"
                
                video_dir = video_dir or default_video_dir
                annotations_path = annotations_path or default_annotations
                
                # Check if validation set is empty and auto_split is enabled
                val_exists = os.path.exists(annotations_path) if split == "validation" else False
                val_has_data = False
                
                if val_exists and split == "validation":
                    try:
                        with open(annotations_path, 'r') as f:
                            val_data = json.load(f)
                            val_has_data = len(val_data.get('videos', [])) > 0
                    except:
                        val_has_data = False
                
                # If validation is empty and auto_split is enabled, create split from training data
                if split == "validation" and auto_split and not val_has_data:
                    train_video_dir = train_video_dir or "data/train/videos"
                    train_annotations_path = train_annotations_path or "data/train/annotations.json"
                    
                    if os.path.exists(train_annotations_path):
                        print(f"Validation set is empty. Creating validation split from training data...")
                        all_samples = []
                        self._load_all_samples_from_json(train_annotations_path, train_video_dir, all_samples)
                        
                        if len(all_samples) > 0:
                            # Validate and clean descriptions before splitting
                            for sample in all_samples:
                                trigger_desc = sample.get('trigger_description', '')
                                video_desc = sample.get('video_description', '')
                                if trigger_desc and not is_english_text(trigger_desc):
                                    sample['trigger_description'] = clean_description(trigger_desc) or 'activity'
                                elif trigger_desc:
                                    sample['trigger_description'] = clean_description(trigger_desc)
                                if video_desc and not is_english_text(video_desc):
                                    sample['video_description'] = clean_description(video_desc) or 'activity'
                                elif video_desc:
                                    sample['video_description'] = clean_description(video_desc)
                            
                            train_samples, val_samples = create_train_val_split(all_samples, val_ratio, random_seed=split_seed)
                            self.samples = val_samples
                            print(f"Created validation set with {len(self.samples)} samples from {len(all_samples)} total samples")
                        else:
                            raise ValueError("No samples found in training data for splitting")
                    else:
                        raise ValueError(
                            f"Cannot create validation split. Training annotations not found at: {train_annotations_path}"
                        )
                elif split == "train" and auto_split:
                    # For training, check if validation is empty - if so, we need to exclude validation portion
                    val_annotations_path_check = annotations_path.replace("train", "val") if "train" in annotations_path else "data/val/annotations.json"
                    val_exists_check = os.path.exists(val_annotations_path_check)
                    val_has_data_check = False
                    
                    if val_exists_check:
                        try:
                            with open(val_annotations_path_check, 'r') as f:
                                val_data_check = json.load(f)
                                val_has_data_check = len(val_data_check.get('videos', [])) > 0
                        except:
                            val_has_data_check = False
                    
                    # If validation is empty, we need to split and only use training portion
                    if not val_has_data_check:
                        print(f"Validation set is empty. Creating train split (excluding validation portion)...")
                        all_samples = []
                        self._load_all_samples_from_json(annotations_path, video_dir, all_samples)
                        
                        # Validate and clean descriptions before splitting
                        for sample in all_samples:
                            trigger_desc = sample.get('trigger_description', '')
                            video_desc = sample.get('video_description', '')
                            if trigger_desc and not is_english_text(trigger_desc):
                                sample['trigger_description'] = clean_description(trigger_desc) or 'activity'
                            elif trigger_desc:
                                sample['trigger_description'] = clean_description(trigger_desc)
                            if video_desc and not is_english_text(video_desc):
                                sample['video_description'] = clean_description(video_desc) or 'activity'
                            elif video_desc:
                                sample['video_description'] = clean_description(video_desc)
                        
                        if len(all_samples) > 0:
                            train_samples, val_samples = create_train_val_split(all_samples, val_ratio, random_seed=split_seed)
                            self.samples = train_samples  # Use only training portion
                            print(f"Created training set with {len(self.samples)} samples from {len(all_samples)} total samples")
                        else:
                            # Fallback to normal loading if splitting fails
                            if os.path.exists(annotations_path) and os.path.exists(video_dir):
                                self._load_from_json(annotations_path, video_dir)
                            else:
                                raise ValueError(f"Cannot load training dataset")
                    else:
                        # Validation exists, load training normally
                        if os.path.exists(annotations_path) and os.path.exists(video_dir):
                            self._load_from_json(annotations_path, video_dir)
                            self._validate_and_clean_descriptions()
                        else:
                            raise ValueError(f"Cannot load training dataset")
                elif split == "validation" and val_has_data:
                    # Validation set exists and has data, load normally
                    if os.path.exists(annotations_path) and os.path.exists(video_dir):
                        self._load_from_json(annotations_path, video_dir)
                        self._validate_and_clean_descriptions()
                    else:
                        raise ValueError(
                            f"Cannot load validation dataset. Files not found at:\n"
                            f"  annotations: {annotations_path}\n"
                            f"  video_dir: {video_dir}"
                        )
                else:
                    # Normal loading when auto_split is False or validation exists
                    if os.path.exists(annotations_path) and os.path.exists(video_dir):
                        self._load_from_json(annotations_path, video_dir)
                        self._validate_and_clean_descriptions()
                    else:
                        raise ValueError(
                            f"Cannot load ActivityNet dataset. "
                            f"FiftyOne failed and JSON files not found at:\n"
                            f"  annotations: {annotations_path}\n"
                            f"  video_dir: {video_dir}\n"
                            f"Please provide valid paths or install MongoDB for FiftyOne."
                        )
        
        # Validate and clean descriptions for all loaded samples
        if len(self.samples) > 0:
            self._validate_and_clean_descriptions()
    
    def _validate_and_clean_descriptions(self):
        """Validate and clean all descriptions to ensure English-only text."""
        english_count = 0
        non_english_count = 0
        
        for sample in self.samples:
            trigger_desc = sample.get('trigger_description', '')
            video_desc = sample.get('video_description', '')
            
            # Validate and clean trigger description
            if trigger_desc:
                if not is_english_text(trigger_desc):
                    cleaned = clean_description(trigger_desc)
                    if cleaned:
                        sample['trigger_description'] = cleaned
                        non_english_count += 1
                    else:
                        sample['trigger_description'] = 'activity'
                        non_english_count += 1
                else:
                    sample['trigger_description'] = clean_description(trigger_desc)
                    english_count += 1
            
            # Validate and clean video description
            if video_desc:
                if not is_english_text(video_desc):
                    cleaned = clean_description(video_desc)
                    if cleaned:
                        sample['video_description'] = cleaned
                        if not trigger_desc:  # Only count if trigger_desc wasn't already counted
                            non_english_count += 1
                    else:
                        sample['video_description'] = 'activity'
                        if not trigger_desc:
                            non_english_count += 1
                else:
                    sample['video_description'] = clean_description(video_desc)
                    if not trigger_desc:
                        english_count += 1
        
        if non_english_count > 0:
            print(f"⚠️  Language validation: Found {non_english_count} non-English descriptions, cleaned/filtered to English-only")
        if english_count > 0 or non_english_count == 0:
            print(f"✓ Using {len(self.samples)} samples with English-only descriptions")
    
    def _load_from_fiftyone(self, dataset_name: Optional[str]):
        """Load dataset from FiftyOne."""
        if dataset_name is None:
            dataset_name = f"activitynet-200-{self.split}"
        
        try:
            # Try to load existing dataset
            self.fo_dataset = fo.load_dataset(dataset_name)
        except:
            # If not found, load from zoo
            self.fo_dataset = fo.zoo.load_zoo_dataset(
                "activitynet-200",
                split=self.split,
                dataset_name=dataset_name
            )
        
        # Build dataset index from FiftyOne samples
        for sample in self.fo_dataset:
            video_path = sample.filepath
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # Get temporal segments (activities) from ActivityNet annotations
            segments = []
            
            # Try different possible field names for segments
            if hasattr(sample, 'segments') and sample.segments is not None:
                if hasattr(sample.segments, 'detections'):
                    segments = sample.segments.detections
                elif isinstance(sample.segments, list):
                    segments = sample.segments
            elif hasattr(sample, 'ground_truth') and sample.ground_truth is not None:
                if hasattr(sample.ground_truth, 'detections'):
                    segments = sample.ground_truth.detections
                elif isinstance(sample.ground_truth, list):
                    segments = sample.ground_truth
            elif hasattr(sample, 'activities') and sample.activities is not None:
                if hasattr(sample.activities, 'detections'):
                    segments = sample.activities.detections
                elif isinstance(sample.activities, list):
                    segments = sample.activities
            
            # Create samples for each segment (activity)
            if segments and len(segments) > 0:
                for segment in segments:
                    try:
                        # ActivityNet segments have start_time and end_time
                        start_time = getattr(segment, 'start_time', None)
                        end_time = getattr(segment, 'end_time', None)
                        
                        # Handle different segment formats
                        if start_time is None or end_time is None:
                            if hasattr(segment, 'get'):
                                start_time = segment.get('start_time', 0.0)
                                end_time = segment.get('end_time', start_time + 1.0)
                            else:
                                start_time = 0.0
                                end_time = 1.0
                        
                        trigger_timestamp = (start_time + end_time) / 2.0
                        
                        # Get activity label
                        label = getattr(segment, 'label', None)
                        if label is None:
                            label = segment.get('label', 'activity') if hasattr(segment, 'get') else 'activity'
                        
                        description = str(label)
                        
                        # Validate and clean English text
                        if not is_english_text(description):
                            description = 'activity'  # Fallback to generic label
                        else:
                            description = clean_description(description)
                        
                        self.samples.append({
                            'video_path': video_path,
                            'trigger_timestamp': trigger_timestamp,
                            'trigger_label': 1,
                            'trigger_description': description,
                            'video_description': description,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                    except Exception as e:
                        print(f"Warning: Error processing segment: {e}")
                        continue
            else:
                # Create negative sample (no activity segments)
                self.samples.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': '',
                    'start_time': None,
                    'end_time': None
                })
        
    def _load_from_json(self, annotations_path: str, video_dir: str):
        """Load dataset directly from ActivityNet JSON format or standard JSON format."""
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Check format: ActivityNet (dict with 'database' or list), or standard format (dict with 'videos')
        if isinstance(data, list):
            # ActivityNet Captions format: list of annotations
            print(f"Detected ActivityNet Captions format (list) - loading all videos and annotations...")
            self._load_activitynet_captions_format(data, video_dir)
            print(f"Loaded {len(self.samples)} samples from ActivityNet Captions dataset ({self.split} split)")
        elif isinstance(data, dict):
            if 'database' in data:
                # ActivityNet format: {"database": {"video_id": {"url": "...", "annotations": [...]}}}
                print(f"Detected ActivityNet format - loading all videos and annotations...")
                self._load_activitynet_format(data, video_dir)
                print(f"Loaded {len(self.samples)} samples from ActivityNet dataset ({self.split} split)")
            elif 'videos' in data:
                # Standard format: {"videos": [{"video_path": "...", "triggers": [...]}]}
                self._load_standard_format(data, video_dir)
                print(f"Loaded {len(self.samples)} samples from JSON ActivityNet-200 {self.split} split")
            else:
                raise ValueError(f"Unknown annotation format in {annotations_path}. Expected 'database' (ActivityNet), 'videos' (standard), or list (Captions format).")
        else:
            raise ValueError(f"Unknown annotation format in {annotations_path}. Expected dict or list.")
    
    def _load_all_samples_from_json(self, annotations_path: str, video_dir: str, samples_list: List[Dict]):
        """
        Load all samples from JSON into the provided list (for splitting purposes).
        This method populates samples_list instead of self.samples.
        """
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Check format: ActivityNet (dict with 'database' or list), or standard format (dict with 'videos')
        if isinstance(data, list):
            # ActivityNet Captions format: list of annotations
            self._load_activitynet_captions_format_to_list(data, video_dir, samples_list)
        elif isinstance(data, dict):
            if 'database' in data:
                # ActivityNet format: {"database": {"video_id": {"url": "...", "annotations": [...]}}}
                self._load_activitynet_format_to_list(data, video_dir, samples_list)
            elif 'videos' in data:
                # Standard format: {"videos": [{"video_path": "...", "triggers": [...]}]}
                self._load_standard_format_to_list(data, video_dir, samples_list)
            else:
                raise ValueError(f"Unknown annotation format in {annotations_path}")
        else:
            raise ValueError(f"Unknown annotation format in {annotations_path}")
    
    def _load_activitynet_captions_format_to_list(self, data: List, video_dir: str, samples_list: List[Dict]):
        """Load from ActivityNet Captions format (list of annotations) into provided list."""
        # Group annotations by video
        video_dict = {}
        for ann in data:
            video_path_str = ann.get('video', '')
            if not video_path_str:
                continue
            
            # Extract video ID from path (e.g., "video/v_ehGHCYKzyZ8.mp4" -> "v_ehGHCYKzyZ8")
            video_id = os.path.splitext(os.path.basename(video_path_str))[0]
            
            if video_id not in video_dict:
                video_dict[video_id] = {
                    'video_path_str': video_path_str,
                    'annotations': []
                }
            
            # Add annotation
            video_dict[video_id]['annotations'].append({
                'start_time': ann.get('start_time', 0.0),
                'end_time': ann.get('end_time', 0.0),
                'caption': ann.get('caption', 'activity')
            })
        
        # Process each video
        for video_id, video_info in video_dict.items():
            # Find video file
            video_path = None
            video_path_str = video_info['video_path_str']
            
            # Try multiple locations and extensions
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                # Try with video_id
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
                
                # Try with original path relative to video_dir
                candidate = os.path.join(video_dir, video_path_str)
                if os.path.exists(candidate):
                    video_path = candidate
                    break
                
                # Try just the filename
                filename = os.path.basename(video_path_str)
                candidate = os.path.join(video_dir, filename)
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                continue  # Skip videos not found
            
            # Create a sample for each annotation
            for ann in video_info['annotations']:
                start_time = float(ann['start_time'])
                end_time = float(ann['end_time'])
                trigger_timestamp = (start_time + end_time) / 2.0
                description = ann['caption']
                
                # Validate and clean English text
                if not is_english_text(description):
                    description = 'activity'  # Fallback
                else:
                    description = clean_description(description)
                
                samples_list.append({
                    'video_path': video_path,
                    'trigger_timestamp': trigger_timestamp,
                    'trigger_label': 1,
                    'trigger_description': description,
                    'video_description': description,
                    'start_time': start_time,
                    'end_time': end_time
                })
    
    def _load_activitynet_captions_format(self, data: List, video_dir: str):
        """Load from ActivityNet Captions format (list of annotations)."""
        # Group annotations by video
        video_dict = {}
        for ann in data:
            video_path_str = ann.get('video', '')
            if not video_path_str:
                continue
            
            # Extract video ID from path
            video_id = os.path.splitext(os.path.basename(video_path_str))[0]
            
            if video_id not in video_dict:
                video_dict[video_id] = {
                    'video_path_str': video_path_str,
                    'annotations': []
                }
            
            video_dict[video_id]['annotations'].append({
                'start_time': ann.get('start_time', 0.0),
                'end_time': ann.get('end_time', 0.0),
                'caption': ann.get('caption', 'activity')
            })
        
        total_videos = len(video_dict)
        videos_with_annotations = 0
        total_annotations = 0
        videos_not_found = 0
        
        print(f"Processing ActivityNet Captions dataset: {total_videos} videos found in annotations...")
        
        # Process each video
        for video_id, video_info in video_dict.items():
            # Find video file
            video_path = None
            video_path_str = video_info['video_path_str']
            
            # Try multiple locations and extensions
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
                
                candidate = os.path.join(video_dir, video_path_str)
                if os.path.exists(candidate):
                    video_path = candidate
                    break
                
                filename = os.path.basename(video_path_str)
                candidate = os.path.join(video_dir, filename)
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                videos_not_found += 1
                if videos_not_found <= 5:
                    print(f"Warning: Video not found for {video_id}")
                continue
            
            # Create a sample for each annotation
            annotations = video_info['annotations']
            if annotations:
                videos_with_annotations += 1
                total_annotations += len(annotations)
                
                for ann in annotations:
                    start_time = float(ann['start_time'])
                    end_time = float(ann['end_time'])
                    trigger_timestamp = (start_time + end_time) / 2.0
                    description = ann['caption']
                    
                    # Validate and clean English text
                    if not is_english_text(description):
                        description = 'activity'
                    else:
                        description = clean_description(description)
                    
                    self.samples.append({
                        'video_path': video_path,
                        'trigger_timestamp': trigger_timestamp,
                        'trigger_label': 1,
                        'trigger_description': description,
                        'video_description': description,
                        'start_time': start_time,
                        'end_time': end_time
                    })
        
        # Print summary
        print(f"ActivityNet Captions loading summary:")
        print(f"  Total videos in annotations: {total_videos}")
        print(f"  Videos found and loaded: {total_videos - videos_not_found}")
        print(f"  Videos with annotations: {videos_with_annotations} ({total_annotations} annotation segments)")
        if videos_not_found > 0:
            print(f"  Videos not found: {videos_not_found}")
        print(f"  Total samples created: {len(self.samples)}")
    
    def _load_activitynet_format_to_list(self, data: Dict, video_dir: str, samples_list: List[Dict]):
        """Load from ActivityNet format JSON into provided list - processes ALL videos and ALL annotations."""
        database = data.get('database', {})
        
        for video_id, video_info in database.items():
            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                # Try with video_id as filename (no extension)
                candidate = os.path.join(video_dir, video_id)
                if os.path.exists(candidate):
                    video_path = candidate
                else:
                    # Try with video_id as directory name
                    candidate = os.path.join(video_dir, video_id, f"{video_id}.mp4")
                    if os.path.exists(candidate):
                        video_path = candidate
            
            if not video_path:
                continue  # Skip videos not found
            
            # Process annotations (ActivityNet uses segments)
            annotations = video_info.get('annotations', [])
            
            if annotations:
                # Create a sample for EACH annotation segment
                for ann in annotations:
                    segment = ann.get('segment', [0, 0])
                    start_time = float(segment[0])
                    end_time = float(segment[1])
                    trigger_timestamp = (start_time + end_time) / 2.0
                    
                    label = ann.get('label', ann.get('description', 'activity'))
                    description = str(label)
                    
                    # Validate and clean English text
                    if not is_english_text(description):
                        # Skip non-English descriptions or use fallback
                        description = 'activity'  # Fallback to generic label
                    else:
                        description = clean_description(description)
                    
                    samples_list.append({
                        'video_path': video_path,
                        'trigger_timestamp': trigger_timestamp,
                        'trigger_label': 1,
                        'trigger_description': description,
                        'video_description': description,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            else:
                # Create negative sample (no activity segments)
                samples_list.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': '',
                    'start_time': None,
                    'end_time': None
                })
    
    def _load_standard_format_to_list(self, data: Dict, video_dir: str, samples_list: List[Dict]):
        """Load from standard JSON format into provided list."""
        for video_info in data.get('videos', []):
            video_path = video_info.get('video_path', '')
            
            # Make absolute path
            if not os.path.isabs(video_path):
                video_path = os.path.join(video_dir, video_path)
            
            video_path = os.path.normpath(video_path)
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            triggers = video_info.get('triggers', [])
            description = video_info.get('description', '')
            
            # Validate and clean video description
            if description and not is_english_text(description):
                description = clean_description(description) or 'activity'
            
            if triggers:
                for trigger in triggers:
                    timestamp = trigger.get('timestamp', 0.0)
                    trigger_description = trigger.get('description', description)
                    
                    # Validate and clean trigger description
                    if trigger_description and not is_english_text(trigger_description):
                        trigger_description = clean_description(trigger_description) or description or 'activity'
                    elif trigger_description:
                        trigger_description = clean_description(trigger_description)
                    
                    # For standard format, we don't have start/end times, so use a window around timestamp
                    window_size = 2.0  # 2 seconds window
                    start_time = max(0.0, timestamp - window_size / 2)
                    end_time = timestamp + window_size / 2
                    
                    samples_list.append({
                        'video_path': video_path,
                        'trigger_timestamp': timestamp,
                        'trigger_label': trigger.get('label', 1),
                        'trigger_description': trigger_description,
                        'video_description': description,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            else:
                samples_list.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': description,
                    'start_time': None,
                    'end_time': None
                })
    
    def _load_activitynet_format(self, data: Dict, video_dir: str):
        """Load from ActivityNet format JSON - processes ALL videos and ALL annotations."""
        database = data.get('database', {})
        total_videos = len(database)
        videos_with_annotations = 0
        videos_without_annotations = 0
        total_annotations = 0
        videos_not_found = 0
        
        print(f"Processing ActivityNet dataset: {total_videos} videos found in annotations...")
        
        for video_id, video_info in database.items():
            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                candidate = os.path.join(video_dir, f"{video_id}{ext}")
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                # Try with video_id as filename (no extension)
                candidate = os.path.join(video_dir, video_id)
                if os.path.exists(candidate):
                    video_path = candidate
                else:
                    # Try with video_id as directory name
                    candidate = os.path.join(video_dir, video_id, f"{video_id}.mp4")
                    if os.path.exists(candidate):
                        video_path = candidate
            
            if not video_path:
                videos_not_found += 1
                if videos_not_found <= 5:  # Only print first 5 warnings
                    print(f"Warning: Video not found for {video_id}")
                continue
            
            # Process annotations (ActivityNet uses segments)
            annotations = video_info.get('annotations', [])
            
            if annotations:
                videos_with_annotations += 1
                total_annotations += len(annotations)
                
                # Create a sample for EACH annotation segment
                for ann in annotations:
                    # ActivityNet segments: [start_time, end_time]
                    segment = ann.get('segment', [0, 0])
                    start_time = float(segment[0])
                    end_time = float(segment[1])
                    trigger_timestamp = (start_time + end_time) / 2.0
                    
                    label = ann.get('label', ann.get('description', 'activity'))
                    description = str(label)
                    
                    # Validate and clean English text
                    if not is_english_text(description):
                        # Skip non-English descriptions or use fallback
                        description = 'activity'  # Fallback to generic label
                    else:
                        description = clean_description(description)
                    
                    self.samples.append({
                        'video_path': video_path,
                        'trigger_timestamp': trigger_timestamp,
                        'trigger_label': 1,
                        'trigger_description': description,
                        'video_description': description,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            else:
                # Create negative sample (no activity segments)
                videos_without_annotations += 1
                self.samples.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': '',
                    'start_time': None,
                    'end_time': None
                })
        
        # Print summary
        print(f"ActivityNet loading summary:")
        print(f"  Total videos in annotations: {total_videos}")
        print(f"  Videos found and loaded: {total_videos - videos_not_found}")
        print(f"  Videos with annotations: {videos_with_annotations} ({total_annotations} annotation segments)")
        print(f"  Videos without annotations (negative samples): {videos_without_annotations}")
        if videos_not_found > 0:
            print(f"  Videos not found: {videos_not_found}")
        print(f"  Total samples created: {len(self.samples)}")
    
    def _load_standard_format(self, data: Dict, video_dir: str):
        """Load from standard JSON format."""
        for video_info in data.get('videos', []):
            video_path = video_info.get('video_path', '')
            
            # Make absolute path
            if not os.path.isabs(video_path):
                video_path = os.path.join(video_dir, video_path)
            
            video_path = os.path.normpath(video_path)
            
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            triggers = video_info.get('triggers', [])
            description = video_info.get('description', '')
            
            # Validate and clean video description
            if description and not is_english_text(description):
                description = clean_description(description) or 'activity'
            
            if triggers:
                for trigger in triggers:
                    timestamp = trigger.get('timestamp', 0.0)
                    trigger_description = trigger.get('description', description)
                    
                    # Validate and clean trigger description
                    if trigger_description and not is_english_text(trigger_description):
                        trigger_description = clean_description(trigger_description) or description or 'activity'
                    elif trigger_description:
                        trigger_description = clean_description(trigger_description)
                    
                    # For standard format, we don't have start/end times, so use a window around timestamp
                    window_size = 2.0  # 2 seconds window
                    start_time = max(0.0, timestamp - window_size / 2)
                    end_time = timestamp + window_size / 2
                    
                    self.samples.append({
                        'video_path': video_path,
                        'trigger_timestamp': timestamp,
                        'trigger_label': trigger.get('label', 1),
                        'trigger_description': trigger_description,
                        'video_description': description,
                        'start_time': start_time,
                        'end_time': end_time
                    })
            else:
                # Create negative sample (no triggers)
                self.samples.append({
                    'video_path': video_path,
                    'trigger_timestamp': None,
                    'trigger_label': 0,
                    'trigger_description': '',
                    'video_description': description,
                    'start_time': None,
                    'end_time': None
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'frames': (T, C, H, W) - sampled frames
                - 'timestamps': (T,) - frame timestamps
                - 'trigger_label': scalar - trigger label
                - 'trigger_timestamp': scalar - trigger timestamp (if exists)
                - 'description': str - ground truth description
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        # Sample frames from video
        frames, timestamps = self.frame_sampler.sample_frames(
            video_path,
            max_frames=self.max_frames
        )
        
        # Process frames to tensors
        frame_tensors = self.video_processor.process_frames(frames)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # Get trigger information
        trigger_label = sample['trigger_label']
        trigger_timestamp = sample.get('trigger_timestamp')
        start_time = sample.get('start_time')
        end_time = sample.get('end_time')
        
        # Create trigger mask (1 if frame is within activity segment, 0 otherwise)
        trigger_mask = torch.zeros(len(timestamps), dtype=torch.long)
        if trigger_timestamp is not None and start_time is not None and end_time is not None:
            # Mark frames within the activity segment
            trigger_mask[(timestamps_tensor >= start_time) & (timestamps_tensor <= end_time)] = 1
        elif trigger_timestamp is not None:
            # Fallback: mark frames within 1 second of trigger
            time_diff = torch.abs(timestamps_tensor - trigger_timestamp)
            trigger_mask[time_diff < 1.0] = 1
        
        # Get ground truth description
        description = sample.get('trigger_description', '') or sample.get('video_description', '')
        
        # Ensure description is English-only
        if description and not is_english_text(description):
            description = clean_description(description)
            if not description:
                description = 'activity'  # Fallback
        
        return {
            'frames': frame_tensors,  # (T, C, H, W)
            'timestamps': timestamps_tensor,  # (T,)
            'trigger_label': torch.tensor(trigger_label, dtype=torch.long),
            'trigger_mask': trigger_mask,  # (T,)
            'trigger_timestamp': torch.tensor(trigger_timestamp if trigger_timestamp else -1.0, dtype=torch.float32),
            'description': description,
            'video_path': video_path
        }

