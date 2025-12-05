#!/usr/bin/env python3
"""
Generate annotations.json from existing video files.
Creates placeholder annotations for videos that don't have annotations yet.
"""

import os
import json
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        
        cap.release()
        return duration
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {e}")
        return 0.0


def generate_annotations_from_videos(video_dir: str, output_path: str, 
                                    create_sample_triggers: bool = True,
                                    num_triggers_per_video: int = 2) -> None:
    """
    Generate annotations.json from video files in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_path: Path to save annotations.json
        create_sample_triggers: If True, create sample triggers at regular intervals
        num_triggers_per_video: Number of sample triggers to create per video
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    annotations = {'videos': []}
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = str(video_file)
        relative_path = video_file.name  # Just the filename
        
        # Get video duration
        duration = get_video_duration(video_path)
        
        if duration <= 0:
            print(f"Warning: Could not get duration for {video_file.name}, skipping")
            continue
        
        # Create triggers
        triggers = []
        if create_sample_triggers and duration > 0:
            # Create triggers at regular intervals
            interval = duration / (num_triggers_per_video + 1)
            for i in range(1, num_triggers_per_video + 1):
                timestamp = interval * i
                triggers.append({
                    'timestamp': round(timestamp, 2),
                    'label': 1,
                    'description': f'Sample trigger at {round(timestamp, 2)}s'
                })
        
        annotations['videos'].append({
            'video_path': relative_path,
            'triggers': triggers,
            'description': f'Video: {video_file.stem}'
        })
    
    # Save annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated annotations for {len(annotations['videos'])} videos")
    total_triggers = sum(len(v['triggers']) for v in annotations['videos'])
    print(f"Total triggers: {total_triggers}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate annotations.json from existing video files'
    )
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output annotations.json file path')
    parser.add_argument('--no-sample-triggers', action='store_true',
                       help='Do not create sample triggers (videos will have no triggers)')
    parser.add_argument('--num-triggers', type=int, default=2,
                       help='Number of sample triggers per video (default: 2)')
    
    args = parser.parse_args()
    
    generate_annotations_from_videos(
        args.video_dir,
        args.output,
        create_sample_triggers=not args.no_sample_triggers,
        num_triggers_per_video=args.num_triggers
    )


if __name__ == '__main__':
    main()

