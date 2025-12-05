#!/usr/bin/env python3
"""
Simple script to generate annotations.json from video files in a directory.
Does not require cv2 - uses basic file operations.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def generate_annotations_from_videos(video_dir: str, output_path: str, 
                                    create_sample_triggers: bool = True,
                                    num_triggers_per_video: int = 2,
                                    default_duration: float = 30.0) -> None:
    """
    Generate annotations.json from video files in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_path: Path to save annotations.json
        create_sample_triggers: If True, create sample triggers at regular intervals
        num_triggers_per_video: Number of sample triggers to create per video
        default_duration: Default duration to use if we can't determine actual duration
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
        
        # Use default duration (you can improve this by using ffprobe or cv2 if available)
        duration = default_duration
        
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
    parser.add_argument('--default-duration', type=float, default=30.0,
                       help='Default video duration in seconds if unknown (default: 30.0)')
    
    args = parser.parse_args()
    
    generate_annotations_from_videos(
        args.video_dir,
        args.output,
        create_sample_triggers=not args.no_sample_triggers,
        num_triggers_per_video=args.num_triggers,
        default_duration=args.default_duration
    )


if __name__ == '__main__':
    main()

