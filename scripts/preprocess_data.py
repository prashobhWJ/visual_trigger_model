#!/usr/bin/env python3
"""
Preprocessing script to convert downloaded data files to annotations.json format.

Supports multiple input formats:
- CSV with video paths and trigger annotations
- JSON with video metadata
- ActivityNet format
- Custom annotation formats
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import cv2


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


def parse_timestamp(timestamp_str: str) -> float:
    """Parse timestamp string to float (seconds)."""
    try:
        # Handle various formats: "5.2", "00:05:20", "5m20s", etc.
        if ':' in timestamp_str:
            # Format: HH:MM:SS or MM:SS
            parts = timestamp_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        elif 'm' in timestamp_str.lower() and 's' in timestamp_str.lower():
            # Format: "5m20s"
            parts = timestamp_str.lower().replace('m', ' ').replace('s', '').split()
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(timestamp_str)
    except:
        pass
    
    return 0.0


def process_csv_annotations(csv_path: str, video_dir: str, 
                           video_path_column: str = 'video_path',
                           timestamp_column: str = 'timestamp',
                           label_column: str = 'label',
                           description_column: str = 'description',
                           video_description_column: Optional[str] = None) -> Dict:
    """Process CSV file with video annotations."""
    annotations = {'videos': []}
    video_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in tqdm(reader, desc="Processing CSV"):
            video_path = row.get(video_path_column)
            if not video_path:
                continue
            
            # Make path absolute if relative
            if not os.path.isabs(video_path):
                video_path = os.path.join(video_dir, video_path)
            
            # Normalize path
            video_path = os.path.normpath(video_path)
            
            # Initialize video entry if not exists
            if video_path not in video_dict:
                video_dict[video_path] = {
                    'video_path': os.path.relpath(video_path, video_dir) if video_dir else video_path,
                    'triggers': [],
                    'description': row.get(video_description_column or description_column, '')
                }
            
            # Parse trigger
            timestamp_str = row.get(timestamp_column, '0')
            timestamp = parse_timestamp(timestamp_str)
            
            label = row.get(label_column, '1')
            try:
                label = int(label)
            except:
                label = 1
            
            description = row.get(description_column, '')
            
            video_dict[video_path]['triggers'].append({
                'timestamp': timestamp,
                'label': label,
                'description': description
            })
    
    annotations['videos'] = list(video_dict.values())
    return annotations


def process_json_annotations(json_path: str, video_dir: str,
                            video_path_key: str = 'video_path') -> Dict:
    """Process JSON file with video annotations."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'videos' in data:
        videos = data['videos']
    elif isinstance(data, list):
        videos = data
    else:
        videos = [data]
    
    annotations = {'videos': []}
    
    for video_info in tqdm(videos, desc="Processing JSON"):
        video_path = video_info.get(video_path_key, video_info.get('path', ''))
        if not video_path:
            continue
        
        # Make path absolute if relative
        if not os.path.isabs(video_path):
            video_path = os.path.join(video_dir, video_path)
        
        video_path = os.path.normpath(video_path)
        
        # Process triggers
        triggers = []
        if 'triggers' in video_info:
            for trigger in video_info['triggers']:
                if isinstance(trigger, dict):
                    timestamp = trigger.get('timestamp', 0.0)
                    if isinstance(timestamp, str):
                        timestamp = parse_timestamp(timestamp)
                    
                    triggers.append({
                        'timestamp': float(timestamp),
                        'label': trigger.get('label', 1),
                        'description': trigger.get('description', '')
                    })
        
        annotations['videos'].append({
            'video_path': os.path.relpath(video_path, video_dir) if video_dir else video_path,
            'triggers': triggers,
            'description': video_info.get('description', '')
        })
    
    return annotations


def process_activitynet_format(annotations_path: str, video_dir: str) -> Dict:
    """Process ActivityNet format annotations."""
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    database = data.get('database', {})
    annotations = {'videos': []}
    
    for video_id, video_info in tqdm(database.items(), desc="Processing ActivityNet"):
        # Find video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            candidate = os.path.join(video_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break
        
        if not video_path:
            # Try with video_id as filename
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = os.path.join(video_dir, video_id)
                if os.path.exists(candidate):
                    video_path = candidate
                    break
        
        if not video_path:
            print(f"Warning: Video not found for {video_id}")
            continue
        
        # Process annotations (ActivityNet uses segments)
        triggers = []
        for ann in video_info.get('annotations', []):
            # ActivityNet segments: [start_time, end_time]
            segment = ann.get('segment', [0, 0])
            start_time = float(segment[0])
            end_time = float(segment[1])
            
            # Use start time as trigger timestamp
            triggers.append({
                'timestamp': start_time,
                'label': 1,
                'description': ann.get('label', ann.get('description', ''))
            })
        
        annotations['videos'].append({
            'video_path': os.path.relpath(video_path, video_dir) if video_dir else video_path,
            'triggers': triggers,
            'description': video_info.get('description', '')
        })
    
    return annotations


def process_download_metadata(metadata_path: str, video_dir: str,
                             annotation_source: Optional[str] = None) -> Dict:
    """Process metadata from download script output."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = data.get('videos', [])
    annotations = {'videos': []}
    
    for video_info in tqdm(videos, desc="Processing download metadata"):
        video_path = video_info.get('video_path', '')
        if not video_path or not os.path.exists(video_path):
            continue
        
        # Get triggers from metadata or annotation source
        triggers = []
        
        # Check if annotations are in metadata (ActivityNet format)
        if 'annotations' in video_info:
            for ann in video_info['annotations']:
                if isinstance(ann, dict):
                    segment = ann.get('segment', [0, 0])
                    start_time = float(segment[0])
                    triggers.append({
                        'timestamp': start_time,
                        'label': 1,
                        'description': ann.get('label', ann.get('description', ''))
                    })
        
        # Load from separate annotation file if provided
        if annotation_source and os.path.exists(annotation_source):
            # Try to match video_id with annotations
            video_id = video_info.get('video_id', '')
            # This would need custom logic based on annotation format
        
        annotations['videos'].append({
            'video_path': os.path.relpath(video_path, video_dir) if video_dir else video_path,
            'triggers': triggers,
            'description': video_info.get('metadata', {}).get('description', '')
        })
    
    return annotations


def validate_annotations(annotations: Dict, video_dir: str) -> Dict:
    """Validate and clean annotations."""
    validated = {'videos': []}
    
    for video_info in tqdm(annotations.get('videos', []), desc="Validating annotations"):
        video_path = video_info.get('video_path', '')
        
        # Make absolute path
        if not os.path.isabs(video_path):
            abs_path = os.path.join(video_dir, video_path)
        else:
            abs_path = video_path
        
        # Check if video exists
        if not os.path.exists(abs_path):
            print(f"Warning: Video not found: {abs_path}")
            continue
        
        # Get video duration
        duration = get_video_duration(abs_path)
        
        # Validate and filter triggers
        valid_triggers = []
        for trigger in video_info.get('triggers', []):
            timestamp = trigger.get('timestamp', 0.0)
            
            # Ensure timestamp is within video duration
            if timestamp < 0:
                timestamp = 0.0
            if duration > 0 and timestamp > duration:
                print(f"Warning: Trigger timestamp {timestamp}s exceeds video duration {duration}s")
                continue
            
            valid_triggers.append({
                'timestamp': float(timestamp),
                'label': int(trigger.get('label', 1)),
                'description': trigger.get('description', '').strip()
            })
        
        # Sort triggers by timestamp
        valid_triggers.sort(key=lambda x: x['timestamp'])
        
        validated['videos'].append({
            'video_path': os.path.relpath(abs_path, video_dir) if video_dir else abs_path,
            'triggers': valid_triggers,
            'description': video_info.get('description', '').strip()
        })
    
    return validated


def main():
    parser = argparse.ArgumentParser(description='Preprocess data to annotations.json format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file (CSV, JSON, or ActivityNet format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output annotations.json file path')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--format', type=str, 
                       choices=['csv', 'json', 'activitynet', 'download-metadata'],
                       default='json',
                       help='Input format type')
    parser.add_argument('--video-path-column', type=str, default='video_path',
                       help='Column/key name for video path (CSV/JSON)')
    parser.add_argument('--timestamp-column', type=str, default='timestamp',
                       help='Column name for timestamp (CSV)')
    parser.add_argument('--label-column', type=str, default='label',
                       help='Column name for label (CSV)')
    parser.add_argument('--description-column', type=str, default='description',
                       help='Column name for description (CSV)')
    parser.add_argument('--annotation-source', type=str, default=None,
                       help='Additional annotation file (for download-metadata format)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate video files and timestamps')
    
    args = parser.parse_args()
    
    print(f"Processing {args.input}...")
    print(f"Video directory: {args.video_dir}")
    print(f"Output: {args.output}")
    
    # Process based on format
    if args.format == 'csv':
        annotations = process_csv_annotations(
            args.input, args.video_dir,
            args.video_path_column, args.timestamp_column,
            args.label_column, args.description_column
        )
    elif args.format == 'json':
        annotations = process_json_annotations(
            args.input, args.video_dir, args.video_path_column
        )
    elif args.format == 'activitynet':
        annotations = process_activitynet_format(args.input, args.video_dir)
    elif args.format == 'download-metadata':
        annotations = process_download_metadata(
            args.input, args.video_dir, args.annotation_source
        )
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    # Validate if requested
    if args.validate:
        annotations = validate_annotations(annotations, args.video_dir)
    
    # Save annotations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully processed {len(annotations['videos'])} videos")
    total_triggers = sum(len(v['triggers']) for v in annotations['videos'])
    print(f"Total triggers: {total_triggers}")
    print(f"Annotations saved to {args.output}")


if __name__ == '__main__':
    main()

