#!/usr/bin/env python3
"""
Download script for video data that can be converted to annotations.json format.

Supports multiple data sources:
- CSV/JSON files with video URLs
- ActivityNet format
- Local video files with annotation files
- YouTube videos (via yt-dlp)
"""

import os
import json
import csv
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import subprocess
import shutil


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_with_ytdlp(url: str, output_dir: str, video_id: str) -> Optional[str]:
    """Download video using yt-dlp (YouTube, Vimeo, etc.)."""
    try:
        # Check if yt-dlp is available
        if not shutil.which('yt-dlp'):
            print("yt-dlp not found. Install with: pip install yt-dlp")
            return None
        
        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', output_path,
            '--no-playlist',
            url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"Error downloading {url}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error with yt-dlp for {url}: {e}")
        return None


def download_from_csv(csv_path: str, output_dir: str, url_column: str = 'url', 
                     id_column: Optional[str] = None) -> List[Dict]:
    """Download videos from CSV file with URLs."""
    videos = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in tqdm(reader, desc="Downloading videos"):
            url = row.get(url_column)
            if not url:
                continue
            
            # Generate video ID
            if id_column and id_column in row:
                video_id = row[id_column]
            else:
                video_id = f"video_{len(videos)}"
            
            # Determine if it's a YouTube/Vimeo URL
            if 'youtube.com' in url or 'youtu.be' in url or 'vimeo.com' in url:
                video_path = download_with_ytdlp(url, str(output_dir), video_id)
            else:
                # Direct download
                ext = os.path.splitext(url)[1] or '.mp4'
                video_path = output_dir / f"{video_id}{ext}"
                if not download_file(url, str(video_path)):
                    continue
            
            if video_path and os.path.exists(video_path):
                videos.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'url': url,
                    'metadata': {k: v for k, v in row.items() if k != url_column}
                })
    
    return videos


def download_from_json(json_path: str, output_dir: str, url_key: str = 'url',
                      id_key: Optional[str] = None) -> List[Dict]:
    """Download videos from JSON file with URLs."""
    videos = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'videos' in data:
            items = data['videos']
        else:
            items = [data]
        
        for idx, item in enumerate(tqdm(items, desc="Downloading videos")):
            url = item.get(url_key)
            if not url:
                continue
            
            # Generate video ID
            if id_key and id_key in item:
                video_id = item[id_key]
            else:
                video_id = item.get('id', f"video_{idx}")
            
            # Determine if it's a YouTube/Vimeo URL
            if 'youtube.com' in url or 'youtu.be' in url or 'vimeo.com' in url:
                video_path = download_with_ytdlp(url, str(output_dir), video_id)
            else:
                # Direct download
                ext = os.path.splitext(url)[1] or '.mp4'
                video_path = output_dir / f"{video_id}{ext}"
                if not download_file(url, str(video_path)):
                    continue
            
            if video_path and os.path.exists(video_path):
                videos.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'url': url,
                    'metadata': {k: v for k, v in item.items() if k not in [url_key, id_key]}
                })
    
    return videos


def download_activitynet_format(annotations_path: str, video_dir: str, 
                                base_url: Optional[str] = None) -> List[Dict]:
    """
    Download videos in ActivityNet format.
    Supports multiple ActivityNet formats:
    1. Standard: {"database": {"video_id": {"url": "...", "annotations": [...]}}}
    2. Captions: [{"video": "path", "caption": "...", "start_time": ..., "end_time": ...}]
    """
    videos = []
    output_dir = Path(video_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's a list (Captions format) or dict with 'database' key (standard format)
    if isinstance(data, list):
        # ActivityNet Captions format: list of caption annotations
        print("Detected ActivityNet Captions format (list of annotations)")
        # Group by video path
        video_dict = {}
        for ann in data:
            video_path = ann.get('video', '')
            if not video_path:
                continue
            
            # Extract video ID from path (e.g., "video/v_ehGHCYKzyZ8.mp4" -> "v_ehGHCYKzyZ8")
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
            if video_id not in video_dict:
                video_dict[video_id] = {
                    'video_id': video_id,
                    'video_path': video_path,
                    'annotations': []
                }
            
            # Add annotation segment
            video_dict[video_id]['annotations'].append({
                'segment': [ann.get('start_time', 0.0), ann.get('end_time', 0.0)],
                'label': ann.get('caption', ''),
                'description': ann.get('caption', '')
            })
        
        # Process each video
        found_count = 0
        not_found_count = 0
        downloaded_count = 0
        
        print(f"Processing {len(video_dict)} unique videos from {len(data)} annotations...")
        
        for video_id, video_info in tqdm(video_dict.items(), desc="Processing ActivityNet Captions videos"):
            video_path = video_info['video_path']
            
            # Try to find the video file locally first
            local_video_path = None
            
            # 1. Check in output directory with video_id
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']:
                candidate = output_dir / f"{video_id}{ext}"
                if candidate.exists():
                    local_video_path = candidate
                    break
            
            # 2. Check if it's an absolute path that exists
            if not local_video_path and os.path.isabs(video_path) and os.path.exists(video_path):
                local_video_path = Path(video_path)
            
            # 3. Check relative to output directory
            if not local_video_path:
                candidate = output_dir / video_path
                if candidate.exists():
                    local_video_path = candidate
            
            # 4. Check relative to annotation file directory
            if not local_video_path:
                ann_dir = os.path.dirname(annotations_path)
                candidate = Path(ann_dir) / video_path
                if candidate.exists():
                    local_video_path = candidate
            
            # 5. Check just the filename in output directory
            if not local_video_path:
                filename = os.path.basename(video_path)
                candidate = output_dir / filename
                if candidate.exists():
                    local_video_path = candidate
            
            # 6. Try to download from base_url if provided
            if not local_video_path and base_url:
                url = f"{base_url.rstrip('/')}/{video_path.lstrip('/')}"
                ext = os.path.splitext(video_path)[1] or '.mp4'
                download_path = output_dir / f"{video_id}{ext}"
                print(f"  Attempting to download {video_id} from {url}...")
                if download_file(url, str(download_path)):
                    local_video_path = download_path
                    downloaded_count += 1
            
            if local_video_path and local_video_path.exists():
                found_count += 1
                videos.append({
                    'video_id': video_id,
                    'video_path': str(local_video_path),
                    'annotations': video_info['annotations'],
                    'metadata': {}
                })
            else:
                not_found_count += 1
        
        print(f"\n{'='*60}")
        print(f"Video processing summary:")
        print(f"  Total unique videos: {len(video_dict)}")
        print(f"  Videos found locally: {found_count - downloaded_count}")
        print(f"  Videos downloaded: {downloaded_count}")
        print(f"  Videos not found: {not_found_count}")
        print(f"{'='*60}")
        
        if not_found_count > 0:
            print(f"\n⚠️  Note: ActivityNet Captions format doesn't include download URLs.")
            print(f"  Options to get videos:")
            print(f"  1. Place videos in: {output_dir}")
            print(f"     - Name them as: {list(video_dict.keys())[0] if video_dict else 'video_id'}.mp4")
            print(f"     - Or use the original path structure from annotations")
            print(f"  2. Use --base-url to download from a server:")
            print(f"     python scripts/download_data.py --input ... --base-url https://example.com/videos/")
            print(f"  3. Download videos separately and place them in the output directory")
        
        return videos
    
    elif isinstance(data, dict):
        # Standard ActivityNet format with 'database' key
        database = data.get('database', {})
        
        if not database:
            raise ValueError("ActivityNet format file does not contain 'database' key and is not a list format")
        
        for video_id, video_info in tqdm(database.items(), desc="Downloading ActivityNet videos"):
            url = video_info.get('url', '')
            if not url:
                continue
            
            # Construct full URL if base_url is provided
            if base_url and not url.startswith('http'):
                url = f"{base_url.rstrip('/')}/{url.lstrip('/')}"
            
            # Download video
            if 'youtube.com' in url or 'youtu.be' in url:
                video_path = download_with_ytdlp(url, str(output_dir), video_id)
            else:
                ext = os.path.splitext(url)[1] or '.mp4'
                video_path = output_dir / f"{video_id}{ext}"
                if not download_file(url, str(video_path)):
                    continue
            
            if video_path and os.path.exists(video_path):
                videos.append({
                    'video_id': video_id,
                    'video_path': str(video_path),
                    'url': url,
                    'annotations': video_info.get('annotations', []),
                    'metadata': {k: v for k, v in video_info.items() if k not in ['url', 'annotations']}
                })
        
        return videos
    else:
        raise ValueError(f"Unsupported ActivityNet format. Expected list or dict with 'database' key, got {type(data)}")


def copy_local_videos(video_list_path: str, output_dir: str) -> List[Dict]:
    """Copy local video files to output directory."""
    videos = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read list of video paths
    with open(video_list_path, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    for idx, video_path in enumerate(tqdm(video_paths, desc="Copying videos")):
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        video_id = f"video_{idx}"
        ext = os.path.splitext(video_path)[1]
        dest_path = output_dir / f"{video_id}{ext}"
        
        try:
            shutil.copy2(video_path, dest_path)
            videos.append({
                'video_id': video_id,
                'video_path': str(dest_path),
                'original_path': video_path
            })
        except Exception as e:
            print(f"Error copying {video_path}: {e}")
    
    return videos


def main():
    parser = argparse.ArgumentParser(description='Download video data for training')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file (CSV, JSON, or ActivityNet format)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for downloaded videos')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'activitynet', 'local'],
                       default='json',
                       help='Input format type')
    parser.add_argument('--url-column', type=str, default='url',
                       help='Column/key name for video URL (for CSV/JSON)')
    parser.add_argument('--id-column', type=str, default=None,
                       help='Column/key name for video ID (for CSV/JSON)')
    parser.add_argument('--base-url', type=str, default=None,
                       help='Base URL for relative paths (ActivityNet)')
    parser.add_argument('--metadata-output', type=str, default=None,
                       help='Output JSON file to save download metadata')
    
    args = parser.parse_args()
    
    print(f"Downloading videos from {args.input}...")
    print(f"Output directory: {args.output_dir}")
    
    # Download based on format
    if args.format == 'csv':
        videos = download_from_csv(args.input, args.output_dir, 
                                  args.url_column, args.id_column)
    elif args.format == 'json':
        videos = download_from_json(args.input, args.output_dir,
                                   args.url_column, args.id_column)
    elif args.format == 'activitynet':
        videos = download_activitynet_format(args.input, args.output_dir, args.base_url)
    elif args.format == 'local':
        videos = copy_local_videos(args.input, args.output_dir)
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    print(f"\nSuccessfully downloaded {len(videos)} videos")
    
    # Save metadata
    if args.metadata_output:
        with open(args.metadata_output, 'w', encoding='utf-8') as f:
            json.dump({'videos': videos}, f, indent=2)
        print(f"Metadata saved to {args.metadata_output}")
    
    return videos


if __name__ == '__main__':
    main()

