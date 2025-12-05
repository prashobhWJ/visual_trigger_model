# Data Download and Preprocessing Scripts

This directory contains scripts for downloading video data and preprocessing it into the `annotations.json` format required by the video trigger model.

## Scripts

### 1. `download_data.py`
Downloads videos from various sources (URLs, YouTube, local files) and saves them to a directory.

### 2. `preprocess_data.py`
Converts downloaded data files into the `annotations.json` format expected by the model.

## Installation

Make sure you have the required dependencies:

```bash
pip install -r ../requirements.txt
```

For YouTube downloads, install `yt-dlp`:
```bash
pip install yt-dlp
# or
pip install yt-dlp --upgrade
```

## Usage Examples

### Example 1: Download from CSV with URLs

1. Create a CSV file (`videos.csv`) with video URLs:
```csv
video_id,url,description
video_001,https://example.com/video1.mp4,Person walking
video_002,https://youtube.com/watch?v=abc123,Person running
```

2. Download videos:
```bash
python scripts/download_data.py \
    --input videos.csv \
    --format csv \
    --url-column url \
    --id-column video_id \
    --output-dir data/raw_videos \
    --metadata-output data/download_metadata.json
```

3. Create annotations CSV (`annotations.csv`):
```csv
video_path,timestamp,label,description
video_001.mp4,5.2,1,Person enters frame
video_001.mp4,12.5,1,Person exits frame
video_002.mp4,3.0,1,Person starts running
```

4. Preprocess to annotations.json:
```bash
python scripts/preprocess_data.py \
    --input annotations.csv \
    --format csv \
    --video-dir data/raw_videos \
    --output data/train/annotations.json \
    --validate
```

### Example 2: Download from JSON

1. Create a JSON file (`videos.json`):
```json
{
  "videos": [
    {
      "id": "video_001",
      "url": "https://example.com/video1.mp4",
      "triggers": [
        {"timestamp": 5.2, "label": 1, "description": "Person enters"},
        {"timestamp": 12.5, "label": 1, "description": "Person exits"}
      ],
      "description": "Person walking video"
    }
  ]
}
```

2. Download videos:
```bash
python scripts/download_data.py \
    --input videos.json \
    --format json \
    --url-column url \
    --id-column id \
    --output-dir data/raw_videos
```

3. Preprocess:
```bash
python scripts/preprocess_data.py \
    --input videos.json \
    --format json \
    --video-dir data/raw_videos \
    --output data/train/annotations.json \
    --validate
```

### Example 3: ActivityNet Format

1. Download ActivityNet annotations (JSON format):
```bash
# Download ActivityNet annotations from official source
# Then download videos:
python scripts/download_data.py \
    --input activitynet_annotations.json \
    --format activitynet \
    --output-dir data/raw_videos \
    --base-url https://example.com/videos/
```

2. Preprocess:
```bash
python scripts/preprocess_data.py \
    --input activitynet_annotations.json \
    --format activitynet \
    --video-dir data/raw_videos \
    --output data/train/annotations.json \
    --validate
```

### Example 4: Local Videos

1. Create a text file (`video_list.txt`) with paths:
```
/path/to/video1.mp4
/path/to/video2.mp4
/path/to/video3.mp4
```

2. Copy videos:
```bash
python scripts/download_data.py \
    --input video_list.txt \
    --format local \
    --output-dir data/raw_videos
```

3. Create annotations CSV and preprocess (same as Example 1, step 3-4).

### Example 5: YouTube Videos

1. Create CSV with YouTube URLs:
```csv
video_id,url
yt_001,https://www.youtube.com/watch?v=dQw4w9WgXcQ
yt_002,https://youtu.be/abc123
```

2. Download (requires yt-dlp):
```bash
python scripts/download_data.py \
    --input youtube_videos.csv \
    --format csv \
    --url-column url \
    --id-column video_id \
    --output-dir data/raw_videos
```

3. Create annotations and preprocess.

## Input Formats

### CSV Format
- **Required columns**: `video_path` (or custom), `timestamp`
- **Optional columns**: `label`, `description`
- **Example**:
```csv
video_path,timestamp,label,description
video1.mp4,5.2,1,Trigger event 1
video1.mp4,12.5,1,Trigger event 2
```

### JSON Format
- **Structure**:
```json
{
  "videos": [
    {
      "video_path": "video1.mp4",
      "triggers": [
        {"timestamp": 5.2, "label": 1, "description": "Event 1"},
        {"timestamp": 12.5, "label": 1, "description": "Event 2"}
      ],
      "description": "Overall video description"
    }
  ]
}
```

### ActivityNet Format
- Standard ActivityNet JSON structure with `database` key
- Each video has `url` and `annotations` with `segment` (start, end) times

## Output Format

Both scripts produce `annotations.json` in the format:
```json
{
  "videos": [
    {
      "video_path": "relative/path/to/video.mp4",
      "triggers": [
        {
          "timestamp": 5.2,
          "label": 1,
          "description": "Person enters the room"
        }
      ],
      "description": "Overall video description"
    }
  ]
}
```

## Command-Line Options

### download_data.py

- `--input`: Input file (CSV, JSON, or ActivityNet format)
- `--output-dir`: Output directory for downloaded videos
- `--format`: Input format (`csv`, `json`, `activitynet`, `local`)
- `--url-column`: Column/key name for video URL (default: `url`)
- `--id-column`: Column/key name for video ID (optional)
- `--base-url`: Base URL for relative paths (ActivityNet)
- `--metadata-output`: Output JSON file to save download metadata

### preprocess_data.py

- `--input`: Input file (CSV, JSON, or ActivityNet format)
- `--output`: Output annotations.json file path
- `--video-dir`: Directory containing video files
- `--format`: Input format (`csv`, `json`, `activitynet`, `download-metadata`)
- `--video-path-column`: Column/key name for video path (default: `video_path`)
- `--timestamp-column`: Column name for timestamp (default: `timestamp`)
- `--label-column`: Column name for label (default: `label`)
- `--description-column`: Column name for description (default: `description`)
- `--annotation-source`: Additional annotation file (for download-metadata format)
- `--validate`: Validate video files and timestamps

## Tips

1. **Validation**: Always use `--validate` flag to ensure video files exist and timestamps are within video duration.

2. **Relative Paths**: The preprocessing script converts paths to relative paths based on `--video-dir`, which is useful for portability.

3. **YouTube Downloads**: For YouTube videos, ensure `yt-dlp` is installed and up-to-date.

4. **Large Datasets**: For large datasets, consider downloading in batches and merging the annotations.

5. **Timestamp Formats**: The preprocessing script supports various timestamp formats:
   - Seconds: `5.2`
   - MM:SS: `05:20`
   - HH:MM:SS: `00:05:20`
   - Minutes+Seconds: `5m20s`

## Troubleshooting

- **Video not found**: Check that video paths are correct and relative to `--video-dir`
- **Download failures**: Check network connection and URL validity
- **YouTube errors**: Update `yt-dlp`: `pip install yt-dlp --upgrade`
- **Invalid timestamps**: Use `--validate` to see warnings about invalid timestamps

