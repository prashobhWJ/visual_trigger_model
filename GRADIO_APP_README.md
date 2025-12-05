Testing
# Gradio Web Interface for Video Trigger Model

A beautiful, user-friendly web interface for running inference on your trained video trigger model checkpoints.

## Features

✨ **Key Features:**
- 🎬 **Video Upload & Playback**: Upload videos and view them directly in the browser
- 📦 **Checkpoint Selection**: Easily select from available checkpoints in the `checkpoints/` folder
- ⏱️ **Timestamp Navigation**: Click on any detected trigger timestamp to instantly jump to that moment in the video
- 📊 **Detailed Analysis**: View trigger analysis with confidence scores and scene descriptions
- 📝 **Video Summary**: Get an AI-generated summary combining all trigger analyses
- 💾 **JSON Export**: Download results as JSON for further analysis
- 🎨 **Beautiful UI**: Modern, responsive design with gradient themes

## Installation

Make sure you have Gradio installed:

```bash
pip install gradio>=4.0.0
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the App

```bash
python gradio_app.py
```

The app will start on `http://localhost:7860` by default.

To make it accessible from other devices on your network:

```bash
python gradio_app.py
# The app will show: Running on local URL: http://127.0.0.1:7860
# And optionally: Running on public URL: https://xxxxx.gradio.live
```

### Using the Interface

1. **Select Checkpoint**
   - Choose a checkpoint from the dropdown (auto-scans `checkpoints/` folder)
   - The model will auto-load when you select a checkpoint
   - Check the "Model Status" to confirm it loaded successfully

2. **Upload Video**
   - Click "Upload Video" and select your video file
   - Supported formats: MP4, AVI, MOV, etc. (any format supported by OpenCV)

3. **Configure Settings**
   - **Device**: Choose "auto" (recommended), "cuda" (GPU), or "cpu"
   - **Trigger Threshold**: Adjust sensitivity (0.1-1.0)
     - Lower values (0.1-0.3) = More sensitive, detects more triggers
     - Higher values (0.5-1.0) = Less sensitive, only high-confidence triggers

4. **Analyze Video**
   - Click "🚀 Analyze Video" button
   - Wait for processing (progress bar will show status)
   - Results will appear automatically

5. **Navigate Results**
   - **View Analysis**: See all detected triggers with timestamps and descriptions
   - **Jump to Timestamp**: Click any timestamp button to jump to that moment in the video
   - **Read Summary**: View the AI-generated summary of all triggers
   - **Download JSON**: Download complete results for offline analysis

## Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│         🎬 Video Trigger Model - Interactive Analysis   │
├──────────────────┬──────────────────────────────────────┤
│ ⚙️ Configuration │        📹 Video Input                │
│                  │                                      │
│ Checkpoint: [▼]  │  [Upload Video Area]                │
│ Device: [auto]   │                                      │
│ Threshold: [━]  │  [🚀 Analyze Video Button]          │
│ [Load Model]     │                                      │
│ Status: ...      │                                      │
├──────────────────┴──────────────────────────────────────┤
│ 📊 Analysis Results          │  🎥 Video Player          │
│                              │                          │
│ [Trigger Cards]              │  [Video Player]          │
│                              │                          │
│ 📝 Video Summary             │  ⏱️ Jump to Timestamp    │
│ [Summary Text]               │  [Timestamp Buttons]    │
│                              │                          │
│ [Download JSON]              │                          │
└──────────────────────────────┴──────────────────────────┘
```

## Features Explained

### Timestamp Navigation

When triggers are detected, you'll see clickable timestamp buttons like:

```
⏱️ 5.23s | Confidence: 87.5%
⏱️ 12.45s | Confidence: 72.3%
⏱️ 18.90s | Confidence: 91.2%
```

Clicking any button will:
- Jump the video player to that exact timestamp
- Start playing the video from that point
- Highlight the corresponding analysis card

### Analysis Cards

Each trigger shows:
- **Trigger Number**: Sequential ID
- **Confidence Score**: Color-coded (Green >70%, Orange >50%, Red ≤50%)
- **Timestamp**: Exact time in seconds
- **Frame Index**: Frame number in the video
- **Analysis**: Detailed scene description from LLaVA

### Video Summary

The summary combines all trigger analyses into a single coherent explanation of what happened in the entire video. This is generated using LLaVA's text summarization capabilities.

## Troubleshooting

### Model Not Loading

- **Check checkpoint path**: Ensure checkpoints are in `checkpoints/` folder
- **Check config.yaml**: Verify the config file exists and is valid
- **Check device**: If using CUDA, ensure GPU is available
- **Check logs**: Look at the console output for error messages

### Video Not Playing

- **Format support**: Ensure video format is supported (MP4 recommended)
- **File size**: Very large videos may take time to load
- **Browser compatibility**: Try a different browser (Chrome/Firefox recommended)

### No Triggers Detected

- **Lower threshold**: Try reducing the threshold (e.g., 0.2 or 0.1)
- **Check video content**: Ensure video contains events the model was trained to detect
- **Check model**: Verify the checkpoint was trained on similar data

### Performance Issues

- **Use GPU**: Select "cuda" device if available (much faster)
- **Reduce video resolution**: Lower resolution videos process faster
- **Close other applications**: Free up GPU/CPU resources

## Advanced Usage

### Custom Port

Edit `gradio_app.py` and change:

```python
app.launch(
    server_port=7860,  # Change this
    ...
)
```

### Share Publicly

To create a public link (temporary):

```python
app.launch(share=True)
```

### Custom Theme

Modify the `theme` parameter:

```python
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    # ... your interface
```

## Technical Details

- **Model Loading**: Models are cached to avoid reloading on every analysis
- **Video Processing**: Uses OpenCV for frame extraction at 3 FPS (configurable)
- **Inference**: Runs full pipeline: ResNet → Trigger Detector → LLaVA
- **Memory Management**: Automatically clears GPU cache when needed

## File Structure

```
gradio_app.py          # Main Gradio application
checkpoints/           # Model checkpoints (auto-scanned)
config.yaml           # Model configuration
requirements.txt      # Dependencies (includes gradio)
```

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure checkpoints are valid and compatible
4. Check that config.yaml matches your training setup

## License

Same as the main project.

