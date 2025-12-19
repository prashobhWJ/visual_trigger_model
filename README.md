# Video Trigger Model

A multi-stage AI model for intelligent video analysis that detects trigger events and generates detailed scene descriptions using vision-language models (LLaVA). The model efficiently processes videos by only performing expensive LLM analysis on frames where triggers are detected.

[![Alt text](https://youtu.be/pVxIGV_BB0o)

## 🎯 Overview

The Video Trigger Model is designed for efficient video analysis with a focus on detecting important events and providing detailed scene descriptions. It uses a three-stage architecture:

1. **Visual Encoder + Trigger Detector**: Extracts visual features and identifies potential trigger frames
2. **Time-aware Encoder**: Processes temporal context around triggers
3. **Temporal LLM (LLaVA)**: Generates detailed scene descriptions only for detected triggers

This approach significantly reduces computational cost by avoiding expensive LLM inference on every frame.

## ✨ Features

- **Efficient Trigger Detection**: Only analyzes frames where triggers are detected, reducing computational cost
- **Multi-stage Architecture**: Combines visual encoding, temporal modeling, and vision-language understanding
- **LLaVA Integration**: Uses LLaVA (Large Language and Vision Assistant) for high-quality scene descriptions
- **Flexible Training**: Supports freezing/unfreezing different model components for efficient training
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) support for faster training and lower memory usage
- **Gradio Web Interface**: User-friendly web UI for video analysis
- **Object Detection**: Optional DETR-based object detection on triggered frames
- **Audio Summaries**: Text-to-speech generation for video summaries
- **Checkpoint Management**: Flexible checkpoint loading with architecture compatibility checking

## 🏗️ Architecture

The model consists of four main components:

### Stage 1: Visual Encoder
- Extracts visual features from video frames
- Supports ResNet18/50, EfficientNet-B0
- Configurable feature dimensions

### Stage 1: Trigger Detector
- Binary/multi-class classifier for trigger detection
- Identifies frames that require detailed analysis
- Configurable threshold for sensitivity

### Stage 2: Time-aware Encoder
- Processes temporal context around triggers
- Transformer or LSTM-based architecture
- Incorporates timestamp information

### Stage 3: Temporal LLM
- **LLaVA** (recommended): Vision-language model for scene description
- **Traditional LLM**: Text-only fallback option
- Temporal LSTM for sequence modeling
- Generates detailed analysis only for triggered frames

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended for training and inference)
- See `requirements.txt` for full dependency list

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd video_trigger_model_01
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Install package in development mode**:
```bash
pip install -e .
```

## ⚙️ Configuration

The model is configured via `config.yaml`. Key configuration sections:

### Data Configuration
- `train_video_dir`: Directory containing training videos
- `train_annotations`: Path to training annotations JSON
- `frame_sampling_rate`: FPS for frame sampling (default: 3)
- `clip_window_size`: Number of frames around trigger (default: 16)
- `max_frames`: Maximum frames per video sample (for memory efficiency)

### Model Configuration
- **Visual Encoder**: Type (resnet50, resnet18, efficientnet_b0), pretrained weights
- **Trigger Detector**: Hidden dimensions, number of classes, threshold
- **Time-aware Encoder**: Transformer/LSTM, layers, heads
- **LLM**: LLaVA model selection, temporal LSTM settings, dtype (float16/float32)

### Training Configuration
- `batch_size`: Batch size (default: 8)
- `num_epochs`: Training epochs (default: 50)
- `learning_rate`: Learning rate (default: 1e-4)
- `gradient_accumulation_steps`: Effective batch size multiplier
- `use_mixed_precision`: Enable AMP for memory efficiency

### Freezing Configuration
Control which components are frozen during training:
- `visual_backbone`: Freeze pretrained visual encoder
- `llm_base`: Freeze base LLM/LLaVA weights
- Other components can be trained while keeping base models frozen

## 🎓 Training

### Basic Training

```bash
python train.py --config config.yaml
```

### Resume from Checkpoint

```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Training Options

- `--config`: Path to config file (default: `config.yaml`)
- `--resume`: Path to checkpoint to resume from
- `--device`: Device to use (`cuda` or `cpu`)

### Training Features

- **Mixed Precision Training**: Automatically enabled if CUDA is available
- **Gradient Accumulation**: Maintains effective batch size with smaller batches
- **Gradient Clipping**: Prevents gradient explosion
- **Learning Rate Warmup**: Linear warmup for stable training
- **TensorBoard Logging**: Monitor training progress
- **Automatic Checkpointing**: Saves checkpoints periodically

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

## 🔍 Inference

### Command Line Inference

```bash
python inference.py \
    --config config.yaml \
    --checkpoint checkpoints/final_checkpoint.pt \
    --video path/to/video.mp4 \
    --output outputs/analysis.json \
    --threshold 0.3
```

### Inference Options

- `--config`: Path to config file
- `--checkpoint`: Path to model checkpoint (required)
- `--video`: Path to input video file (required)
- `--output`: Path to output JSON file (default: `outputs/analysis.json`)
- `--threshold`: Trigger detection threshold (overrides config)
- `--device`: Device to use (`cuda` or `cpu`)
- `--strict-loading`: Use strict checkpoint loading (fails on mismatches)

### Output Format

The inference script generates a JSON file with:
- `video_path`: Path to analyzed video
- `num_frames_analyzed`: Total frames processed
- `num_triggers_detected`: Number of triggers found
- `triggers`: Array of trigger results with:
  - `timestamp`: Trigger timestamp in seconds
  - `frame_index`: Frame index
  - `trigger_confidence`: Detection confidence
  - `analysis`: Detailed scene description
- `video_summary`: Combined summary of all analyses

## 🌐 Gradio Web Interface

Launch an interactive web interface for video analysis:

```bash
python gradio_app.py
```

The interface provides:
- **Checkpoint Selection**: Choose from available trained models
- **Video Upload**: Upload and view videos
- **Interactive Analysis**: Adjust threshold and max frames
- **Detailed Results**: View trigger detections with scene descriptions
- **Object Detection**: DETR-based object detection on triggered frames
- **Audio Summary**: Listen to text-to-speech summary
- **JSON Download**: Download analysis results

### Interface Features

- Real-time model loading status
- Configurable detection threshold
- Max frames limit for efficient analysis
- Frame-by-frame analysis display
- Object detection visualization
- Audio summary playback

## 📁 Project Structure

```
video_trigger_model_01/
├── models/                  # Model architectures
│   ├── video_trigger_model.py    # Main model class
│   ├── visual_encoder.py         # Visual feature extraction
│   ├── trigger_detector.py       # Trigger detection
│   ├── time_aware_encoder.py     # Temporal encoding
│   ├── temporal_llm.py            # Traditional LLM
│   └── temporal_llava.py         # LLaVA integration
├── utils/                  # Utility functions
│   ├── data_loader.py            # Dataset loading
│   ├── video_utils.py            # Video processing
│   ├── training_utils.py          # Training helpers
│   └── detr_detector.py          # Object detection
├── scripts/               # Helper scripts
│   ├── preprocess_data.py        # Data preprocessing
│   └── generate_annotations_from_videos.py
├── train.py               # Training script
├── inference.py           # Inference script
├── gradio_app.py          # Web interface
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 💡 Usage Examples

### Example 1: Basic Training

```python
# Train with default config
python train.py
```

### Example 2: Custom Inference

```python
from models import VideoTriggerModel
from utils import VideoProcessor, FrameSampler
import torch

# Load model
model = VideoTriggerModel(...)
checkpoint = torch.load('checkpoints/final_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process video
frame_sampler = FrameSampler(fps=3)
frames, timestamps = frame_sampler.sample_frames('video.mp4')
frame_tensors = VideoProcessor().process_frames(frames)

# Run inference
results = model.infer_triggered_analysis(
    frame_tensors,
    timestamps=torch.tensor(timestamps),
    trigger_threshold=0.3
)
```

### Example 3: Using the Gradio Interface

1. Start the interface: `python gradio_app.py`
2. Select a checkpoint from the dropdown
3. Upload a video file
4. Adjust threshold and max frames
5. Click "Analyze Video"
6. Review results and download JSON

## 🔧 Troubleshooting

### Memory Issues

- **Reduce batch size**: Lower `batch_size` in config
- **Enable gradient checkpointing**: Set `use_gradient_checkpointing: true`
- **Use mixed precision**: Set `use_mixed_precision: true`
- **Limit frames**: Set `max_frames` in config
- **Use smaller LLaVA model**: Try `llava-phi-3-mini` instead of larger models

### Checkpoint Loading Issues

- **Architecture mismatch**: Use `--strict-loading` to see exact errors
- **LLM model mismatch**: Ensure `llm.model_name` in config matches training
- **Missing weights**: Model will load compatible parts and initialize others

### Training Issues

- **No triggers detected**: Lower `trigger_threshold` in config
- **Loss not decreasing**: Check learning rate, try warmup
- **CUDA out of memory**: Reduce batch size, enable gradient accumulation

### Inference Issues

- **No triggers found**: Lower threshold with `--threshold 0.2`
- **Slow inference**: Reduce `max_frames` or use smaller LLaVA model
- **Poor descriptions**: Try different LLaVA models or adjust prompts in config

## 📊 Model Performance

The model is designed for efficiency:
- **Frame Sampling**: Analyzes at 3 FPS (configurable)
- **Trigger-based Analysis**: Only processes ~10-30% of frames with LLaVA
- **Memory Efficient**: Supports gradient checkpointing and mixed precision
- **Scalable**: Can handle videos of varying lengths

## 🔄 Checkpoint Management

Checkpoints are saved in the `checkpoints/` directory with:
- Model state dict
- Optimizer state
- Epoch number
- Loss value
- Model metadata (LLM name, etc.)

Checkpoints are automatically saved:
- Every N epochs (configurable via `save_every`)
- Final checkpoint at end of training

## 📝 Configuration Tips

### For Faster Training
- Use smaller LLaVA model (`llava-phi-3-mini`)
- Enable mixed precision
- Freeze more components
- Use gradient accumulation

### For Better Quality
- Use larger LLaVA model (`llava-1.5-7b-hf`)
- Train more components (unfreeze)
- Increase `clip_window_size` for more context
- Use higher resolution for LLaVA (`llava_image_size: [512, 512]`)

### For Memory Efficiency
- Reduce `batch_size`
- Set `max_frames` limit
- Enable `use_gradient_checkpointing`
- Use `float16` dtype
- Lower `clip_window_size`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- LLaVA models from HuggingFace
- PyTorch and Transformers libraries
- Gradio for the web interface

## 📧 Contact

[Your contact information]

---

**Note**: This model requires significant computational resources, especially for training. A GPU with at least 16GB VRAM is recommended for full functionality. For inference, 8GB+ VRAM is typically sufficient.

