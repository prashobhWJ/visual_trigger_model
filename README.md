# Video Trigger Model

A multi-stage AI model architecture for video analysis that triggers a large language model (LLM) based on video analysis triggers and performs detailed frame analysis with high temporal awareness.

## Architecture Overview

The model consists of three main stages:

### Stage 1: Lightweight Video Analysis and Trigger Detection
- **Visual Encoder**: Processes video frames at modest frame rate (2-5 FPS) using CNN or Transformer-based encoder
- **Trigger Detector**: Detects events or triggers (object appearance, action start, scene change) using classification head
- Acts as a filter to decide when to involve a more complex LLM for deep analysis

### Stage 2: Frame Selection and Time-Aware Feature Encoding
- Selects a sequence of frames around detected triggers for detailed analysis
- **Time-Aware Encoder**: Encodes frames with temporal awareness using timestamps and spatiotemporal context
- Supports overlapping temporal windows for context

### Stage 3: High Temporal Awareness with Large LLM
- Feeds encoded features into a larger LLM with temporal modules (bidirectional LSTM or Transformer layers)
- Captures long-term dependencies and temporal patterns
- Outputs natural language descriptions, summaries, or answers based on video content and temporal context

📊 **For detailed architecture diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md) or [ARCHITECTURE_SIMPLE.txt](ARCHITECTURE_SIMPLE.txt)**

## Project Structure

```
video_trigger_model/
├── models/                 # Model components
│   ├── visual_encoder.py      # Stage 1: Visual feature extraction
│   ├── trigger_detector.py    # Stage 1: Trigger detection
│   ├── time_aware_encoder.py  # Stage 2: Temporal encoding
│   ├── temporal_llm.py        # Stage 3: LLM with temporal reasoning
│   └── video_trigger_model.py # Complete model pipeline
├── utils/                 # Utility functions
│   ├── video_utils.py         # Video processing utilities
│   ├── data_loader.py         # Dataset and data loading
│   └── training_utils.py       # Training utilities (loss, checkpointing)
├── config.yaml           # Configuration file
├── train.py              # Training script
├── inference.py          # Inference script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
cd video_trigger_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to configure:
- Data paths (training/validation videos and annotations)
- Model architecture (encoder types, dimensions, etc.)
- Training parameters (batch size, learning rate, etc.)
- Inference settings

## Data Format

The model expects annotations in JSON format:

```json
{
  "videos": [
    {
      "video_path": "path/to/video.mp4",
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

## Training

### Pre-training
1. Pre-train the visual feature extractor and clip encoder on large video datasets
2. Pre-train the LLM on aligned video-text datasets

### Trigger Detection Training
Train the trigger detection head on event-annotated videos

### Joint Fine-tuning
Fine-tune the entire pipeline end-to-end using video clips with natural language annotations

### Run Training

```bash
python train.py --config config.yaml
```

To resume from a checkpoint:
```bash
python train.py --config config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

## Inference

Run inference on a video:

```bash
python inference.py \
    --config config.yaml \
    --checkpoint checkpoints/final_checkpoint.pt \
    --video path/to/video.mp4 \
    --output outputs/analysis.json
```

The output JSON contains:
- Detected triggers with timestamps
- Confidence scores
- LLM-generated analysis for each trigger

## Model Components

### VisualEncoder
- Supports ResNet (18, 50) and EfficientNet-B0 backbones
- Extracts frame-level features
- Configurable feature dimensions

### TriggerDetector
- Binary or multi-class trigger detection
- Optional temporal context using LSTM
- Configurable threshold for detection

### TimeAwareEncoder
- Transformer or LSTM-based temporal encoding
- Incorporates timestamp information
- Supports positional encoding

### TemporalLLM
- Integrates pre-trained language models (GPT-2, LLaMA, Gemma, Smol-Llama, etc.)
- Currently configured to use Gemma 2B (instruction-tuned)
- Supports Gemma 3 270M if available on HuggingFace
- Bidirectional LSTM for temporal reasoning
- Generates natural language descriptions (20 words or less)
- Supports both GPT-style and LLaMA-style architectures

## Training Loop

The training loop follows this structure:

```python
for each video in training_set:
    frames = sample_frames(video)
    features = visual_encoder(frames)
    triggers = trigger_detector(features)
    
    for each detected_trigger in triggers:
        clip_frames = extract_frames_around(trigger.timestamp)
        clip_features = time_aware_encoder(clip_frames, timestamps)
        llm_input = prepare_llm_input(clip_features)
        llm_output = LLM(llm_input)
        loss = compute_loss(llm_output, ground_truth_text)
        backpropagate(loss)
        update_model_parameters()
```

## Loss Functions

The model uses a multi-task loss:
- **Trigger Loss**: Cross-entropy for trigger detection
- **LLM Loss**: Language generation loss (next-token prediction)
- **Temporal Loss**: Encourages temporal smoothness in predictions

## Performance Considerations

- **Efficiency**: Only processes frames at 2-5 FPS for trigger detection
- **Selective Processing**: Detailed LLM analysis only on triggered frames
- **Batch Processing**: Supports batched inference for multiple videos
- **GPU Acceleration**: Fully supports CUDA for faster training/inference

## Customization

### Adding New Encoders
Extend `VisualEncoder` class to support new backbone architectures

### Custom Trigger Types
Modify `TriggerDetector` to support different trigger classes or detection strategies

### Different LLMs
Update `TemporalLLM` to use different base models (LLaMA, Mistral, etc.)

## References

This implementation is inspired by:
- TemporalVLM: Short-term clip encoding with timestamp fusion
- BiLSTM-based global feature aggregation for temporal reasoning
- Recent advances in vision-language models for video understanding

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

