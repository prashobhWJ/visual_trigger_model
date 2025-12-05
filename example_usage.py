"""
Example usage of the Video Trigger Model
Demonstrates how to use the model for inference
"""

import torch
import yaml
from models import VideoTriggerModel
from utils import VideoProcessor, FrameSampler

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = VideoTriggerModel(
    visual_encoder_type=config['model']['visual_encoder']['type'],
    visual_encoder_pretrained=config['model']['visual_encoder']['pretrained'],
    visual_feature_dim=config['model']['visual_encoder']['feature_dim'],
    trigger_input_dim=config['model']['trigger_detector']['input_dim'],
    trigger_hidden_dim=config['model']['trigger_detector']['hidden_dim'],
    trigger_num_classes=config['model']['trigger_detector']['num_classes'],
    trigger_threshold=config['model']['trigger_detector']['threshold'],
    time_aware_input_dim=config['model']['time_aware_encoder']['input_dim'],
    time_aware_hidden_dim=config['model']['time_aware_encoder']['hidden_dim'],
    time_aware_num_layers=config['model']['time_aware_encoder']['num_layers'],
    time_aware_num_heads=config['model']['time_aware_encoder']['num_heads'],
    time_aware_encoder_type=config['model']['time_aware_encoder']['type'],
    llm_model_name=config['model']['llm']['model_name'],
    llm_max_length=config['model']['llm']['max_length'],
    use_temporal_lstm=config['model']['llm']['use_temporal_lstm'],
    temporal_lstm_hidden=config['model']['llm']['temporal_lstm_hidden'],
    temporal_lstm_layers=config['model']['llm']['temporal_lstm_layers'],
    clip_window_size=config['data']['clip_window_size'],
    clip_overlap=config['data']['clip_overlap']
).to(device)

# Load checkpoint (if available)
# checkpoint = torch.load('checkpoints/final_checkpoint.pt', map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Example: Process a video
video_path = "path/to/your/video.mp4"

# Initialize processors
frame_sampler = FrameSampler(fps=config['data']['frame_sampling_rate'])
video_processor = VideoProcessor()

# Sample frames
print(f"Processing video: {video_path}")
frames, timestamps = frame_sampler.sample_frames(video_path)

# Process frames to tensors
frame_tensors = video_processor.process_frames(frames)  # (T, C, H, W)
timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)

# Run inference
print("Running inference...")
results = model.infer_triggered_analysis(
    frame_tensors,
    timestamps=timestamps_tensor,
    trigger_threshold=config['inference']['trigger_threshold']
)

# Print results
print(f"\nDetected {len(results)} triggers:")
for i, result in enumerate(results):
    print(f"\nTrigger {i+1}:")
    print(f"  Timestamp: {result['timestamp']:.2f}s")
    print(f"  Frame Index: {result['frame_index']}")
    print(f"  Confidence: {result['trigger_confidence']:.4f}")
    print(f"  Analysis: {result['analysis']}")

