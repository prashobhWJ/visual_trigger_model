"""
Gradio Web Interface for Video Trigger Model Inference
Provides a user-friendly interface for video analysis with checkpoint selection,
video viewing, timestamp navigation, and analysis display.
"""

import gradio as gr
import torch
import yaml
import json
import os
import glob
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import tempfile
import shutil

from models import VideoTriggerModel
from utils import VideoProcessor, FrameSampler

# Text-to-speech imports
try:
    from gtts import gTTS
    TTS_AVAILABLE = "gtts"
except ImportError:
    try:
        import pyttsx3
        TTS_AVAILABLE = "pyttsx3"
    except ImportError:
        TTS_AVAILABLE = None
        print("⚠️ No TTS library found. Install gtts (pip install gtts) or pyttsx3 for audio summary.")

# Global variables to cache model and config
cached_model = None
cached_config = None
cached_checkpoint_path = None


def get_checkpoint_files() -> List[str]:
    """Get list of available checkpoint files."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
    return [os.path.basename(cp) for cp in checkpoints]


def load_model(checkpoint_name: str, config_path: str = "config.yaml", device: str = "auto") -> Tuple[Optional[str], Optional[Dict]]:
    """
    Load model from checkpoint.
    Returns: (error_message, results_dict)
    """
    global cached_model, cached_config, cached_checkpoint_path
    
    try:
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device_obj = torch.device(device)
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if we can reuse cached model
        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        if cached_model is not None and cached_checkpoint_path == checkpoint_path:
            return None, {"status": "Model already loaded", "device": str(device_obj)}
        
        # Clear GPU cache
        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_state = checkpoint['model_state_dict']
        
        # Detect architecture from checkpoint
        all_keys = list(checkpoint_state.keys())
        has_llava = any('temporal_llm.model.' in key for key in all_keys)
        
        # Create model
        model = VideoTriggerModel(
            visual_encoder_type=config['model']['visual_encoder']['type'],
            visual_encoder_pretrained=False,  # Don't load pretrained weights, use checkpoint
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
            llm_model_name=config['model']['llm'].get('model_name', 'google/gemma-3-1b-it'),
            llm_max_length=config['model']['llm']['max_length'],
            use_temporal_lstm=config['model']['llm'].get('use_temporal_lstm', True),
            temporal_lstm_hidden=config['model']['llm'].get('temporal_lstm_hidden', 512),
            temporal_lstm_layers=config['model']['llm'].get('temporal_lstm_layers', 2),
            llm_dtype=config['model']['llm'].get('dtype', 'float32'),
            use_gradient_checkpointing=config['model']['llm'].get('use_gradient_checkpointing', True),
            use_llava=config['model']['llm'].get('use_llava', True),
            llava_model_name=config['model']['llm'].get('llava_model_name', 'llava-hf/llava-1.5-7b-hf'),
            skip_llava_loading=has_llava,  # Skip if checkpoint has LLaVA weights
            clip_window_size=config['data']['clip_window_size'],
            clip_overlap=config['data']['clip_overlap']
        ).to(device_obj)
        
        # Load checkpoint weights
        try:
            model.load_state_dict(checkpoint_state, strict=False)
        except Exception as e:
            print(f"Warning: Some weights couldn't be loaded: {e}")
        
        model.eval()
        
        # Cache model
        cached_model = model
        cached_config = config
        cached_checkpoint_path = checkpoint_path
        
        # Get checkpoint info
        epoch = checkpoint.get('epoch', 'Unknown')
        loss = checkpoint.get('loss', 'Unknown')
        
        return None, {
            "status": "Model loaded successfully",
            "device": str(device_obj),
            "epoch": epoch,
            "loss": loss,
            "checkpoint": checkpoint_name
        }
        
    except Exception as e:
        return f"Error loading model: {str(e)}", None


def text_to_speech(text: str) -> Optional[str]:
    """
    Convert text to speech and return audio file path.
    Returns: Path to generated audio file or None if failed
    """
    if not text or len(text.strip()) == 0:
        return None
    
    try:
        # Create temporary audio file
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        audio_path = audio_file.name
        audio_file.close()
        
        if TTS_AVAILABLE == "gtts":
            # Use Google TTS (requires internet)
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_path)
            print(f"✓ Generated audio summary with gTTS: {audio_path}")
            return audio_path
            
        elif TTS_AVAILABLE == "pyttsx3":
            # Use pyttsx3 (offline)
            engine = pyttsx3.init()
            engine.save_to_file(text, audio_path)
            engine.runAndWait()
            print(f"✓ Generated audio summary with pyttsx3: {audio_path}")
            return audio_path
        else:
            print("⚠️ No TTS library available")
            return None
            
    except Exception as e:
        print(f"⚠️ TTS generation failed: {e}")
        return None


def analyze_video(
    video_file: Optional[str],
    checkpoint_name: str,
    max_frames: int,
    threshold: float,
    device: str,
    progress=gr.Progress()
) -> Tuple[gr.Video, Optional[str], Optional[str], Optional[str], Optional[gr.File]]:
    """
    Analyze video and return results.
    Args:
        max_frames: Maximum number of frames to analyze (will sample evenly if more detected)
        threshold: Threshold for detecting triggers
    Returns: (video_update, analysis_html, summary_text, audio_path, json_file)
    """
    global cached_model, cached_config
    
    if video_file is None:
        return gr.Video(), "⚠️ Please upload a video file", "", None, gr.File(visible=False)
    
    if cached_model is None or cached_config is None:
        error, info = load_model(checkpoint_name, device=device)
        if error:
            return gr.Video(), f"⚠️ {error}", "", None, gr.File(visible=False)
    
    try:
        progress(0.1, desc="Loading model...")
        
        # Ensure model is loaded
        if cached_model is None:
            error, info = load_model(checkpoint_name, device=device)
            if error:
                return gr.Video(), f"⚠️ {error}", "", None, gr.File(visible=False)
        
        progress(0.2, desc="Processing video...")
        
        # Initialize processors
        frame_sampler = FrameSampler(fps=cached_config['data']['frame_sampling_rate'])
        video_processor = VideoProcessor()
        
        # Sample frames
        frames, timestamps = frame_sampler.sample_frames(video_file)
        frame_tensors = video_processor.process_frames(frames)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # Move to device
        device_obj = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        frame_tensors = frame_tensors.to(device_obj)
        timestamps_tensor = timestamps_tensor.to(device_obj)
        
        progress(0.4, desc="Running inference...")
        
        # Get LLaVA image size
        llava_image_size = cached_config['inference'].get('llava_image_size', [336, 336])
        if isinstance(llava_image_size, list):
            llava_image_size = tuple(llava_image_size)
        elif isinstance(llava_image_size, int):
            llava_image_size = (llava_image_size, llava_image_size)
        
        # Get prompts from config
        llava_prompt = cached_config['inference'].get('llava_prompt', None)
        llm_prompt = cached_config['inference'].get('llm_prompt', None)
        
        # Run inference - sampling happens INSIDE the model before LLaVA analysis
        print(f"Running inference with threshold={threshold:.3f}, max_frames={max_frames}")
        print(f"  LLaVA will only analyze up to {max_frames} frames (sampled evenly if more triggers detected)")
        results = cached_model.infer_triggered_analysis(
            frame_tensors,
            timestamps=timestamps_tensor,
            trigger_threshold=threshold,
            video_path=video_file,
            llava_image_size=llava_image_size,
            max_frames=max_frames,  # Pass max_frames to model for early sampling
            llava_prompt=llava_prompt,
            llm_prompt=llm_prompt
        )
        
        # Track if sampling occurred (for display purposes)
        # Note: We don't know the exact original count since sampling happens in model
        # But we can check if we hit the limit
        original_count = len(results)  # This is already sampled
        was_sampled = len(results) == max_frames  # Likely sampled if we hit the exact limit
        
        progress(0.8, desc="Generating summary...")
        
        # Generate summary if enabled
        video_summary = None
        if len(results) > 0 and cached_config['inference'].get('generate_summary', True):
            analysis_texts = [r['analysis'] for r in results if r.get('analysis')]
            if len(analysis_texts) > 0:
                try:
                    if hasattr(cached_model, 'temporal_llm') and hasattr(cached_model.temporal_llm, 'summarize_text'):
                        video_summary = cached_model.temporal_llm.summarize_text(
                            texts=analysis_texts,
                            prompt=None,
                            max_new_tokens=cached_config['inference'].get('summary_max_tokens', 300),
                            temperature=0.7,
                            do_sample=False
                        )
                except Exception as e:
                    print(f"Summary generation error: {e}")
                    video_summary = " ".join(analysis_texts)
        
        progress(1.0, desc="Complete!")
        
        # Format analysis HTML (without timestamp buttons)
        # Note: Since sampling happens in model, we show results as-is
        # If results == max_frames, likely means sampling occurred
        was_sampled_flag = len(results) == max_frames
        analysis_html = format_analysis_html(results, video_file, was_sampled_flag, None)
        
        # Format summary
        summary_text = video_summary if video_summary else "No summary available."
        
        # Generate audio from summary
        audio_path = None
        if video_summary and TTS_AVAILABLE:
            print("🔊 Generating audio summary...")
            audio_path = text_to_speech(video_summary)
            if audio_path:
                print("✓ Audio summary ready")
        
        # Create results dict for JSON download
        results_dict = {
            'video_path': video_file,
            'num_frames_analyzed': len(frames),
            'num_triggers_detected': len(results),
            'max_frames_limit': max_frames,
            'note': 'LLaVA analysis performed only on sampled frames if max_frames exceeded',
            'triggers': results,
            'video_summary': video_summary
        }
        
        # Save results to JSON file
        json_file = None
        if results_dict:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(results_dict, f, indent=2)
                json_file = gr.File(value=f.name, visible=True)
        else:
            json_file = gr.File(visible=False)
        
        # Return gr.Video() to keep the video in place (don't reload it)
        return gr.Video(), analysis_html, summary_text, audio_path, json_file
        
    except Exception as e:
        import traceback
        error_msg = f"⚠️ Error during analysis: {str(e)}\n\n{traceback.format_exc()}"
        return gr.Video(), error_msg, "", None, gr.File(visible=False)


def format_analysis_html(results: List[Dict], video_path: str, was_sampled: bool = False, total_triggers: int = None) -> str:
    """Format analysis results as HTML."""
    if len(results) == 0:
        return """
        <div style='padding: 20px; text-align: center; color: #666;'>
            <h3>No triggers detected</h3>
            <p>Try lowering the threshold or check if the video contains the events the model was trained to detect.</p>
        </div>
        """
    
    sampling_info = ""
    if was_sampled:
        sampling_info = f"""
        <div style='background: #d1ecf1; border-left: 4px solid #0c5460; padding: 12px; margin-bottom: 15px; border-radius: 5px;'>
            <strong>ℹ️ Efficient Analysis:</strong> Analyzed {len(results)} frames with LLaVA (sampled evenly based on max frames setting)
        </div>
        """
    
    html = f"""
    <div style='font-family: Arial, sans-serif; max-width: 100%;'>
        <h3 style='color: #2c3e50; margin-bottom: 20px;'>
            📊 Analysis Results ({len(results)} {'sampled ' if was_sampled else ''}frames analyzed)
        </h3>
        {sampling_info}
        <div style='max-height: 600px; overflow-y: auto;'>
    """
    
    for i, result in enumerate(results):
        timestamp = result['timestamp']
        confidence = result['trigger_confidence']
        analysis = result.get('analysis', 'No analysis available')
        frame_idx = result.get('frame_index', 'N/A')
        
        # Color code by confidence
        if confidence > 0.7:
            conf_color = "#27ae60"  # Green
        elif confidence > 0.5:
            conf_color = "#f39c12"  # Orange
        else:
            conf_color = "#e74c3c"  # Red
        
        html += f"""
        <div style='
            border: 2px solid #ecf0f1;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <h4 style='margin: 0; color: #2c3e50;'>
                    🎯 Trigger #{i+1}
                </h4>
                <span style='
                    background: {conf_color};
                    color: white;
                    padding: 5px 12px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 0.9em;
                '>
                    {confidence:.1%}
                </span>
            </div>
            <div style='margin-bottom: 10px;'>
                <strong style='color: #7f8c8d;'>Timestamp:</strong>
                <span style='color: #3498db; font-weight: bold; font-size: 1.1em; margin-left: 8px;'>
                    {timestamp:.2f}s
                </span>
                <span style='color: #95a5a6; margin-left: 10px; font-size: 0.9em;'>
                    (Frame: {frame_idx})
                </span>
            </div>
            <div style='
                background: #f8f9fa;
                padding: 12px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
                margin-top: 10px;
            '>
                <strong style='color: #2c3e50; display: block; margin-bottom: 5px;'>Analysis:</strong>
                <p style='margin: 0; color: #34495e; line-height: 1.6;'>{analysis}</p>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html


def create_timestamp_buttons(results: List[Dict]) -> List[gr.Button]:
    """Create timestamp navigation buttons."""
    buttons = []
    for i, result in enumerate(results):
        timestamp = result['timestamp']
        confidence = result['trigger_confidence']
        btn = gr.Button(
            value=f"⏱️ {timestamp:.2f}s (Conf: {confidence:.1%})",
            variant="secondary",
            size="sm"
        )
        buttons.append(btn)
    return buttons


def jump_to_timestamp(timestamp: float, video_player) -> Tuple[float, str]:
    """Jump video player to specific timestamp."""
    return timestamp, f"Jumped to {timestamp:.2f}s"


# Create Gradio interface
def create_interface():
    """Create and return Gradio interface."""
    
    # Get available checkpoints
    checkpoints = get_checkpoint_files()
    if not checkpoints:
        checkpoints = ["No checkpoints found"]
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    """
    
    # Use Gradio Blocks - inject CSS via HTML (works with all versions)
    # Note: Some Gradio versions don't support theme parameter, so we use plain Blocks
    with gr.Blocks() as app:
        # Inject CSS via HTML
        gr.HTML(f"<style>{custom_css}</style>")
        
        # Header
        gr.Markdown("""
        <div class="main-header">
            <h1>🎬 Video Trigger Model - Interactive Analysis</h1>
            <p>Upload a video, select a checkpoint, and analyze triggers with detailed scene descriptions</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### ⚙️ Configuration")
                
                checkpoint_dropdown = gr.Dropdown(
                    choices=checkpoints,
                    value=checkpoints[0] if checkpoints else None,
                    label="📦 Checkpoint",
                    info="Select a trained model checkpoint"
                )
                
                device_radio = gr.Radio(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="🖥️ Device",
                    info="Auto will use GPU if available"
                )
                
                max_frames_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=10,
                    step=1,
                    label="🎯 Max Frames to Analyze",
                    info="Maximum number of frames to analyze (will sample evenly if more triggers detected)"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.000000001,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="🔍 Detection Threshold",
                    info="Threshold for detecting triggers (lower = more sensitive)"
                )
                
                load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Model not loaded",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                # Video viewer (both input and output)
                gr.Markdown("### 📹 Video Input & Playback")
                video_player = gr.Video(
                    label="Upload and View Video",
                    sources=["upload"],
                    elem_id="main_video_player"
                )
                
                analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
        
        # Results section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Analysis Results")
                analysis_html = gr.HTML(label="Trigger Analysis")
                
                gr.Markdown("### 📝 Video Summary")
                summary_text = gr.Textbox(
                    label="Summary",
                    lines=5,
                    interactive=False,
                    placeholder="Summary will appear here after analysis..."
                )
                
                gr.Markdown("### 🔊 Audio Summary")
                audio_player = gr.Audio(
                    label="Listen to Summary",
                    autoplay=True
                )
                
                download_json = gr.File(
                    label="Download Results (JSON)",
                    visible=False
                )
        
        # Event handlers
        def on_load_model(checkpoint, device):
            error, info = load_model(checkpoint, device=device)
            if error:
                return f"❌ {error}"
            else:
                status = f"✅ {info.get('status', 'Loaded')}"
                if 'epoch' in info:
                    status += f" | Epoch: {info['epoch']}"
                if 'device' in info:
                    status += f" | Device: {info['device']}"
                return status
        
        def on_analyze(video, checkpoint, max_frames, threshold, device, progress=gr.Progress()):
            return analyze_video(video, checkpoint, max_frames, threshold, device, progress)
        
        # Bind events
        load_model_btn.click(
            fn=on_load_model,
            inputs=[checkpoint_dropdown, device_radio],
            outputs=[model_status]
        )
        
        analyze_btn.click(
            fn=on_analyze,
            inputs=[video_player, checkpoint_dropdown, max_frames_slider, threshold_slider, device_radio],
            outputs=[video_player, analysis_html, summary_text, audio_player, download_json]
        )
        
        # Auto-load model when checkpoint or device changes
        checkpoint_dropdown.change(
            fn=on_load_model,
            inputs=[checkpoint_dropdown, device_radio],
            outputs=[model_status]
        )
        
        device_radio.change(
            fn=on_load_model,
            inputs=[checkpoint_dropdown, device_radio],
            outputs=[model_status]
        )
        
        # Instructions
        gr.Markdown("""
        ---
        ### 📖 Instructions
        
        1. **Select a checkpoint** from the dropdown (or it will auto-load)
        2. **Upload a video** file using the video player
        3. **Set max frames** - Maximum number of frames to analyze (default: 10)
        4. **Adjust threshold** - Detection sensitivity (lower = more triggers detected)
        5. **Click "Analyze Video"** to run inference
        6. **Review results** showing detected triggers with detailed analysis
        7. **Listen to audio summary** - Auto-plays when analysis completes
        8. **Download results** as JSON for further analysis
        
        **Tips:**
        - **Max Frames**: Controls cost/time - if more triggers detected, they'll be sampled evenly
        - **Threshold**: Lower values (0.1-0.3) detect more events, higher (0.5+) are more conservative
        - **Audio Summary**: Automatically generated and plays when analysis completes
        - The model analyzes frames at 3 FPS for efficiency
        - Only frames above the threshold trigger detailed LLaVA analysis
        - If triggers exceed max frames, the system samples them evenly across the timeline
        """)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="192.168.68.58",
        server_port=7860,
        share=True,
        show_error=True
    )

