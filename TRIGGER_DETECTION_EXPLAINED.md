# How Trigger Detection Works

## Overview

The trigger detector **does NOT explicitly detect frame differences or similarity**. Instead, it's trained to detect **specific events/activities** that were annotated in your training data.

## Architecture

### 1. **Visual Feature Extraction (ResNet)**
```
Video Frames → ResNet Visual Encoder → Visual Features (512-dim)
```
- ResNet extracts high-level visual features from each frame
- These features capture objects, scenes, actions, etc.
- **This is where frame content is analyzed**

### 2. **Temporal Context (LSTM)**
```
Visual Features → LSTM → Temporal Features
```
- LSTM processes sequences of visual features
- Considers previous frames to understand temporal patterns
- **This helps detect changes over time, but indirectly**

### 3. **Trigger Classification**
```
Temporal Features → Classifier → Trigger Probability (0-1)
```
- Binary classifier: "trigger" (1) or "no trigger" (0)
- Outputs probability for each frame
- Frames above threshold are considered triggers

## Training Process

### What the Model Learns

The model is trained using **supervised learning** with ground truth annotations:

1. **Training Data Format**:
   ```json
   {
     "videos": [{
       "video_path": "video.mp4",
       "triggers": [
         {"timestamp": 5.2, "label": 1, "description": "person enters"},
         {"timestamp": 12.5, "label": 1, "description": "object moves"}
       ]
     }]
   }
   ```

2. **Label Creation**:
   - Frames within **1 second** of an annotated trigger timestamp → label = 1 (trigger)
   - All other frames → label = 0 (no trigger)

3. **Loss Function**:
   - **Cross-entropy loss** between predicted trigger probabilities and ground truth labels
   - Model learns: "What visual features indicate a trigger event?"

### What It's NOT Learning

- ❌ **NOT** explicitly learning frame-to-frame differences
- ❌ **NOT** learning similarity metrics
- ❌ **NOT** detecting scene changes directly

### What It IS Learning

- ✅ Learning visual patterns that correlate with annotated trigger events
- ✅ Learning temporal patterns (via LSTM) that indicate trigger events
- ✅ Learning to distinguish "interesting" vs "boring" frames based on training annotations

## Why Triggers Every 2 Seconds?

If you're seeing triggers every 2 seconds, possible causes:

### 1. **Model Not Properly Trained**
- Using random/untrained weights
- Checkpoint might not have trigger detector weights
- **Solution**: Verify checkpoint contains trained trigger detector weights

### 2. **Threshold Too Low**
- Default threshold: 0.3 (in config.yaml)
- If model outputs high confidences for all frames, low threshold = many triggers
- **Solution**: Increase threshold or check if model is trained

### 3. **Training Data Pattern**
- If training data had triggers at regular intervals, model might learn that pattern
- **Solution**: Check training annotations for regular patterns

### 4. **Model Detecting Visual Patterns**
- Model might be detecting some visual pattern (lighting, motion, etc.) as triggers
- **Solution**: Check trigger confidences in debug output

## How to Verify Model is Working

### 1. Check Trigger Confidences
The debug output shows:
```
Trigger Detection Results (threshold=0.300):
  Total sampled frames: X
  Confidence range: [0.XXXX, 0.XXXX]
  Mean confidence: 0.XXXX
  First 10 frame confidences:
    Frame 0 (t=0.00s): confidence=0.XXXX
```

**What to look for**:
- If all confidences are > 0.5 → Model might not be trained (outputting high values)
- If confidences vary → Model is working, but threshold might be wrong
- If confidences are very low (< 0.1) → Model is trained but threshold too low

### 2. Check Checkpoint Contents
```python
import torch
ckpt = torch.load('checkpoint.pt', map_location='cpu')
keys = list(ckpt['model_state_dict'].keys())
trigger_keys = [k for k in keys if 'trigger_detector' in k]
print(f"Trigger detector keys: {len(trigger_keys)}")
```

**What to look for**:
- If `len(trigger_keys) == 0` → Trigger detector not in checkpoint
- If `len(trigger_keys) > 0` → Trigger detector weights are saved

### 3. Check Training History
- Did training loss decrease?
- Did trigger accuracy improve?
- If loss stayed constant → Model didn't learn

## Improving Trigger Detection

### Option 1: Train on Better Data
- Annotate specific events you want to detect
- Ensure annotations are accurate (within 1 second of actual events)
- Include negative examples (videos with no triggers)

### Option 2: Adjust Threshold
- Increase threshold (e.g., 0.5) for fewer triggers
- Decrease threshold (e.g., 0.2) for more triggers
- Use validation set to find optimal threshold

### Option 3: Add Frame Difference Detection
If you want explicit frame difference detection, you could:
1. Add a separate module that computes frame-to-frame differences
2. Use optical flow or feature matching
3. Combine with trigger detector output

### Option 4: Use Temporal Smoothing
The model already has temporal loss that encourages smooth predictions, but you could:
- Add post-processing to filter out isolated triggers
- Require triggers to persist for N consecutive frames
- Use non-maximum suppression on trigger detections

## Summary

**The trigger detector learns to detect events based on training annotations, not explicit frame differences.** It uses:
- ResNet to extract visual features
- LSTM to understand temporal context
- Classifier to predict trigger probability

If triggers are happening every 2 seconds, check:
1. Is the model trained? (check checkpoint)
2. Is the threshold appropriate? (check confidences)
3. Does training data have regular patterns? (check annotations)

