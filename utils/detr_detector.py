"""
DETR (Detection Transformer) Object Detection Utility
Provides object detection capabilities for detecting people and major objects in video frames.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Tuple, Optional
import numpy as np

# COCO classes (80 classes + 1 N/A)
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Major objects to focus on (people and common objects)
MAJOR_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'chair', 'couch', 'bed', 'dining table', 'tv', 'laptop',
    'cell phone', 'book', 'bottle', 'cup', 'bowl', 'pizza', 'cake'
]


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.
    
    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


def box_cxcywh_to_xyxy(x):
    """Convert boxes from [x_center, y_center, w, h] to [x0, y0, x1, y1] format."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """Convert boxes from [0; 1] to image scales."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    # Create tensor on the same device as out_bbox
    scale_tensor = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    b = b * scale_tensor
    return b


class DETRDetector:
    """DETR-based object detector for video frames."""
    
    def __init__(self, device: str = "auto", confidence_threshold: float = 0.7):
        """
        Initialize DETR detector.
        
        Args:
            device: Device to run inference on ("auto", "cuda", or "cpu")
            confidence_threshold: Minimum confidence for detections
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load pretrained DETR model."""
        print("Loading DETR model...")
        self.model = DETRdemo(num_classes=91).to(self.device)
        
        # Load pretrained weights
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
                map_location=self.device
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("✓ DETR model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load DETR weights: {e}")
            print("   DETR detection will be disabled")
            self.model = None
            return
        
        # Setup transforms
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect(self, image: Image.Image) -> List[Dict]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detection dictionaries with keys:
            - 'class': class name
            - 'confidence': confidence score
            - 'bbox': [x0, y0, x1, y1] bounding box coordinates
        """
        if self.model is None:
            return []
        
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check image size
            if image.size[0] > 1600 or image.size[1] > 1600:
                # Resize if too large
                max_size = max(image.size)
                scale = 1600 / max_size
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Ensure minimum size
            if image.size[0] < 32 or image.size[1] < 32:
                # Resize if too small
                min_size = min(image.size)
                scale = 32 / min_size
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Transform image - ensure tensor is on correct device
            img_tensor = self.transform(image).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            # Process outputs
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove "no object" class
            keep = probas.max(-1).values > self.confidence_threshold
            
            if not keep.any():
                return []
            
            # Get class predictions
            scores, labels = probas[keep].max(-1)
            
            # Convert boxes
            bboxes = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
            
            # Format results
            detections = []
            for i in range(len(scores)):
                class_idx = labels[i].item()
                class_name = COCO_CLASSES[class_idx]
                confidence = scores[i].item()
                bbox = bboxes[i].cpu().numpy().tolist()
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox  # [x0, y0, x1, y1]
                })
            
            return detections
            
        except Exception as e:
            print(f"⚠️ DETR detection error: {e}")
            return []
    
    def detect_major_objects(self, image: Image.Image) -> List[Dict]:
        """
        Detect only major objects (people and common objects) in an image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detection dictionaries filtered to major objects
        """
        all_detections = self.detect(image)
        
        # Filter to major objects
        major_detections = [
            det for det in all_detections 
            if det['class'] in MAJOR_OBJECTS
        ]
        
        return major_detections
    
    def format_detections_text(self, detections: List[Dict]) -> str:
        """
        Format detections as human-readable text.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Formatted text string
        """
        if not detections:
            return "No objects detected"
        
        # Group by class
        class_counts = {}
        for det in detections:
            class_name = det['class']
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(det['confidence'])
        
        # Format output
        parts = []
        for class_name, confidences in sorted(class_counts.items()):
            count = len(confidences)
            avg_conf = sum(confidences) / len(confidences)
            if count == 1:
                parts.append(f"{class_name} ({avg_conf:.1%})")
            else:
                parts.append(f"{count} {class_name}s ({avg_conf:.1%} avg)")
        
        return ", ".join(parts)

