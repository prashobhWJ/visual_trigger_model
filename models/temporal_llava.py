"""
Stage 3: Temporal LLaVA (Large Language and Vision Assistant)
Vision-language model for video understanding with detailed scene descriptions.
"""

import torch
import torch.nn as nn
from transformers import AutoProcessor
from typing import Optional, List
import warnings

# Use AutoModelForImageTextToText if available, fallback to AutoModelForVision2Seq
try:
    from transformers import AutoModelForImageTextToText
    VisionModel = AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq
    VisionModel = AutoModelForVision2Seq


class TemporalLLaVA(nn.Module):
    """
    LLaVA-based vision-language model for video understanding.
    Uses pre-trained LLaVA model that understands both vision and language.
    """
    
    def __init__(
        self,
        llava_model_name: str = "llava-hf/llava-1.5-7b-hf",  # LLaVA 1.5 7B
        feature_dim: int = 768,
        max_length: int = 512,
        freeze_llm: bool = True,
        llm_dtype: str = "float16",  # LLaVA models work better with float16
        use_gradient_checkpointing: bool = True,
        skip_loading: bool = False  # Skip loading from HuggingFace (for inference when checkpoint has weights)
    ):
        """
        Args:
            llava_model_name: Name of the LLaVA model from HuggingFace
            feature_dim: Dimension of input features from time-aware encoder (not used directly, kept for compatibility)
            max_length: Maximum sequence length
            freeze_llm: Whether to freeze LLM weights (only train adapters)
            llm_dtype: Data type for model weights
            use_gradient_checkpointing: Enable gradient checkpointing for memory savings
            skip_loading: If True, skip loading from HuggingFace (weights will be loaded from checkpoint)
        """
        super().__init__()
        self.llava_model_name = llava_model_name
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        self._llm_dtype = dtype_map.get(llm_dtype.lower(), torch.float16)
        
        # Initialize as None - will be loaded later if needed
        self.processor = None
        self.model = None
        self._skip_loading = skip_loading
        
        if not skip_loading:
            # Load LLaVA processor and model from HuggingFace
            self._load_llava_model(llava_model_name, freeze_llm, llm_dtype)
        else:
            # Only load processor (needed for inference), delay model loading until checkpoint is loaded
            print("⏭️  Skipping LLaVA model loading from HuggingFace (will load from checkpoint)")
            print("   Loading processor only (needed for inference)...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    llava_model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
            except:
                self.processor = AutoProcessor.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf",
                    trust_remote_code=True,
                    use_fast=True
                )
            print("✓ Processor loaded")
            # Validate processor after loading
            self._validate_processor()
    
    def _validate_processor(self):
        """
        Validate that the processor is properly initialized with required attributes.
        Fixes missing patch_size attribute if needed.
        """
        if self.processor is None:
            return False
        
        # Check if processor has patch_size attribute
        if not hasattr(self.processor, 'patch_size') or self.processor.patch_size is None:
            # Try to get patch_size from model config if model is loaded
            if self.model is not None:
                try:
                    # Try to get from vision config
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'vision_config'):
                        vision_config = self.model.config.vision_config
                        if hasattr(vision_config, 'patch_size'):
                            self.processor.patch_size = vision_config.patch_size
                            print(f"✓ Set processor.patch_size from model config: {self.processor.patch_size}")
                        elif hasattr(vision_config, 'image_size'):
                            # Infer patch_size from image_size (typically image_size // patch_size = grid_size)
                            # For LLaVA, patch_size is usually 14 or 16
                            image_size = vision_config.image_size
                            if isinstance(image_size, (list, tuple)):
                                image_size = image_size[0]
                            # Common patch sizes: 14 for LLaVA 1.5, 16 for LLaVA-NeXT
                            # Try to infer from image_size (336 // 14 = 24, 336 // 16 = 21)
                            if image_size % 14 == 0:
                                self.processor.patch_size = 14
                            elif image_size % 16 == 0:
                                self.processor.patch_size = 16
                            else:
                                self.processor.patch_size = 14  # Default
                            print(f"✓ Inferred processor.patch_size: {self.processor.patch_size}")
                except Exception as e:
                    print(f"⚠️  Could not get patch_size from model config: {e}")
            
            # If still None, set default based on model name
            if not hasattr(self.processor, 'patch_size') or self.processor.patch_size is None:
                # Set default based on model type
                if "OneVision" in self.llava_model_name or "llava-next" in self.llava_model_name.lower():
                    default_patch_size = 16  # LLaVA-NeXT uses 16
                else:
                    default_patch_size = 14  # LLaVA 1.5 uses 14
                
                # Set patch_size directly if processor has image_processor
                if hasattr(self.processor, 'image_processor'):
                    try:
                        if hasattr(self.processor.image_processor, 'patch_size'):
                            self.processor.image_processor.patch_size = default_patch_size
                        # Also set on processor itself
                        self.processor.patch_size = default_patch_size
                        print(f"✓ Set default processor.patch_size: {default_patch_size}")
                    except Exception as e:
                        print(f"⚠️  Could not set patch_size on image_processor: {e}")
                        # Try setting directly on processor
                        try:
                            self.processor.patch_size = default_patch_size
                            print(f"✓ Set default processor.patch_size directly: {default_patch_size}")
                        except Exception as e2:
                            print(f"⚠️  Could not set patch_size: {e2}")
                else:
                    # Set directly on processor
                    try:
                        self.processor.patch_size = default_patch_size
                        print(f"✓ Set default processor.patch_size: {default_patch_size}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not set patch_size attribute: {e}")
        
        # Verify processor has required attributes
        has_image_processor = hasattr(self.processor, 'image_processor') or hasattr(self.processor, 'patch_size')
        if not has_image_processor:
            print("⚠️  Warning: Processor may not be fully initialized")
            return False
        
        return True
    
    def _load_llava_model(self, llava_model_name: str, freeze_llm: bool, llm_dtype: str):
        """Load LLaVA model from HuggingFace"""
        try:
            print(f"Loading LLaVA model: {llava_model_name}")
            # Use fast processor to avoid deprecation warning
            self.processor = AutoProcessor.from_pretrained(
                llava_model_name, 
                trust_remote_code=True,
                use_fast=True
            )
            
            # Load model with appropriate dtype
            self.model = VisionModel.from_pretrained(
                llava_model_name,
                dtype=self._llm_dtype,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Enable gradient checkpointing if available
            if self.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled for LLaVA")
            
            if self._llm_dtype != torch.float32:
                print(f"✓ LLaVA loaded with {llm_dtype} precision")
            
            # Freeze LLM if requested
            if freeze_llm:
                for param in self.model.parameters():
                    param.requires_grad = False
                print("✓ LLaVA base model frozen (only adapters will be trained)")
            
            # Validate processor after loading
            self._validate_processor()
            
            print("✓ LLaVA model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Could not load LLaVA model {llava_model_name}: {e}")
            print(f"   This model may require a newer version of transformers or may not be compatible.")
            print(f"   Trying fallback: llava-hf/llava-1.5-7b-hf")
            try:
                # Fallback to a known working model
                self.processor = AutoProcessor.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf", 
                    trust_remote_code=True,
                    use_fast=True
                )
                self.model = VisionModel.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf",
                    dtype=self._llm_dtype,
                    trust_remote_code=True,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                if freeze_llm:
                    for param in self.model.parameters():
                        param.requires_grad = False
                
                # Validate processor after fallback loading
                self._validate_processor()
                
                print("✓ Fallback LLaVA model loaded")
            except Exception as e2:
                raise RuntimeError(f"Failed to load LLaVA model: {e2}")
    
    def ensure_model_loaded(self):
        """Ensure model is loaded (for lazy loading when skip_loading=True)"""
        if self.model is not None:
            return  # Already loaded
        
        if self._skip_loading:
            # Need to load model structure (weights will come from checkpoint)
            dtype_str = "float16" if self._llm_dtype == torch.float16 else "float32"
            print("Loading LLaVA model structure (weights will be loaded from checkpoint)...")
            self._load_llava_model(self.llava_model_name, freeze_llm=True, llm_dtype=dtype_str)
    
    def forward(
        self,
        frames: torch.Tensor,
        prompt: Optional[str] = None,
        return_dict: bool = True
    ) -> dict:
        # Ensure model is loaded (for lazy loading)
        if self.model is None:
            self.ensure_model_loaded()
        """
        Forward pass through LLaVA model.
        
        Args:
            frames: Tensor of shape (B, T, C, H, W) or (B, C, H, W) - video frames or single image
            prompt: Optional text prompt (default: scene description prompt)
            return_dict: Whether to return dictionary output
        
        Returns:
            Dictionary with 'logits' and 'generated_text'
        """
        if frames.dim() == 5:
            # (B, T, C, H, W) - video frames, use middle frame or average
            batch_size, num_frames = frames.shape[0], frames.shape[1]
            # Use middle frame for efficiency (can be changed to average or all frames)
            middle_idx = num_frames // 2
            frames = frames[:, middle_idx, :, :, :]  # (B, C, H, W)
        
        # Convert frames to PIL Images for LLaVA processor
        # Frames may be normalized with ImageNet stats, so we need to denormalize first
        frames_processed = frames.clone().detach()
        
        # Check if frames are normalized (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalized frames typically have values outside [0, 1] range
        is_normalized = frames_processed.min() < 0 or frames_processed.max() > 1.5
        
        if is_normalized:
            # Denormalize: pixel = (normalized_pixel * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames_processed.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frames_processed.device).view(1, 3, 1, 1)
            frames_processed = frames_processed * std + mean
            frames_processed = frames_processed.clamp(0, 1)
        
        # Convert to [0, 255] range
        if frames_processed.max() <= 1.0:
            frames_processed = frames_processed * 255.0
        
        # Convert to uint8
        frames_processed = frames_processed.clamp(0, 255).byte()
        
        # Convert tensor to list of PIL Images
        from PIL import Image
        
        images = []
        for i in range(frames_processed.shape[0]):
            # Convert tensor to PIL Image
            img_tensor = frames_processed[i].cpu()
            # Convert CHW to HWC
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            # Convert to numpy then PIL
            img_np = img_tensor.numpy().astype('uint8')
            # Handle RGB vs BGR (LLaVA expects RGB)
            if img_np.shape[2] == 3:
                # Assume RGB, but if needed can convert BGR to RGB
                img = Image.fromarray(img_np, mode='RGB')
            else:
                img = Image.fromarray(img_np)
            images.append(img)
        
        # Prompt should be provided by caller - if None, use minimal default
        # This default should be overridden via config.yaml for better control
        if prompt is None:
            prompt = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"
        
        # Process images - LLaVA can have issues with batch processing, so process individually
        device = next(self.model.parameters()).device
        was_training = self.training
        self.eval()
        
        # Validate processor before processing
        self._validate_processor()
        
        responses = []
        with torch.no_grad():
            # Process each image individually to avoid token/feature mismatches
            for img in images:
                # Process single image with prompt - with error handling
                try:
                    inputs = self.processor(
                        text=prompt,
                        images=img,  # Single image, not a list
                        return_tensors="pt",
                        padding=True
                    )
                except TypeError as e:
                    if "patch_size" in str(e) or "NoneType" in str(e):
                        # patch_size is None - try to fix it
                        print(f"⚠️  Processor error detected in forward(): {e}")
                        print("   Attempting to fix processor.patch_size...")
                        
                        # Validate and fix processor
                        if self._validate_processor():
                            # Retry processor call after fixing
                            try:
                                inputs = self.processor(
                                    text=prompt,
                                    images=img,
                                    return_tensors="pt",
                                    padding=True
                                )
                            except Exception as e2:
                                raise RuntimeError(f"Failed to process image even after fixing patch_size: {e2}")
                        else:
                            raise RuntimeError(f"Could not fix processor.patch_size. Original error: {e}")
                    else:
                        # Different TypeError - re-raise
                        raise
                
                # Move inputs to model device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Generate with LLaVA
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
                
                # Decode generated text
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract only the assistant's response (remove prompt)
                if len(generated_text) > 0:
                    text = generated_text[0]
                    if "ASSISTANT:" in text:
                        response = text.split("ASSISTANT:")[-1].strip()
                    else:
                        response = text.strip()
                    responses.append(response)
                else:
                    responses.append("")
        
        if was_training:
            self.train()
        
        # For training compatibility, we need to return logits
        # LLaVA doesn't provide logits in generate mode, so we'll return None
        # The training pipeline should handle this (LLaVA is typically frozen)
        if return_dict:
            return {
                'generated_text': responses,
                'logits': None,  # LLaVA doesn't return logits in generate mode
                'hidden_states': None,
                'llava_output': responses  # Store for compatibility
            }
        else:
            return responses
    
    def generate(
        self,
        frames: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> List[str]:
        """
        Generate text descriptions from video frames.
        
        Args:
            frames: Tensor of shape (B, T, C, H, W) or (B, C, H, W) - video frames
            prompt: Optional text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (if do_sample=True)
            do_sample: Whether to use sampling
        
        Returns:
            List of generated text strings (one per batch item)
        """
        # Ensure model is loaded (for lazy loading)
        if self.model is None:
            self.ensure_model_loaded()
        
        self.eval()
        
        if frames.dim() == 5:
            # (B, T, C, H, W) - use middle frame or can average multiple frames
            batch_size, num_frames = frames.shape[0], frames.shape[1]
            # For now, use middle frame (can be improved to use multiple frames)
            middle_idx = num_frames // 2
            frames = frames[:, middle_idx, :, :, :]  # (B, C, H, W)
        
        # Convert frames to PIL Images
        # Frames may be normalized with ImageNet stats, so we need to denormalize first
        frames_processed = frames.clone().detach()
        
        # Check if frames are normalized (ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalized frames typically have values outside [0, 1] range
        is_normalized = frames_processed.min() < 0 or frames_processed.max() > 1.5
        
        if is_normalized:
            # Denormalize: pixel = (normalized_pixel * std) + mean
            mean = torch.tensor([0.485, 0.456, 0.406], device=frames_processed.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=frames_processed.device).view(1, 3, 1, 1)
            frames_processed = frames_processed * std + mean
            frames_processed = frames_processed.clamp(0, 1)
        
        # Convert to [0, 255] range
        if frames_processed.max() <= 1.0:
            frames_processed = frames_processed * 255.0
        
        # Convert to uint8
        frames_processed = frames_processed.clamp(0, 255).byte()
        
        from PIL import Image
        images = []
        for i in range(frames_processed.shape[0]):
            img_tensor = frames_processed[i].cpu()
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
            img_np = img_tensor.numpy().astype('uint8')
            # Handle RGB format
            if img_np.shape[2] == 3:
                img = Image.fromarray(img_np, mode='RGB')
            else:
                img = Image.fromarray(img_np)
            
            # For OneVision / LLaVA-NeXT, we might need to ensure image size is compatible
            # But the processor handles resizing, so we just pass the PIL image
            images.append(img)
        
        # Determine which prompt format to use based on model name
        is_onevision = "OneVision" in self.llava_model_name or "llava-next" in self.llava_model_name.lower()
        
        if is_onevision:
            # OneVision models use a different chat template or prompt format
            # Typically: <|im_start|>user <image>\nPrompt<|im_end|><|im_start|>assistant
            if prompt is None:
                # Minimal default - should be overridden via config
                prompt_text = "Describe what you see in this image."
            else:
                prompt_text = prompt
                # Strip existing templates if user provided them manually
                if "USER:" in prompt_text:
                    prompt_text = prompt_text.split("USER:")[-1].split("ASSISTANT:")[0].strip()
                    prompt_text = prompt_text.replace("<image>", "").strip()
            
            # Construct conversation for processor
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            
            # Apply chat template using processor
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            # Prompt should be provided by caller - if None, use minimal default
            # This default should be overridden via config.yaml for better control
            if prompt is None:
                prompt = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"
        
        # Process with LLaVA - with error handling for patch_size issues
        try:
            if is_onevision:
                inputs = self.processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = self.processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
        except TypeError as e:
            if "patch_size" in str(e) or "NoneType" in str(e):
                # patch_size is None - try to fix it
                print(f"⚠️  Processor error detected: {e}")
                print("   Attempting to fix processor.patch_size...")
                
                # Validate and fix processor
                if self._validate_processor():
                    # Retry processor call after fixing
                    try:
                        if is_onevision:
                            inputs = self.processor(
                                text=prompt,
                                images=images,
                                return_tensors="pt",
                                padding=True
                            )
                        else:
                            inputs = self.processor(
                                text=prompt,
                                images=images,
                                return_tensors="pt",
                                padding=True
                            )
                        print("✓ Processor call succeeded after fixing patch_size")
                    except Exception as e2:
                        raise RuntimeError(f"Failed to process images even after fixing patch_size: {e2}")
                else:
                    raise RuntimeError(f"Could not fix processor.patch_size. Original error: {e}")
            else:
                # Different TypeError - re-raise
                raise
        
        # Verify input shapes for OneVision to avoid "Image features and image tokens do not match" error
        # This often happens if the number of image tokens <image> in prompt doesn't match actual images
        # or if the processor expands one image into multiple patches/features
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                use_cache=True
            )
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract assistant responses
        responses = []
        for text in generated_text:
            if "ASSISTANT:" in text:
                response = text.split("ASSISTANT:")[-1].strip()
            else:
                response = text.strip()
            responses.append(response)
        
        return responses
    
    def summarize_text(
        self,
        texts: List[str],
        prompt: Optional[str] = None,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> str:
        """
        Summarize multiple text analyses into a single comprehensive summary.
        Uses LLaVA's text generation capabilities with a placeholder image for text-only summarization.
        
        Args:
            texts: List of text analyses to summarize
            prompt: Optional custom prompt for summarization
            max_new_tokens: Maximum tokens for summary
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Summarized text string
        """
        if not texts or len(texts) == 0:
            return "No analyses to summarize."
        
        # Ensure model is loaded
        if self.model is None:
            self.ensure_model_loaded()
        
        self.eval()
        
        # Combine all analyses with timestamps if available
        combined_text = "\n\n".join([f"Analysis {i+1}: {text}" for i, text in enumerate(texts)])
        
        # Default prompt for summarization
        if prompt is None:
            prompt = f"USER: <image>\nBelow are multiple scene analyses from different moments in a video. Please provide a comprehensive summary that combines all these analyses into one coherent explanation of what happened in the entire video. Focus on the main events, actions, and important details.\n\n{combined_text}\n\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{prompt}\n\n{combined_text}\n\nASSISTANT:"
        
        # Create a placeholder image (1x1 pixel) for LLaVA processor
        # LLaVA processor requires images even for text-only tasks
        from PIL import Image
        import numpy as np
        placeholder_image = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        
        # Process with LLaVA (using placeholder image for text-only summarization)
        try:
            inputs = self.processor(
                text=prompt,
                images=[placeholder_image],  # Placeholder image required by processor
                return_tensors="pt",
                padding=True
            )
        except Exception as e:
            # Fallback: try without image if processor supports it
            try:
                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt",
                    padding=True
                )
            except:
                # Last resort: simple text concatenation
                return f"Video Summary: {' '.join(texts)}"
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    use_cache=True
                )
            except Exception as e:
                # Fallback: simple concatenation if generation fails
                print(f"Warning: Could not generate summary with LLaVA: {e}")
                return f"Video Summary: {' '.join(texts)}"
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract assistant response
        if len(generated_text) > 0:
            text = generated_text[0]
            if "ASSISTANT:" in text:
                summary = text.split("ASSISTANT:")[-1].strip()
            else:
                summary = text.strip()
            return summary
        else:
            return "Failed to generate summary."
    
    def freeze_llm(self):
        """Freeze the base LLaVA model"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_llm(self):
        """Unfreeze the base LLaVA model"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def freeze_projections(self):
        """LLaVA doesn't have separate projections, this is for compatibility"""
        pass
    
    def unfreeze_projections(self):
        """LLaVA doesn't have separate projections, this is for compatibility"""
        pass

