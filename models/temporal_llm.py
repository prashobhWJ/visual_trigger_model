"""
Stage 3: Temporal LLM
Large language model with temporal modules for fine-grained understanding and reasoning
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, List
import warnings


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM module for capturing long-term temporal dependencies
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(output_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, input_dim)
        Returns:
            output: Tensor of shape (B, T, input_dim)
        """
        lstm_out, _ = self.lstm(x)
        output = self.output_projection(lstm_out)
        return output


class TemporalLLM(nn.Module):
    """
    LLM with temporal reasoning capabilities for video understanding.
    Combines pre-trained language model with temporal modules.
    """
    
    def __init__(
        self,
        llm_model_name: str = "google/gemma-3-270m",  # Default to Gemma 3 270M
        feature_dim: int = 768,
        max_length: int = 512,
        use_temporal_lstm: bool = True,
        temporal_lstm_hidden: int = 512,
        temporal_lstm_layers: int = 2,
        bidirectional: bool = True,
        freeze_llm: bool = False,
        llm_dtype: str = "float32",  # Options: "float32", "float16", "bfloat16"
        use_gradient_checkpointing: bool = True  # Enable gradient checkpointing for memory savings
    ):
        """
        Args:
            llm_model_name: Name of the base LLM model
            feature_dim: Dimension of input features from time-aware encoder
            max_length: Maximum sequence length
            use_temporal_lstm: Whether to use temporal LSTM before LLM
            temporal_lstm_hidden: Hidden dimension for temporal LSTM
            temporal_lstm_layers: Number of LSTM layers
            bidirectional: Whether LSTM is bidirectional
            freeze_llm: Whether to freeze LLM weights (only train adapters)
        """
        super().__init__()
        self.llm_model_name = llm_model_name
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.use_temporal_lstm = use_temporal_lstm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        self._llm_dtype = dtype_map.get(llm_dtype.lower(), torch.float32)
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            # Gemma and LLaMA models often need special token setup
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Gemma models may require a token (check if token is needed)
            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is not None:
                # Ensure pad token is set properly
                pass
        except Exception as e:
            warnings.warn(f"Could not load tokenizer for {llm_model_name}: {e}")
            self.tokenizer = None
        
        # Load LLM config and model
        # Determine dtype for memory efficiency
        llm_dtype = getattr(self, '_llm_dtype', torch.float32)  # Default to float32
        
        # Try AutoModelForCausalLM first (better for LLaMA/GPT models), fallback to AutoModel
        try:
            config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
            
            # Try loading as CausalLM first (for LLaMA, GPT, etc.)
            try:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_model_name, 
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=llm_dtype  # Use specified dtype (float16/bf16 for memory savings)
                )
            except:
                # Fallback to AutoModel if CausalLM doesn't work
                self.llm = AutoModel.from_pretrained(
                    llm_model_name, 
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=llm_dtype
                )
            
            # Enable gradient checkpointing if available (trades compute for memory)
            if use_gradient_checkpointing and hasattr(self.llm, 'gradient_checkpointing_enable'):
                self.llm.gradient_checkpointing_enable()
                print(f"✓ Gradient checkpointing enabled for LLM (reduces memory usage)")
            
            if llm_dtype != "float32":
                print(f"✓ LLM loaded with {llm_dtype} precision (reduces memory usage by ~50%)")
            
            if freeze_llm:
                for param in self.llm.parameters():
                    param.requires_grad = False
            
            # Get LLM input dimension - try to get from embedding layer first (most accurate)
            llm_input_dim = None
            if hasattr(self.llm, 'get_input_embeddings'):
                try:
                    embedding_layer = self.llm.get_input_embeddings()
                    if hasattr(embedding_layer, 'embedding_dim'):
                        llm_input_dim = embedding_layer.embedding_dim
                    elif hasattr(embedding_layer, 'weight'):
                        llm_input_dim = embedding_layer.weight.shape[1]  # (vocab_size, embedding_dim)
                except:
                    pass
            
            # Fallback to config if embedding layer check failed
            if llm_input_dim is None:
                if hasattr(self.llm.config, 'hidden_size'):
                    llm_input_dim = self.llm.config.hidden_size
                elif hasattr(self.llm.config, 'd_model'):
                    llm_input_dim = self.llm.config.d_model
                elif hasattr(self.llm.config, 'n_embd'):  # GPT-style
                    llm_input_dim = self.llm.config.n_embd
                elif hasattr(self.llm.config, 'vocab_size'):  # Some models store it differently
                    # Try to infer from model structure
                    if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'embed_tokens'):
                        llm_input_dim = self.llm.model.embed_tokens.embedding_dim
                    else:
                        llm_input_dim = 768  # Default fallback
                else:
                    llm_input_dim = 768  # Default
            
            # Verify the dimension matches what we'll get from embeddings
            try:
                test_embedding = self.llm.get_input_embeddings()
                if hasattr(test_embedding, 'embedding_dim'):
                    actual_embed_dim = test_embedding.embedding_dim
                elif hasattr(test_embedding, 'weight'):
                    actual_embed_dim = test_embedding.weight.shape[1]
                else:
                    actual_embed_dim = llm_input_dim
                
                if actual_embed_dim != llm_input_dim:
                    warnings.warn(
                        f"Dimension mismatch: detected {llm_input_dim} but embedding layer has {actual_embed_dim}. "
                        f"Using {actual_embed_dim}."
                    )
                    llm_input_dim = actual_embed_dim
                
                print(f"✓ LLM embedding dimension: {llm_input_dim}")
            except Exception as e:
                print(f"⚠ Could not verify embedding dimension: {e}")
                print(f"  Using detected dimension: {llm_input_dim}")
                
        except Exception as e:
            warnings.warn(f"Could not load LLM model {llm_model_name}: {e}")
            # Fallback to a simple projection
            llm_input_dim = 768
            self.llm = None
        
        # Temporal LSTM module
        if use_temporal_lstm:
            self.temporal_lstm = TemporalLSTM(
                input_dim=feature_dim,
                hidden_dim=temporal_lstm_hidden,
                num_layers=temporal_lstm_layers,
                bidirectional=bidirectional
            )
            temporal_output_dim = feature_dim
        else:
            self.temporal_lstm = None
            temporal_output_dim = feature_dim
        
        # Projection from video features to LLM input space
        self.feature_projection = nn.Sequential(
            nn.Linear(temporal_output_dim, llm_input_dim),
            nn.LayerNorm(llm_input_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Output projection for text generation
        # For trained models, we should use the LLM's own language modeling head
        # For untrained models, we'll create a custom projection (but it needs training)
        self.use_llm_head = False
        if self.llm is not None:
            # Try to use LLM's built-in language modeling head (preferred for inference)
            if hasattr(self.llm, 'lm_head'):
                # Check if lm_head has the right input dimension
                if hasattr(self.llm.lm_head, 'in_features'):
                    if self.llm.lm_head.in_features == llm_input_dim:
                        self.output_projection = self.llm.lm_head
                        self.use_llm_head = True
                        print("✓ Using LLM's built-in language modeling head")
                    else:
                        print(f"⚠ LLM lm_head input dim ({self.llm.lm_head.in_features}) doesn't match llm_input_dim ({llm_input_dim})")
                else:
                    # Try to use it anyway (some models don't expose in_features)
                    try:
                        # Test if dimensions match by checking weight shape
                        if hasattr(self.llm.lm_head, 'weight'):
                            # lm_head typically has shape (vocab_size, hidden_size)
                            if self.llm.lm_head.weight.shape[1] == llm_input_dim:
                                self.output_projection = self.llm.lm_head
                                self.use_llm_head = True
                                print("✓ Using LLM's built-in language modeling head")
                    except:
                        pass
            
            if not self.use_llm_head:
                if hasattr(self.llm, 'get_output_embeddings'):
                    output_emb = self.llm.get_output_embeddings()
                    if output_emb is not None:
                        self.output_projection = output_emb
                        self.use_llm_head = True
                        print("✓ Using LLM's output embeddings")
            
            # Get vocab size
            if hasattr(self.llm, 'get_input_embeddings'):
                vocab_size = self.llm.get_input_embeddings().weight.shape[0]
            elif hasattr(self.llm, 'config') and hasattr(self.llm.config, 'vocab_size'):
                vocab_size = self.llm.config.vocab_size
            else:
                vocab_size = 50257  # GPT-2 default
            
            # Create custom projection only if not using LLM head
            if not self.use_llm_head:
                self.output_projection = nn.Linear(llm_input_dim, vocab_size)
                print("⚠ Using custom output projection (needs training to generate meaningful text)")
        else:
            vocab_size = 50257
            self.output_projection = nn.Linear(llm_input_dim, vocab_size)
            self.use_llm_head = False
        
        # Track freezing state
        self._llm_frozen = freeze_llm
        self._temporal_lstm_frozen = False
        self._projections_frozen = False
    
    def forward(
        self,
        encoded_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> dict:
        """
        Args:
            encoded_features: Tensor of shape (B, T, feature_dim) - time-aware encoded features
            input_ids: Optional text input IDs for conditioning
            attention_mask: Optional attention mask for text input
            return_dict: Whether to return dictionary output
        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'
        """
        batch_size, seq_len, _ = encoded_features.shape
        
        # Apply temporal LSTM if enabled
        if self.use_temporal_lstm and self.temporal_lstm is not None:
            temporal_features = self.temporal_lstm(encoded_features)
        else:
            temporal_features = encoded_features
        
        # Project to LLM input dimension
        llm_input = self.feature_projection(temporal_features)  # (B, T, llm_input_dim)
        
        # If LLM is available, use it for processing
        if self.llm is not None:
            # For now, we'll use the LLM's encoder/transformer
            # In practice, you might want to concatenate with text embeddings
            if input_ids is not None:
                # Get text embeddings
                text_embeddings = self.llm.get_input_embeddings()(input_ids)
                
                # Ensure dimensions match - if they don't, project llm_input to match text_embeddings
                if llm_input.shape[-1] != text_embeddings.shape[-1]:
                    # Create a projection layer on-the-fly if needed (shouldn't happen if init was correct)
                    # But handle it gracefully just in case
                    if not hasattr(self, '_dimension_fix_projection'):
                        self._dimension_fix_projection = nn.Linear(
                            llm_input.shape[-1], 
                            text_embeddings.shape[-1]
                        ).to(llm_input.device)
                        warnings.warn(
                            f"Dimension mismatch detected: llm_input={llm_input.shape[-1]}, "
                            f"text_embeddings={text_embeddings.shape[-1]}. "
                            f"Using projection layer to fix. This suggests a configuration issue."
                        )
                    llm_input = self._dimension_fix_projection(llm_input)
                
                # Concatenate or combine with video features
                # Simple approach: prepend video features
                combined_input = torch.cat([llm_input, text_embeddings], dim=1)
            else:
                combined_input = llm_input
            
            # Pass through LLM
            # Handle different model architectures
            if hasattr(self.llm, 'transformer'):
                # GPT-style models (GPT-2, etc.)
                llm_output = self.llm.transformer(inputs_embeds=combined_input)[0]
            elif hasattr(self.llm, 'model'):
                # LLaMA-style models (LLaMA, Gemma, TinyLlama, Smol-Llama, etc.)
                llm_output = self.llm.model(inputs_embeds=combined_input)[0]
            elif hasattr(self.llm, 'encoder'):
                # BERT-style models
                llm_output = self.llm.encoder(inputs_embeds=combined_input)[0]
            elif hasattr(self.llm, '__call__'):
                # Try direct call (for some model types)
                try:
                    outputs = self.llm(inputs_embeds=combined_input)
                    if isinstance(outputs, tuple):
                        llm_output = outputs[0]
                    elif hasattr(outputs, 'last_hidden_state'):
                        llm_output = outputs.last_hidden_state
                    else:
                        llm_output = combined_input
                except:
                    llm_output = combined_input
            else:
                # Fallback: just use the input
                llm_output = combined_input
        else:
            # Fallback: use simple projection
            llm_output = llm_input
        
        # Project to vocabulary
        # For sequence output, we need logits for each position
        if llm_output.dim() == 3:
            # (B, T, hidden_dim) - get logits for all positions
            batch_size, seq_len, hidden_dim = llm_output.shape
            # Reshape to (B*T, hidden_dim) for efficient projection
            llm_output_flat = llm_output.reshape(-1, hidden_dim)  # (B*T, hidden_dim)
            logits_flat = self.output_projection(llm_output_flat)  # (B*T, vocab_size)
            # Reshape back to (B, T, vocab_size)
            logits = logits_flat.reshape(batch_size, seq_len, -1)  # (B, T, vocab_size)
        else:
            # (B, hidden_dim) or (hidden_dim,)
            if llm_output.dim() == 1:
                llm_output = llm_output.unsqueeze(0)  # (1, hidden_dim)
            logits = self.output_projection(llm_output)  # (B, vocab_size)
            # Add sequence dimension if needed
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)  # (B, 1, vocab_size)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': llm_output
            }
        else:
            return logits
    
    def generate(
        self,
        encoded_features: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_nucleus_sampling: bool = True,
        prompt: Optional[str] = None,
        max_words: int = 20,
        repetition_penalty: float = 1.2
    ) -> List[str]:
        """
        Generate text from encoded video features using autoregressive generation.
        Args:
            encoded_features: Tensor of shape (B, T, feature_dim)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (keep only top k tokens)
            top_p: Nucleus sampling threshold (cumulative probability)
            use_nucleus_sampling: Whether to use nucleus (top_p) or top_k sampling
            prompt: Optional text prompt to guide generation
            max_words: Maximum number of words to generate (default: 20)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition)
        Returns:
            List of generated text strings (one per batch item)
        """
        self.eval()
        batch_size = encoded_features.size(0)
        device = encoded_features.device
        
        # Get tokenizer special tokens
        if self.tokenizer is None:
            # Fallback: return empty strings if no tokenizer
            return [""] * batch_size
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else None
        bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else None
        
        # Prompt should be provided by caller - if None, use minimal default
        # This default should be overridden via config.yaml for better control
        if prompt is None:
            prompt = "Describe what you see: "
        
        # Encode the prompt - use proper tokenization for Gemma models
        # For Gemma models, we should use the tokenizer's encode method with proper settings
        try:
            # Try encoding with add_special_tokens=True first (for instruction-tuned models)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
        except:
            # Fallback to add_special_tokens=False if that fails
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        
        prompt_ids = prompt_ids.to(device)  # (1, prompt_len)
        prompt_len = prompt_ids.size(1)
        
        # Debug: print prompt info if needed
        if prompt_len == 0:
            warnings.warn(f"Prompt encoded to empty sequence. Prompt: '{prompt}'")
            # Fallback: use a simple prompt
            prompt = "Describe what you see: "
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
            prompt_len = prompt_ids.size(1)
        
        # Initialize with prompt tokens for each batch item
        prompt_tokens_list = prompt_ids[0].tolist()
        generated_ids = [prompt_tokens_list.copy() for _ in range(batch_size)]
        
        # Track word counts for each sequence (start with prompt word count)
        word_counts = [len(prompt.split()) for _ in range(batch_size)]
        
        # Store which sequences have finished
        finished = [False] * batch_size
        
        with torch.no_grad():
            # Step 1: Get initial logits from video features + prompt tokens
            # Create input_ids tensor with prompt for all batch items
            prompt_input_ids = prompt_ids.repeat(batch_size, 1).to(device)  # (B, prompt_len)
            
            # Forward pass with video features + prompt to get initial logits
            initial_outputs = self.forward(encoded_features, input_ids=prompt_input_ids, return_dict=True)
            initial_logits = initial_outputs['logits']  # (B, T_video + prompt_len, vocab_size) or (B, vocab_size)
            
            # Handle different logit shapes
            if initial_logits.dim() == 3:
                # (B, T, vocab_size) - use last position
                current_logits = initial_logits[:, -1, :]  # (B, vocab_size)
            elif initial_logits.dim() == 2:
                # (B, vocab_size) - use directly
                current_logits = initial_logits
            else:
                # Fallback: try to get logits from last hidden state
                if 'hidden_states' in initial_outputs:
                    hidden = initial_outputs['hidden_states']
                    if hidden.dim() == 3:
                        hidden = hidden[:, -1, :]  # (B, hidden_dim)
                    current_logits = self.output_projection(hidden)  # (B, vocab_size)
                else:
                    raise ValueError(f"Unexpected logits shape: {initial_logits.shape}")
            
            # Store accumulated input_ids starting with prompt
            accumulated_input_ids = [prompt_tokens_list.copy() for _ in range(batch_size)]
            
            # Generate tokens autoregressively
            for step in range(max_new_tokens):
                # Check if all sequences are finished
                if all(finished):
                    break
                
                # Sample next token for each batch item
                next_tokens = []
                for b in range(batch_size):
                    if finished[b]:
                        next_tokens.append(pad_token_id)
                        continue
                    
                    # Get logits for this batch item
                    logits = current_logits[b].clone()  # (vocab_size,)
                    
                    # Apply repetition penalty to prevent token loops
                    if repetition_penalty != 1.0 and len(generated_ids[b]) > len(prompt_tokens_list):
                        # Get recently generated tokens (last 10 tokens)
                        recent_tokens = generated_ids[b][-10:] if len(generated_ids[b]) > 10 else generated_ids[b]
                        for token_id in recent_tokens:
                            if 0 <= token_id < logits.size(-1):
                                if logits[token_id] > 0:
                                    logits[token_id] /= repetition_penalty
                                else:
                                    logits[token_id] *= repetition_penalty
                    
                    # Apply temperature
                    if temperature != 1.0 and temperature > 0:
                        logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        # Get top-k values and indices
                        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        # Create a mask: set non-top-k logits to very negative value
                        filtered_logits = torch.full_like(logits, float('-inf'))
                        filtered_logits.scatter_(0, top_k_indices, top_k_values)
                        logits = filtered_logits
                    
                    # Apply nucleus (top-p) sampling
                    if use_nucleus_sampling and top_p < 1.0:
                        # Sort logits in descending order
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        
                        # Calculate cumulative probabilities
                        cumprobs = torch.cumsum(sorted_probs, dim=-1)
                        
                        # Find cutoff point where cumulative probability exceeds top_p
                        sorted_indices_to_remove = cumprobs > top_p
                        # Keep at least one token
                        sorted_indices_to_remove[0] = False
                        
                        # Create mask for tokens to remove
                        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Get vocab size for filtering
                    if hasattr(self.tokenizer, 'vocab_size'):
                        max_vocab_id = self.tokenizer.vocab_size - 1
                    else:
                        max_vocab_id = logits.size(-1) - 1
                    
                    # Filter out invalid tokens (out of vocabulary range)
                    # Only consider tokens in valid range
                    token_indices = torch.arange(logits.size(-1), device=logits.device)
                    valid_mask = (token_indices >= 0) & (token_indices <= max_vocab_id)
                    
                    # Set invalid tokens to very negative value
                    logits_filtered = logits.clone()
                    logits_filtered[~valid_mask] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = torch.softmax(logits_filtered, dim=-1)
                    
                    # Additional filtering: remove tokens with very low probability
                    # This helps avoid random low-probability tokens
                    probs[probs < 1e-6] = 0.0
                    probs_sum = probs.sum()
                    if probs_sum > 0:
                        probs = probs / probs_sum  # Renormalize
                    
                    # Sample token
                    if probs_sum > 0:
                        try:
                            token_id = torch.multinomial(probs, 1).item()
                            # Ensure token ID is valid
                            if token_id < 0 or token_id > max_vocab_id:
                                token_id = logits_filtered.argmax().item()
                        except RuntimeError as e:
                            # If multinomial fails (e.g., all probs are 0), use argmax
                            token_id = logits_filtered.argmax().item()
                            if token_id < 0 or token_id > max_vocab_id:
                                token_id = 0
                    else:
                        # Fallback: use most likely valid token
                        token_id = logits_filtered.argmax().item()
                        if token_id < 0 or token_id > max_vocab_id:
                            # Last resort: use a safe default token
                            token_id = 0  # Usually corresponds to a padding or special token
                    
                    # Additional check: prevent infinite loops by checking for repeated tokens
                    if len(generated_ids[b]) > 2:
                        last_two = generated_ids[b][-2:]
                        if last_two[0] == last_two[1] == token_id:
                            # If we're about to repeat the same token 3 times, force a different token
                            # Get top-5 tokens and pick a different one
                            top_tokens = torch.topk(logits_filtered, min(5, logits_filtered.size(-1)))[1]
                            for candidate_id in top_tokens:
                                candidate_id = candidate_id.item()
                                if candidate_id != token_id and 0 <= candidate_id <= max_vocab_id:
                                    token_id = candidate_id
                                    break
                    
                    # Check for EOS token
                    if eos_token_id is not None and token_id == eos_token_id:
                        finished[b] = True
                    
                    # Check word count limit (excluding prompt)
                    # Decode current sequence to count words
                    if not finished[b]:
                        current_text = self.tokenizer.decode(
                            generated_ids[b],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )
                        # Remove prompt to count only generated words
                        if prompt and current_text.startswith(prompt):
                            generated_text = current_text[len(prompt):].strip()
                        else:
                            generated_text = current_text.strip()
                        
                        # Count words in generated text only (excluding prompt)
                        generated_words = len(generated_text.split()) if generated_text else 0
                        word_counts[b] = generated_words
                        
                        # Stop if we've reached the word limit
                        if generated_words >= max_words:
                            finished[b] = True
                    
                    next_tokens.append(token_id)
                    generated_ids[b].append(token_id)
                    accumulated_input_ids[b].append(token_id)
                
                # For next iteration, get logits using accumulated tokens
                # This maintains full autoregressive context: video features + all generated tokens
                # Convert accumulated tokens to tensor
                batch_input_ids = []
                max_len = max(len(ids) for ids in accumulated_input_ids) if accumulated_input_ids else 1
                
                for b in range(batch_size):
                    ids = accumulated_input_ids[b]
                    if len(ids) < max_len:
                        # Pad with pad_token_id for batching
                        ids = ids + [pad_token_id] * (max_len - len(ids))
                    batch_input_ids.append(ids)
                
                input_ids_tensor = torch.tensor(batch_input_ids, device=device)  # (B, seq_len)
                
                # Forward pass with video features + all generated tokens
                # The forward method concatenates: [video_features, text_embeddings]
                # So logits shape is (B, T_video + N_tokens, vocab_size) or (B, vocab_size)
                
                outputs = self.forward(encoded_features, input_ids=input_ids_tensor, return_dict=True)
                next_logits = outputs['logits']  # (B, T_video + N_tokens, vocab_size) or (B, vocab_size)
                
                # Get logits for the next token (last position in the sequence)
                # Handle different logit shapes
                if next_logits.dim() == 3:
                    # (B, T, vocab_size) - use last position
                    current_logits = next_logits[:, -1, :]  # (B, vocab_size)
                elif next_logits.dim() == 2:
                    # (B, vocab_size) - use directly
                    current_logits = next_logits
                else:
                    # Fallback: get from hidden states
                    if 'hidden_states' in outputs:
                        hidden = outputs['hidden_states']
                        if hidden.dim() == 3:
                            hidden = hidden[:, -1, :]  # (B, hidden_dim)
                        current_logits = self.output_projection(hidden)  # (B, vocab_size)
                    else:
                        raise ValueError(f"Unexpected logits shape: {next_logits.shape}")
            
            # Decode generated sequences
            generated_texts = []
            for b in range(batch_size):
                if len(generated_ids[b]) == 0:
                    generated_texts.append("")
                else:
                    # Decode the sequence, skipping special tokens
                    full_text = self.tokenizer.decode(
                        generated_ids[b],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Remove the prompt prefix if it exists
                    if prompt and full_text.startswith(prompt):
                        text = full_text[len(prompt):].strip()
                    else:
                        text = full_text.strip()
                    
                    # Ensure we don't exceed max_words by truncating if needed
                    words = text.split()
                    if len(words) > max_words:
                        text = ' '.join(words[:max_words])
                        # Remove trailing incomplete punctuation if any
                        if text and text[-1] not in '.!?':
                            # Try to end at a natural point
                            last_punct = max(
                                (i for i, c in enumerate(text) if c in '.!?'),
                                default=-1
                            )
                            if last_punct >= 0:
                                text = text[:last_punct + 1]
                    
                    generated_texts.append(text)
            
            return generated_texts
    
    def freeze_llm(self):
        """Freeze the base LLM model"""
        if self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = False
            self._llm_frozen = True
    
    def unfreeze_llm(self):
        """Unfreeze the base LLM model"""
        if self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad = True
            self._llm_frozen = False
    
    def freeze_temporal_lstm(self):
        """Freeze the temporal LSTM"""
        if self.temporal_lstm is not None:
            for param in self.temporal_lstm.parameters():
                param.requires_grad = False
            self._temporal_lstm_frozen = True
    
    def unfreeze_temporal_lstm(self):
        """Unfreeze the temporal LSTM"""
        if self.temporal_lstm is not None:
            for param in self.temporal_lstm.parameters():
                param.requires_grad = True
            self._temporal_lstm_frozen = False
    
    def freeze_projections(self):
        """Freeze feature and output projections"""
        for param in self.feature_projection.parameters():
            param.requires_grad = False
        for param in self.output_projection.parameters():
            param.requires_grad = False
        self._projections_frozen = True
    
    def unfreeze_projections(self):
        """Unfreeze feature and output projections"""
        for param in self.feature_projection.parameters():
            param.requires_grad = True
        for param in self.output_projection.parameters():
            param.requires_grad = True
        self._projections_frozen = False

