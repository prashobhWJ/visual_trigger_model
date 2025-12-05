# GPU Memory Optimizations

This document describes the memory optimization techniques implemented to reduce GPU memory usage while retaining performance.

## Summary of Optimizations

The following optimizations have been implemented:

### 1. **Mixed Precision Training (AMP)** ✅
- **Implementation**: Automatic Mixed Precision (AMP) using `torch.cuda.amp`
- **Memory Savings**: ~50% reduction in memory usage
- **Performance Impact**: Minimal (typically 5-10% slower, but allows larger batch sizes)
- **Status**: Enabled by default in `config.yaml` (`use_mixed_precision: true`)

### 2. **Gradient Checkpointing** ✅
- **Implementation**: Enabled for LLM model via `gradient_checkpointing_enable()`
- **Memory Savings**: ~30-40% reduction in activation memory
- **Performance Impact**: ~20-30% slower (trades compute for memory)
- **Status**: Enabled by default (`use_gradient_checkpointing: true`)

### 3. **Half Precision LLM Loading** ✅
- **Implementation**: Load LLM with `bfloat16` or `float16` precision
- **Memory Savings**: ~50% reduction in LLM model memory (2B model: ~4GB → ~2GB)
- **Performance Impact**: Minimal on modern GPUs (Ampere+)
- **Status**: Set to `bfloat16` by default (`dtype: "bfloat16"`)

### 4. **Maximum Frames Limit** ✅
- **Implementation**: Limit number of frames per video sample
- **Memory Savings**: Proportional to sequence length reduction
- **Performance Impact**: None (just processes fewer frames)
- **Status**: Set to 100 frames by default (`max_frames: 100`)

### 5. **Optimized Batch Size & Gradient Accumulation** ✅
- **Implementation**: Reduced batch size, increased gradient accumulation
- **Memory Savings**: Lower peak memory during forward/backward pass
- **Performance Impact**: None (effective batch size maintained)
- **Status**: `batch_size: 4`, `gradient_accumulation_steps: 8` (effective batch size = 32)

## Memory Usage Comparison

### Before Optimizations:
- **LLM Model**: ~4GB (float32, 2B parameters)
- **Activations**: ~8-12GB (depending on sequence length)
- **Total**: ~12-16GB per batch

### After Optimizations:
- **LLM Model**: ~2GB (bfloat16, 2B parameters) - **50% reduction**
- **Activations**: ~3-5GB (with gradient checkpointing) - **60% reduction**
- **Mixed Precision**: Additional ~30-40% reduction on activations
- **Total**: ~4-6GB per batch - **~70% reduction overall**

## Configuration Options

All optimizations can be controlled via `config.yaml`:

```yaml
# Data configuration
data:
  max_frames: 100  # Limit frames per video (None = no limit)

# Model configuration
model:
  llm:
    dtype: "bfloat16"  # Options: "float32", "float16", "bfloat16"
    use_gradient_checkpointing: true  # Enable gradient checkpointing

# Training configuration
training:
  batch_size: 4  # Reduced batch size
  gradient_accumulation_steps: 8  # Maintain effective batch size
  use_mixed_precision: true  # Enable AMP
```

## Recommendations

### For Maximum Memory Savings:
1. Use `bfloat16` for LLM (best balance of memory and stability)
2. Enable gradient checkpointing
3. Enable mixed precision training
4. Reduce `max_frames` to 50-75 if still running out of memory
5. Reduce `batch_size` to 2-3 if needed

### For Maximum Performance:
1. Use `float32` for LLM (if you have enough memory)
2. Disable gradient checkpointing
3. Keep mixed precision enabled (minimal performance impact)
4. Increase `batch_size` if memory allows
5. Remove `max_frames` limit

### For Balanced Approach (Current Default):
- `bfloat16` LLM dtype
- Gradient checkpointing enabled
- Mixed precision enabled
- `batch_size: 4` with `gradient_accumulation_steps: 8`
- `max_frames: 100`

## GPU Requirements

### Minimum (with all optimizations):
- **GPU Memory**: 8GB (e.g., RTX 3060, RTX 3070)
- **Batch Size**: 2-4
- **Max Frames**: 50-75

### Recommended:
- **GPU Memory**: 16GB (e.g., RTX 3080, RTX 4080, A100)
- **Batch Size**: 4-8
- **Max Frames**: 100-150

### Optimal:
- **GPU Memory**: 24GB+ (e.g., RTX 3090, A100 40GB)
- **Batch Size**: 8-16
- **Max Frames**: 200+

## Troubleshooting

### Out of Memory (OOM) Errors:

1. **Reduce batch size**: Decrease `batch_size` in config
2. **Increase gradient accumulation**: Increase `gradient_accumulation_steps` to maintain effective batch size
3. **Reduce max_frames**: Lower `max_frames` value
4. **Use float16 instead of bfloat16**: If your GPU doesn't support bfloat16 well
5. **Disable gradient checkpointing**: Only if you have enough memory (will use more memory)

### Performance Issues:

1. **Disable gradient checkpointing**: If you have enough memory
2. **Use float32**: If memory allows (better numerical stability)
3. **Increase batch size**: If memory allows
4. **Remove max_frames limit**: If memory allows

## Notes

- **bfloat16 vs float16**: bfloat16 is recommended for modern GPUs (Ampere architecture and newer) as it has better numerical stability while maintaining memory savings
- **Gradient Checkpointing**: This technique recomputes activations during backward pass instead of storing them, trading ~20-30% compute time for ~30-40% memory savings
- **Mixed Precision**: Automatically uses float16 for operations where it's safe, keeping float32 for operations that need precision
- **Effective Batch Size**: `batch_size * gradient_accumulation_steps` = effective batch size for training

