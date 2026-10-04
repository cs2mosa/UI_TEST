"""
Flash-attention monkey-patch for 3D position_ids (Qwen3-VL / HERMES).

Applied at import time by hermes_engine.py.  If flash-attention is not
installed (e.g. when using standard SDPA/eager attention with a 4-bit
quantized model), the patch is silently skipped.
"""
try:
    import torch
    import transformers.modeling_flash_attention_utils as _fa_utils

    _target = "prepare_fa_kwargs_from_position_ids"
    if hasattr(_fa_utils, _target) and not hasattr(_fa_utils, "_original_prepare_fa_kwargs"):
        _fa_utils._original_prepare_fa_kwargs = getattr(_fa_utils, _target)

        def _patched_prepare_fa_kwargs(position_ids, attention_mask=None):
            if position_ids is not None and position_ids.dim() == 3:
                batch_size = position_ids.shape[1]
                seq_len = position_ids.shape[2]
                device = position_ids.device
                cu_seq_lens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=device,
                )
                max_length = seq_len
                return (cu_seq_lens, cu_seq_lens), (max_length, max_length)
            return _fa_utils._original_prepare_fa_kwargs(position_ids, attention_mask)

        setattr(_fa_utils, _target, _patched_prepare_fa_kwargs)
except Exception:
    pass  # Flash-attention not installed; standard SDPA path is used instead

