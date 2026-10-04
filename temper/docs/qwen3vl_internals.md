# Qwen3-VL Internals & InfiniPot-V Compatibility Record

## Local Repository Paths
- **UI_for_VLM (Target Repo)**: `F:\URCA_PROJECTS`
- **InfiniPot-V (Reference Repo)**: `F:\URCA_PROJECTS\InfiniPot-V`
- **Target Model**: `Qwen/Qwen3-VL-4B-Instruct`
- **Local Model Snapshot**: `F:\URCA_PROJECTS\.hf_cache\hub\models--Qwen--Qwen3-VL-4B-Instruct\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17`
- **Device Placement**: Single GPU (`cuda:0`), no CPU offloading observed under 4-bit BitsAndBytes quantization (`{'cuda:0'}`).

---

## Introspection Findings (Phase 0.1 Spike Results)

Resolved Model Class: `<class 'transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration'>`  
Resolved Config Class: `<class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'>`  
Resolved Processor Class: `<class 'transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor'>`

| Reference attribute (from InfiniPot-V) | Exists on Qwen3-VL? | Actual path/name if different | Notes |
|---|---|---|---|
| `model.model.language_model.layers` | **Yes** | `model.model.language_model.layers` | Exactly matches reference. Count: 36 decoder layers for Qwen3-VL-4B. |
| `model.config.video_token_id` | **Yes** | `model.config.video_token_id` | Exactly matches reference. Token ID value: `151656`. |
| `model.config.vision_start_token_id` / `vision_end_token_id` | **Yes** | `model.config.vision_start_token_id`, `model.config.vision_end_token_id` | Exactly matches reference. Start ID: `151652`, End ID: `151653`. |
| `model.get_video_features(...)` | **Yes** | `model.get_video_features(pixel_values_videos, video_grid_thw=...)` | Callable on model root. Returns `BaseModelOutputWithDeepstackFeatures`. `pooler_output[0]` shape: `[num_vision_tokens, 2560]` (`torch.bfloat16`). Also produces `deepstack_features` (list of 3 tensors of shape `[num_vision_tokens, 2560]`). |
| `model.get_rope_index(...)` | **Yes** (on `model.model`) | `model.model.get_rope_index(input_ids, mm_token_type_ids, video_grid_thw=...)` | Located on `model.model` (callable), exactly matching InfiniPot-V reference usage (`self.model.model.get_rope_index`). Qwen3-VL requires `mm_token_type_ids` as second argument. Returns `(rope_position_ids, rope_deltas)` where `rope_position_ids` shape is `[3, bsz, seq_len]`. |
| `past_key_values` type and per-layer K/V access pattern | **Yes** | `transformers.cache_utils.DynamicCache` with `.layers[i].keys` and `.layers[i].values` | Type is `DynamicCache`. Contains 36 `DynamicLayer` instances. Key tensor: `past_key_values.layers[i].keys`, Value tensor: `past_key_values.layers[i].values`. Matches InfiniPot-V reference access pattern exactly. |
| K/V tensor shape `[bsz, num_heads, seq_len, head_dim]` | **Yes** | `[1, 8, seq_len, 128]` | `bsz=1`, `num_heads=8` (GQA KV heads), `seq_len=212` (in 2-frame test), `head_dim=128`. Layout is strictly 4D `[bsz, num_heads, seq_len, head_dim]`. Dtype: `torch.bfloat16`, Device: `cuda:0`. |
| Device placement & Quantization | **Yes** | Single-GPU on `cuda:0` | Under 4-bit BitsAndBytes (`nf4`, double quant, compute dtype `torch.bfloat16`), `device_map="auto"` resolves to `{'': 0}` (`cuda:0`). All visual, language, and lm_head parameters reside on GPU. Total VRAM footprint is ~2.5 GB reserved. No CPU offloading occurs, guaranteeing zero PCIe transfer overhead during InfiniPot-V cache compression. |

---

## Additional Qwen3-VL Specific Details
1. **Processor & Input Format**:
   - `processor(...)` outputs keys: `['input_ids', 'attention_mask', 'mm_token_type_ids', 'pixel_values_videos', 'video_grid_thw']`.
   - `mm_token_type_ids` indicates multimodal vs text token positions and is required for 3D RoPE indexing in `model.model.get_rope_index`.
2. **DeepStack Visual Embeddings**:
   - Qwen3-VL uses hierarchical DeepStack features during visual forward. `video_outputs` contains `pooler_output` and `deepstack_features` which are consumed in `model.model.forward`.
3. **Video Frame Handling**:
   - Bypassing file-based video decoding with `cv2` to pass raw RGB frame arrays/PIL Images directly to `processor` cleanly circumvents Windows torchvision video reader limitations while directly aligning with `VideoChunk.frames`.

---

## Phase 0.3 Decision Gate

**Selected Option**: **Option A (port to Qwen3-VL)**

### Reasoning:
1. Every critical attribute assumed by InfiniPot-V exists with full semantic and structural parity on `Qwen/Qwen3-VL-4B-Instruct`:
   - `model.model.language_model.layers` is directly accessible (36 layers).
   - `past_key_values` is a `DynamicCache` with `.layers[i].keys` and `.values` having standard 4D shape `[bsz, num_heads, seq_len, head_dim]`.
   - `video_token_id`, `vision_start_token_id`, `vision_end_token_id` match.
   - `model.get_video_features` and `model.model.get_rope_index` exist and produce the expected 3D RoPE indices (`[3, bsz, seq_len]`).
2. There are no structural incompatibilities or architectural blocks requiring a fallback to Qwen2.5-VL. Option A proceeds with zero guesswork based on empirically verified attributes.
