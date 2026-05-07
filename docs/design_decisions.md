# LoRA Adapter Inheritance: No Merge-and-Unload on 4-bit Weights

**Date:** 2026-05-07  
**Affects:** Phase 1 trainer.py, Phase 2-3 cannibalism via LoRA interpolation

When the base model is loaded in 4-bit quantization (BnB NF4), merging LoRA
adapter weights into the base model requires dequantization followed by
re-quantization. This process alters the numerical properties of the inherited
weights unpredictably and cannot be considered a faithful transfer of the
parent adapter.

**Rule:** Never call merge_and_unload() on a 4-bit quantized base model in
this project.

**Correct pattern for inheritance (Phase 1):**
Load base model in 4-bit, then apply parent adapter via
PeftModel.from_pretrained(..., is_trainable=True). Fine-tune from that state.
The adapter weights remain separate from the base model throughout.

**Implication for Phase 2-3 cannibalism:**
LoRA interpolation (W_new = W_strong*(1-alpha) + W_weak*alpha) must operate
on adapter weights only, never on merged model weights. Both adapters must be
loaded separately, interpolation done at the adapter tensor level, result
saved as a new adapter file.
