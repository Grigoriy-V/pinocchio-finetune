# Why 750 tokens/s, and what would make it faster — research, 2026-09-11

Read while the v1 run trains; nothing here has been measured on our
stack except the two numbers in the first table. Every lever below is an
option for v2, to be confirmed by a 5-step smoke before it is trusted.

## What we measured

| GPU | $/h | s/step (8 samples, ~41k tokens) | tokens/s | MFU (dense bf16) |
|---|---|---|---|---|
| L40S | 1.95 | 55 | ~750 | ~40 % of 181 TFLOPS |
| A100-40 | 2.10 | 50.5 | ~820 | ~25 % of 312 TFLOPS |

Compute per sample, counted: forward 2·12B·5k = 120 TFLOP; backward
through activations ≈ 2×; the checkpointing recompute +1× → ~480 TFLOP a
sample, 3.8 PFLOP a step. A card with 1.7× the matmul throughput gave
8 %: **the run is not bound by matrix multiplication.** Something whose
cost does not scale with the card's FLOPS sets the pace.

## Where the time goes, by reading

1. **Attention without a flash kernel.** Gemma 4 12B has 30 text
   layers: 26 sliding-window (window 512, head_dim 256) and 4 global
   (head_dim 512). With `attn_implementation="sdpa"` and a sliding-window
   mask, PyTorch's SDPA cannot take its flash path (flash accepts no
   arbitrary mask); it takes the memory-efficient or the math kernel,
   which materialise or stream a 5k×5k score matrix per head per layer
   — and the 4 global layers with head_dim 512 exceed FlashAttention-2's
   256 limit, so `flash_attention_2` crashes on them and the whole model
   falls back to SDPA. transformers issue #45201 (open) asks for per-layer
   dispatch and puts the loss at "~87 % of potential FA2 speedup". At 5k
   tokens attention is quadratic while the MLP is linear, and a
   mask-driven kernel is bandwidth-bound, which is why a bigger card
   barely helps.
2. **NF4 dequantisation on every linear, three times a step** (forward,
   recompute, backward). Published numbers put 4-bit training at a
   1.2–2.3× penalty over bf16 LoRA when the model fits anyway — the
   GEMM runs in bf16 either way, dequant is extra, and it is small kernels
   with launch overhead, not big matmuls.
3. **The checkpointing recompute**: a second forward, +33 % of the
   forward/backward work, in exchange for the 12 GiB peak.
4. **The 262,144-token LM head.** TRL's chunked cross-entropy — the
   default, and the thing that avoids a 5k×262k fp32 logits tensor — is
   documented as "not compatible with PEFT", so a LoRA run takes the
   plain path. ~5–8 % of the FLOPs, more of the memory traffic.
5. **The prefix.** Of a 5.2k-token sample, the system prelude and the 17
   tool schemas are ~3k tokens, identical in all 782 samples, and each
   of the 814 calls of a run re-encodes the run's whole history. Training
   has no prefix cache; this is the largest cost of all and it is a
   property of "one sample per call".

## Levers, with what they are expected to give

| lever | change | expected | risk / cost |
|---|---|---|---|
| **A. One sample per run**, not per call | the sample is the run's last call rendered whole, loss on every assistant segment (260 sequences instead of 814) | tokens/epoch 4.0M → ~1.4M, **~3× less work** | earlier assistant segments are rendered as stored history (no empty thought channel, text after responses), slightly different from what the model saw when it generated them; the per-call split logic already knows the shapes; a converter change plus a mask per segment |
| **B. bf16 LoRA instead of NF4** | drop `quantization_config` | 1.2–2× from no dequant | 24 GB of weights: A100-80 or H100, ~40 GB with checkpointing on |
| **C. No gradient checkpointing** | `gradient_checkpointing=False` | −25–33 % | activations at 6k tokens × 30 layers: fits the 80 GB cards with B, not the 40 GB ones |
| **D. Unsloth** | its patched Gemma 4 (fused dequant, fused CE, its own attention path) | claims 1.5× over FA2 setups and 2–3× over vanilla QLoRA at 4k on H100 for 8–9B models | a second stack pinned to its own transformers; `save_pretrained_merged` replaces our merge; the smoke says whether it loads 5.14.1's Gemma 4 |
| **E. Liger kernel** | `use_liger_kernel=True` | fused CE/RMSNorm/SwiGLU, 10–20 % | Gemma 4 support to check; incompatible with chunked loss (already off under PEFT) |
| F. FA2 on the sliding layers | per-layer dispatch (issue #45201) or the Triton kernel `zzhhjjj/gemma-triton-flash-attn` | the missing 87 % of attention's flash gain | not in transformers yet; a third-party kernel is v3 material |
| G. Packing, bigger batch | — | none at batch 1 × 5k tokens: the card is already full per step | — |
| H. One epoch | halve `num_train_epochs` | 2× | 34k target tokens seen once; a quality question, not a speed one |

Combined B + C on an H100 is the plain-stack option: no dequant, no
recompute, the bigger card's bandwidth for the attention kernels —
1.5–3× over today, at $3.95/h. A alone gives ~3× on any card and also
shrinks the epoch's teacher-token count without touching a single
target token.

C is the one that does not fit, and for the same reason the run is slow:
without a flash kernel every layer keeps its 8 × 6.4k × 6.4k attention
scores, 1.3–2.6 GB a layer on top of ~1.4 GB of linear-layer activations,
30 layers → 75–120 GB for the longest sample. Not on an 80 GB card at our
lengths; C needs F first. A + B on an A100-80 (~30 GB peak, $2.50/h) is
the realistic v2: 20–40 minutes a run by arithmetic, to be measured.

## Measured the same evening (`2026-09-11_smoke_v2.md`)

A + B on an A100-40, 5 steps: 53.7 s/step at the same ~42k tokens per
step — **B gives nothing**, dequantisation is not the ceiling; **A gives
its full 3.4×** by having fewer steps; peak 28.3 GiB. Two epochs of the
run-per-sample set: 58 steps, ~52 min, ~$1.8. The remaining suspect is
the attention path (lever F).

## What to do, in order (v2, after the loop closes)

1. Two 5-step smokes on an H100, ~$0.50 each: today's recipe, and
   B + C. The ratio between them is the fact this note lacks.
2. A as a converter option (`--per-run`), tokenized and smoked the same
   way, with the D1–3/V1–3/X1–3 held-out measurement telling whether the
   history-rendering difference costs anything.
3. Unsloth only if 1 and 2 leave the run over an hour.

## Sources

- transformers issue #45201, Gemma 4 per-layer FlashAttention —
  https://github.com/huggingface/transformers/issues/45201
- Gemma 4 model doc (30 layers, sliding window 512, vocab 262,144, sdpa
  recommended) — https://huggingface.co/docs/transformers/main/en/model_doc/gemma4
- TRL, reducing memory usage (chunked CE "not compatible with PEFT",
  padding-free needs FA2, Liger, activation offloading) —
  https://huggingface.co/docs/trl/main/en/reducing_memory_usage
- TRL kernels hub (`attn_implementation="kernels-community/flash-attn2"`) —
  https://huggingface.co/docs/trl/main/en/kernels_hub
- Unsloth Gemma 4 guide and Gemma 3 blog (1.5–1.6× over FA2, 60 % less VRAM) —
  https://unsloth.ai/docs/models/gemma-4/train , https://unsloth.ai/blog/gemma3
- Unsloth vs TRL comparisons (QLoRA 2–3× at 4k on H100 for 8–9B) —
  https://www.marktechpost.com/2026/07/22/unsloth-vs-axolotl-vs-trl-vs-llama-factory-a-fine-tuning-framework-comparison-on-speed-vram-and-multi-gpu/
- 4-bit training penalty 1.2–2.3× over bf16 when the model fits —
  https://alain-airom.medium.com/run-big-llms-on-small-gpus-a-hands-on-guide-to-4-bit-quantization-and-qlora-40e9e2c95054
- Triton flash attention for Gemma 4's 256/512 head dims —
  https://github.com/zzhhjjj/gemma-triton-flash-attn
