# Roadmap

The only plan of this repository. Kept the way the harness keeps its own:
approved work only, state and order, evidence in `reports/`.

## Current state

- **Base:** `google/gemma-4-12B-it`, bf16, on the Modal Volume
  `pinocchio-tune` of the second workspace (`grigoriy98smile`); the first
  workspace's balance is out and nothing runs there.
- **Data:** v1 (782 samples, one per model call) and v2 (232, one per run)
  from the harness export of 2026-09-11, uploaded and tokenized.
- **Adapters:** `v1-r16` (SFT) and `dpo-v1` (DPO on loop pairs), both
  merged to bf16 and served (`pinocchio-tune-serve`, `-dpo`) beside
  `pinocchio-tune-serve-base` (the untuned
  base), both L40S, scaled to zero. The harness reaches them as
  `[model.sets.tuned]`, `[model.sets.dpo]` and `[model.sets.base]`.
- **Measuring stick:** `loop_live --deployed --model <set> D V X` on the
  harness's copy of `assistant-control` on the same workspace, then
  `tools/judge_pack.py` → three blind Sonnet judges → `tools/judge_unblind.py`.

## Done

- **v1, the loop closed once** (2026-09-11 → 12): converter, tokenizer
  path through Gemma's template, QLoRA on A100-40 (2 h 46 min, ≈ $8),
  merge, serving, measurement. Blind judges: GLM 9.81, untuned bf16 9.25,
  **tuned 8.67**, int4 QAT 8.15; checks 16/17 vs 15/17. The fine-tune did
  not help; its gap is two repeat loops. Closed by the human 2026-09-12.
  `reports/2026-09-12_v1_run.md`, `reports/2026-09-12_after_measurement.md`.
- **DPO on the loop, two A10s** (2026-09-12, the human's choice of the
  cards): 34 loop pairs from the exports (`tune/pairs.py`), 23 with the
  teacher's move (`tune/teach.py`), 16 after three blind judges
  (`tune/judge_pairs.py`); `modal_apps/dpo_app.py` — bf16 12B FSDP-sharded
  over `A10:2`, logits at the completion positions only, one row of a
  pair per backward, pure bf16, exact resume — 12 steps, loss 0.69 → 0.20,
  19.6 GiB peak. Measured blind beside `base` on D V X: **dpo 9.37, base
  9.47, GLM 9.70**; checks 55/57 vs 57/57. It did not stop the loop: D4
  is the same six-fold repeat (4.0), everything else level. Closed by the
  human's word 2026-09-12. `reports/2026-09-12_loop_pairs.md`,
  `reports/2026-09-12_dpo_smoke.md`, `reports/2026-09-12_dpo_measurement.md`.
- **Speed**: the L40S/A100 ceiling is the attention path, not the card or
  the quantisation; one sample per run is 3.4× fewer steps.
  `reports/2026-09-11_speed_research.md`, `reports/2026-09-11_smoke_v2.md`.

## Queue

Recorded 2026-09-12 on the human's word as the next experiments, in this
order. Each starts on his word; every Modal run is a gate.

1. **v2 recipe when training again:** one sample per run (`--per-run`,
   set v2), bf16 base (`--quant none`), checkpoints every 10 steps, 2 cores
   / 16 GiB — 58 steps, ~52 min, ~$2.3 on an A100-40.
2. **Harness first, then re-measure:** ISS-0069 (Gemma's empty answer after
   the repeat guard) and a guard that catches a cycle of several commands
   are the harness's; when they land, `base` and `tuned` are measured
   again on the same cases before any new training is judged.
3. **More D and X trajectories** (the thin families) by a second parallel
   generation run in the harness: the loop left sixteen pairs, and the DPO
   of 2026-09-12 needs more before it is trained again.

## Not started, no order

- Flash attention on the sliding-window layers (transformers issue #45201
  or the Triton kernel) — the remaining speed lever.
- Unsloth as a second stack, only if 2 leaves a run over an hour.
- Serving the product's Gemma in bf16 instead of int4 QAT — a harness and
  product decision, recorded here because the measurement found it.
