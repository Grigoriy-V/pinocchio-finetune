# Roadmap

The only plan of this repository. Kept the way the harness keeps its own:
approved work only, state and order, evidence in `reports/`.

## Current state

- **Base:** `google/gemma-4-12B-it`, bf16, on the Modal Volume
  `pinocchio-tune` of the second workspace (`grigoriy98smile`); the first
  workspace's balance is out and nothing runs there.
- **Data:** v1 (782 samples, one per model call) and v2 (232, one per run)
  from the harness export of 2026-09-11, uploaded and tokenized.
- **Adapter:** `v1-r16` trained 2026-09-12, merged to bf16; served by
  `pinocchio-tune-serve` beside `pinocchio-tune-serve-base` (the untuned
  base), both L40S, scaled to zero. The harness reaches them as
  `[model.sets.tuned]` and `[model.sets.base]`.
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
- **Speed**: the L40S/A100 ceiling is the attention path, not the card or
  the quantisation; one sample per run is 3.4× fewer steps.
  `reports/2026-09-11_speed_research.md`, `reports/2026-09-11_smoke_v2.md`.

## Queue

Recorded 2026-09-12 on the human's word as the next experiments, in this
order. Each starts on his word; every Modal run is a gate.

1. **The loop, taught by preference.** Pairs on the state where a Gemma
   run repeated an already-seen call: `rejected` = the repeat, `chosen` =
   GLM's move in the same state (or its answer). DPO or KTO on the same
   LoRA shape, TRL, A100-40, reference model in NF4 beside the policy.
   Only where nothing changed between the repeats. Measured the same way,
   `base` beside it. Needs first: a pair extractor over the export (the
   Gemma loops of V1, V6, X2, D4, D5 and GLM's runs of the same cases).
   Sketch in `reports/2026-09-12_after_measurement.md`.
2. **v2 recipe when training again:** one sample per run (`--per-run`,
   set v2), bf16 base (`--quant none`), checkpoints every 10 steps, 2 cores
   / 16 GiB — 58 steps, ~52 min, ~$2.3 on an A100-40.
3. **Harness first, then re-measure:** ISS-0069 (Gemma's empty answer after
   the repeat guard) and a guard that catches a cycle of several commands
   are the harness's; when they land, `base` and `tuned` are measured
   again on the same cases before any new training is judged.
4. **More D and X trajectories** (the thin families) by a second parallel
   generation run in the harness, if 1 needs them.

## Not started, no order

- Flash attention on the sliding-window layers (transformers issue #45201
  or the Triton kernel) — the remaining speed lever.
- Unsloth as a second stack, only if 2 leaves a run over an hour.
- Serving the product's Gemma in bf16 instead of int4 QAT — a harness and
  product decision, recorded here because the measurement found it.
