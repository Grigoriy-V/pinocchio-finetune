# pinocchio-finetune

A LoRA of **Gemma 4 12B** on trajectories from my own agent harness
([pinocchio-harness](https://github.com/Grigoriy-V/pinocchio-harness)),
where GLM 5.3 Flash is the teacher, measured before and after on the same
scenario suite with blind LLM judges. An experiment for experience and
portfolio: the harness keeps its hosted model, the result is a table.

`ROADMAP.md` is the plan; `reports/` the evidence. The v1 loop closed
2026-09-12: the LoRA did not make the model more agentic (blind judges:
untuned 9.25, tuned 8.67, GLM 9.81 of 10); the next experiments are in the
roadmap.

## The loop

```
harness ──export──▶ data/export ──convert──▶ data/sft/v1/train.jsonl
                                                  │  modal volume put
                                                  ▼
             /vol/data/v1 ──tokenize──▶ tokenized ──train (L40S)──▶ /vol/runs/<run>/adapter
                                                                        │ merge (CPU)
                                                                        ▼
harness ◀──[model.sets.tuned]──── serve_app (vLLM, L40S) ◀──── /vol/runs/<run>/merged
```

| step | where | command | cost |
|---|---|---|---|
| convert | this machine | `python -m tune.convert --export ../local-multimodal-agent/data/export --out data/sft/v1 --ignore-check "the answer does not claim five passed"` | none |
| upload | Modal Volume | `modal volume put pinocchio-tune data/sft/v1/train.jsonl /data/v1/train.jsonl` | none |
| fetch base | Modal CPU | `modal run modal_apps/train_app.py::fetch_base` | cents |
| tokenize | Modal CPU | `modal run modal_apps/train_app.py::tokenize --set v1` | cents |
| smoke | Modal L40S | `modal run modal_apps/train_app.py::train --set v1 --run smoke --max-steps 5` | ~$0.30 |
| train | Modal L40S | `modal run modal_apps/train_app.py::train --set v1 --run v1-r16` | ~$3–5 |
| merge | Modal CPU | `modal run modal_apps/train_app.py::merge --run v1-r16` | cents |
| serve | Modal L40S | `modal deploy modal_apps/serve_app.py` | $1.95/h while measured |
| measure | harness | `loop_live --deployed --model tuned <letters>` | the harness's |

Every Modal command is a gate: it starts a priced worker and runs only on
the human's word for that command.

## Data

`tune/convert.py` turns the export into one sample per model call —
`prompt` (the messages the model saw), `completion` (the assistant message
it produced), `tools` (the schemas it had) — in the conversational
prompt-completion shape TRL takes. It is a rename of fields, never a
reconstruction: text parts are joined, tool-call arguments become JSON
strings, a tool message gets the name of the call it answers, a call that
carried media is dropped. The filter is the harness report's: the teacher's
runs, not held out (D1–3, V1–3, X1–3 are the measuring set), checks
passed, answer delivered. A check dropped after the run is passed with
`--ignore-check`.

v1 set (export of 2026-09-11): 260 runs, 814 samples, 4.37M teacher tokens
per epoch (only 33k of them completion tokens: an agent's turns are short
tool calls), mean 5.4k tokens per sample, longest ~8k.

## Training

`modal_apps/train_app.py`. Base `google/gemma-4-12B-it` in bf16, loaded
NF4 (QLoRA); LoRA r=16, α=32, dropout 0.05 on every linear of the language
model only (the vision and audio towers stay frozen and are kept, so the
merged checkpoint still takes images); lr 2e-4 cosine, warmup 5 %, two
epochs, batch 1 × accumulation 8, sequences up to 12,288 tokens, loss on
the completion only. `tokenize` renders each sample through Gemma's chat
template on CPU first and checks that the prompt is a prefix of
prompt+completion, so a template surprise costs cents, not GPU minutes.

Why not the QAT int4 checkpoint the harness serves: a LoRA over a
compressed-tensors base has a known vLLM correctness bug at rank 32
([vllm #50059](https://github.com/vllm-project/vllm/issues/50059)), and the
NF4 base QLoRA trains over is a different quantization anyway. The merge
is bf16 (24 GB) and serves on an L40S.

Prices (Modal, 2026-09-11): L40S $1.95/h, A100-40 $2.10/h, A100-80
$2.50/h, H100 $3.95/h. Estimate for v1: 4.4M tokens × 2 epochs on an
L40S at ~1.5–2.5k tokens/s is 1–1.5 h, $2–3, plus the smoke run.

## Measuring

The harness measures: `loop_live --deployed --model tuned` on the held-out
families and the rest, trajectories captured, judged by three blind Sonnet
judges through the harness's `tools/judge_pack.py` against the baseline
runs of GLM and untuned Gemma. The rubric and the baselines are in the
harness's `reports/2026-09-11_gemma_finetune_experiment.md`.

## State

- Modal Volume `pinocchio-tune`: `/hf` (base weights), `/data/<set>`,
  `/runs/<run>/{adapter,merged,train.json}`.
- Modal secret `huggingface` with `HF_TOKEN` (Gemma is gated).
- `reports/`: one file per training run and per decision.
