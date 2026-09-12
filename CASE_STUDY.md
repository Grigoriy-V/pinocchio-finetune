# Fine-tuning Gemma 4 12B on my own agent's trajectories — a case study

*September 11–12, 2026. One person, one agent harness, one open model, two
days, about $25 of GPU time. The question was whether imitation data from
a stronger model makes a smaller open model better at agentic work. The
answer was no, and the reasons why are the useful part.*

## 1. Setting

I run a personal multimodal assistant on a harness I built myself
([pinocchio-harness](https://github.com/Grigoriy-V/pinocchio-harness)):
a LangGraph loop over an OpenAI-compatible model, seventeen tools (files,
commands in a sandbox, a browser, web search, memory, history, a goal),
deployed serverless on Modal with a Telegram front end and a Postgres
store. The production model is a hosted GLM 5.3 Flash; Gemma 4 12B sits
beside it as a self-hosted alternative on a scale-to-zero GPU.

The harness had grown a scenario suite: scripted tasks with seeded files
and *outcome* checks (the answer gives 60; the file was not changed;
fewer than five files were made), never route checks (which tool ran
when), because a check that expects a route the prompt did not ask for
is a crutch, not a measurement. It also had a rubric for an LLM judge —
five criteria, 0–2 each: (a) did what was asked, (b) no wasted moves,
(c) honest about what it saw, (d) handled a wrong turn, (e) a clear
answer — and a rule that judging is done by blind subagents, several at
once, never by the agent that ran the turn.

Gemma's known weakness in this harness had one shape: when a tool's
result surprised it, it re-ran the same command until a repeat guard
ended the turn. GLM did not do that. The experiment: capture GLM's turns,
fine-tune Gemma on them, measure before and after.

## 2. The contract between the two repositories

Training does not belong in a product harness: its dependencies must not
enter the worker image, its failed runs must not fill the product's
records. So the loop was split. The harness owns what is a property of
the system — the trajectory capture, the scenario suite and checks, the
judge's rubric, the model sets in its config — and hands over one thing:
an export of trajectories, each with its outcome and check results
attached. This repository takes that export, filters, converts, trains,
merges, serves, and hands back one thing: an endpoint the harness lists
as a model set and measures with the same suite.

## 3. Capturing the data

A trajectory is every model call of a turn exactly as the model saw it:
the messages, the tool schemas, the completion. The harness writes one
JSONL file per run on its Volume. A first count of what the deployed
store already held gave ~40 usable turns, because the scenario runner
resets its threads — so data is captured at run time, not mined
afterwards.

Seven scenario families were written for the data (after reading what
moved 7–13B models elsewhere: FireAct's 500 trajectories, SWE-Gym's 491,
the τ-bench and TOUCAN lineage): a wrong turn on the way (D), a listing
to reason over (L), numbers to compute (N), a task with a trap (T), a
change across files (U), a tool that lies back (V), tasks long enough to
need a goal (X). 38 prompts with variants; D1–3, V1–3, X1–3 held out and
never trained on. Eight containers ran GLM in parallel at temperature
0.7 — the product samples at 0, but repeats at 0 are identical — for
304 turns in half an hour; 292 passed their checks.

Export: 447 runs; 260 GLM runs kept by the filter (teacher model, not
held out, checks passed, answer delivered); **782 SFT samples**, one per
model call. 4.04M tokens, of which **33.9k are targets** — 41 per sample.
An agent's turn is a short tool call on a long context; the loss sees
almost nothing, and training loss is correspondingly noisy.

## 4. What Gemma's chat template taught before any GPU ran

The converter is a rename of fields, not a reconstruction — but the
sample must be *what the model sees at inference* plus *what it should
emit*, and Gemma 4's template (the July 2026 revision that "fixed
tool-calling loops, turn closures and thinking content-ordering") made
that non-trivial. A CPU step renders every sample through the real
tokenizer and reports what it rejects, so each surprise cost cents:

1. Tool-call arguments must be a mapping; the template raises on a JSON
   string. 814 of 814 rejected on the first run.
2. With thinking off, the generation prompt ends in an empty thought
   channel `<|channel>thought\n<channel|>` that a stored assistant turn is
   never rendered with. The naive "prompt is a prefix of prompt+completion"
   check failed on every first call. The sample is now the prompt as
   rendered with the generation prompt, plus the target cut from the full
   render after `<|turn>model\n`.
3. The stop tokens are `<eos>`, `<turn|>` and — after a tool call —
   `<|tool_response>`. A first version cut the trailing `<|tool_response>`
   from call targets; that would have taught the model never to stop
   after a call.
4. When an assistant message carries text *together with* a tool call,
   the template renders call → tool response → text and then closes the
   model turn. The next request to Gemma therefore starts from a closed
   turn with no generation prompt at all. 32 samples had no target
   consistent with inference and were dropped — and this is an
   inference-time property of the harness's own Gemma endpoint, recorded
   for the harness rather than fixed here.

Two hundred lines of segmentation logic, unit-tested against the
template read from the model's own files, before a dollar of GPU.

## 5. Training

QLoRA: `google/gemma-4-12B-it` loaded NF4, LoRA r=16 α=32 dropout 0.05
on the language model's q/k/v/o/gate/up/down projections only (65.6M
trainable parameters; the vision and audio towers frozen and kept, so
the merged checkpoint still takes images), lr 2e-4 cosine with 5 %
warmup, two epochs, batch 1 × accumulation 8, sequences up to 12,288
tokens, loss on the target tokens only via a pre-tokenized mask. TRL +
PEFT, transformers 5.14.1, pinned to the pair the harness had validated
for Gemma 4 in vLLM.

A 5-step smoke on an L40S closed every open risk (versions, the
multimodal model class with a 4-bit language stack, the LoRA regex, the
fit: 12.2 GiB peak) and produced the one number the plan had wrong: 55 s
per step, ~750 tokens/s, against an estimate of 1.5–2.5k.

The full run: A100-40, 196 steps, **2 h 46 min, ≈ $8** — $5.9 of it GPU
and $2.4 the 8 cores and 64 GiB I had reserved beside it without
thinking (now 2 cores / 16 GiB). Loss 1.47 → 0.17, target-token accuracy
0.83 → 0.95, grad norm mostly 2–20 before clipping with single spikes to
1104 that led nowhere.

## 6. Why it was slow, and what would not have helped

An A100-40 gave 8 % over the L40S for the same step. A bf16 base instead
of NF4 gave nothing. The compute per step (3.8 PFLOP) was 25 % of the
A100's peak. What was left, by reading: Gemma 4 12B's 26 sliding-window
layers and 4 global layers with head_dim 512 leave `sdpa` without a flash
path (FlashAttention-2 crashes on head_dim > 256; transformers issue
#45201 puts the loss at ~87 % of the attention speedup), so attention is
quadratic and bandwidth-bound at 5k tokens — and that also rules out
turning gradient checkpointing off, because without a flash kernel every
layer's score matrices (8 × 6.4k × 6.4k) would need 75–120 GB.

What did help: **one sample per run instead of per call.** A run's
sample is the same ~5.2k tokens as a call's, so a step costs the same,
but there are 3.4× fewer of them: 58 steps, ~52 minutes, ~$2, for the
same targets minus 6k tokens. Measured by a 5-step smoke, not run in
full, because the loop had to close once before any polish.

## 7. Serving and measuring

The merge folds the adapter into bf16 weights (24 GB) on a CPU
container; vLLM serves them on an L40S behind Modal's proxy auth with
the same Gemma flags the harness uses (`gemma4` tool and reasoning
parsers). The same App with an environment flag serves the untuned base
from the same Volume, so the "before" and "after" endpoints differ in
nothing but the weights. The harness lists both as model sets.

Three things went wrong on the way to a number, each caught by a fact
rather than a feeling:

- The harness's scenario runner applied `--model` only on its deployed
  path; two "tuned" runs answered in 8 seconds from a cold server. They
  were GLM. No container had woken — the owner noticed before I did.
- The local Windows run of the harness kills every Cygwin tool inside its
  command sandbox (`CreateFileMapping … Win32 error 5`), and the scenarios
  are written for a Linux worker. Five of six "failures" in the first real
  tuned run measured the sandbox, not the model. So the harness itself
  was deployed to the training workspace and the measurement redone on
  Linux workers.
- The base App read its target from an environment variable at import,
  inside a container that did not have it, and served the tuned weights
  under the tuned name. A 404 on the base name told the story.

The measurement: D, V and X — 17 cases each (the 9 held-out plus the
variants that had been training prompts) — at temperature 0, one sample
per case, both endpoints in parallel. The teacher's and the product's
int4-Gemma runs of the same held-out cases from the day before went into
the same batch: 52 transcripts, anonymised and shuffled, no model name,
no run id, no system prompt; three Sonnet judges each scored all 52 on
the rubric; the key was read only afterwards.

## 8. Result

| model | n | checks | judges, mean of 10 | a | b | c | d | e |
|---|---|---|---|---|---|---|---|---|
| GLM 5.3 Flash (teacher) | 9 | 9/9 | **9.81** | 2.00 | 1.81 | 2.00 | 2.00 | 2.00 |
| Gemma-IT bf16, untuned | 17 | 16/17 | **9.25** | 1.88 | 1.45 | 2.00 | 1.92 | 2.00 |
| Gemma-IT bf16 + LoRA v1 | 17 | 15/17 | **8.67** | 1.84 | 1.31 | 1.88 | 1.75 | 1.88 |
| Gemma-IT int4 QAT (the product's endpoint) | 9 | 7/9 | **8.15** | 1.78 | 1.22 | 1.78 | 1.59 | 1.78 |

Held-out cases only: untuned 9.1, tuned 8.2, int4 8.6, GLM 9.8. Judges
agreed within a point on 47 of 52; the disagreements were 8–10 splits
over an unnecessary `set_goal` or a global `pip install`, none over a
loop.

The tuned model's entire deficit is two turns. V1 (a build script that
prints "written" and writes nothing): `cd tools_task && sh build.sh &&
ls -la`, again and again, until the guard — then no answer at all. V6 (a
cleanup script that says "removed 4 files" and removes nothing): 92
calls cycling `find`, `cat`, `ls` for six minutes, until the harness's
turn check asked whether it was making progress and it answered, in one
line, correctly. Outside those two the tuned model scores what the base
scores, and on D4/D5 it does better (the base edited a config the task
said not to touch).

**The fine-tune did not make Gemma more agentic, and could not have.**
The behaviour to fix is a repeat loop. The teacher never loops, so its
trajectories contain the *absence* of a loop and never the moment of
resisting one; the guard's own message ("this exact call has already
succeeded twice…") occurs in almost no training prompt. Supervised
imitation cannot teach a correction it never sees. 34k target tokens
seen twice moved the loss from 1.47 to 0.17 and the behaviour not at all.

## 9. What was worth more than the result

1. **Quantisation cost more than fine-tuning gained.** The untuned bf16
   base scores a full point above the int4 QAT checkpoint the product had
   been serving (9.25 vs 8.15), with 16/17 against 7/9 on the checks. The
   cheapest improvement to the product's Gemma was never a fine-tune.
2. **One harness defect zeroes every Gemma.** After the repeat guard, the
   request goes out with no tools; Gemma emits 16–171 tokens; the harness
   receives no text and no call; the person receives nothing. Tuned V1
   and int4 X2 both scored 0.0 for that reason. Fix it in the harness and
   the floor rises for all three.
3. **The guard misses cycles.** V6's 92 calls were three commands in
   rotation; the exact-repeat guard never fired. A harness property, not a
   model one.
4. **Measure in the environment the baselines were measured in**, or
   measure both sides in the same new one. Windows sandbox failures
   looked exactly like model failures until the trajectories were read.
5. **Render the training sample the way inference renders it.** Every
   template surprise above would have trained a model that could not
   stop, or that expected a state it never sees. A CPU tokenizer pass
   that reports what it drops is the cheapest test in the pipeline.
6. **Count the reservations.** CPU and memory beside the GPU were 40 % of
   the training bill.

## 10. What would address the actual problem

Preference data on the loop itself: pairs on the state where Gemma
repeated an already-seen call — `rejected` the repeat, `chosen` the
teacher's move in the same state — trained with DPO or KTO on the same
LoRA shape. The raw material exists (every Gemma V1/V6/X2 loop, and the
teacher's clean runs of the same deterministic scenarios); a pair
extractor is the missing piece, with the care not to teach "never
repeat" where a repeat after a change is right. Queued, not started.

## Numbers

| | |
|---|---|
| trajectories generated | 304 turns in ~30 min, 8 parallel workers |
| training set | 782 samples / 4.04M tokens / 33.9k targets |
| trainable parameters | 65,568,768 (r=16) |
| training | A100-40, 196 steps, 2 h 46 min, ≈ $8 |
| smoke runs | 3 × 5 steps, ≈ $1 |
| serving | bf16 on L40S, cold start 2–3 min, $1.95/h while measured |
| measurement | 2 × 17 turns on Linux workers, ≈ $2; 3 blind judges, 52 transcripts |
| total GPU and workers | ≈ $25 across two Modal workspaces |
| lines of code here | converter + tests ~450, Modal apps ~550 |

The harness side of the story — the trajectory capture, the seven
families, the judge tooling, the two issues — is in the harness's
repository and its report `reports/2026-09-11_gemma_finetune_experiment.md`.
