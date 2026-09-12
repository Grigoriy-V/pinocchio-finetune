# Fine-tuning Gemma 4 12B on my own agent's trajectories — a case study

*September 11–12, 2026. One person, one agent harness, one open model, two
days, about $30 of GPU time, two rounds. Round one asked whether imitation
data from a stronger model makes a smaller open model better at agentic
work: no, and the reasons why are the useful part (§1–10). Round two
asked whether preference pairs on the failure itself do better, and did
it as distributed training — a 12B model in bf16 sharded over two 24 GB
A10s, because it does not fit one (§11–14). Also no, with a different
lesson, and the sharded recipe is the part that stands.*

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
repeat" where a repeat after a change is right. That is round two.

## 11. Round two: pairs on the loop itself

The material was already in the exports: every turn where a Gemma had
re-issued a call it had already made. The extractor (`tune/pairs.py`)
takes a repeat only where *nothing had changed* — evidence, not a guess:
either the harness's repeat guard refused the call, or the call ran
again and returned the same text (the runner's timing stamp aside).
`python3 stats/total.py` after the edit that fixes it is a normal
re-run, not a loop, and 45 of 127 repeats were that. The state the model
saw is the prompt; the repeat is `rejected`; at most ten pairs a run, or
one 92-call cycle would have been the whole set. 34 pairs from 7 runs.

`chosen` came from the teacher on the very same state, with the same
tools, temperature 0 (`tune/teach.py`). Where GLM made the same call the
student had — eleven of 34 — the pair was dropped: the state confuses
everyone, and the pair would teach nothing. Then three blind Sonnet
judges saw each remaining pair as two anonymous moves from one state and
were asked two things: is this state a dead end, and which move leaves
it. Seven pairs where the judges called both moves the same — `find …
| wc -l` against `find …`, a third `cat` of the same script — were
dropped; no judge preferred the student's move anywhere; sixteen pairs
remained, thirteen after the 12k-token ceiling. Small, and honest about
it: it is what the loop had left in the data.

## 12. Distributed training on two A10s: what it took

The choice was the human's, and deliberate: not an A100, not four cards,
but two A10s at $1.10 an hour. A 12B model in bf16 is ~24 GB of weights;
an A10 has 24 GB. On one card the bf16 policy does not fit at all, so
sharding is a real need rather than an acceleration — that is the
honest version of "distributed training" for a portfolio. FSDP full
shard over `gpu="A10:2"` in one Modal container, torchrun with one
process per card, a fresh LoRA of round one's shape as the policy and
the same model with the adapter disabled as the reference, so no second
copy of the weights is ever loaded.

The smoke took seven starts, each two steps, each falling on one thing,
each fixed from the log before the next (the human's rule: minutes of
smoke, never a debugging budget the size of training):

1. TRL's DPO trainer had been rewritten; two config fields no longer
   existed. The trainer was then read in full rather than guessed at.
2. TRL turns the model's own gradient checkpointing on by default;
   under FSDP the checkpointing is FSDP's, and transformers refuses both.
3. Out of memory in the reference pass, 20 GB held: the policy's
   activations were still alive. The reference pass goes first now,
   without gradients, and its memory is back before the policy runs.
4. Out of memory in the backward, 21 GB: both rows of a pair in one
   graph. Now the reference and the policy score both rows without
   gradients, which gives the margin and the derivative of the sigmoid
   DPO loss with respect to each row's log-probability (±β·σ(−margin));
   then each row goes forward and backward alone with that derivative
   as its weight. Same gradient, one row's activations at a time.
5. A bf16 gradient into an fp32 leaf: PEFT upcasts LoRA weights to fp32
   under a bf16 base. Created the adapter in the base's dtype, as TRL
   itself does under ZeRO-3. Necessary, not sufficient.
6. The same error: accelerate upcasts every trainable parameter to fp32
   whenever FSDP runs with mixed precision, whatever dtype the adapter
   was created in. Mixed precision off; the model and the adapter are
   bf16 already, and AdamW on a 65M-parameter LoRA in bf16 is fine for
   twelve steps.
7. Ran: 18.8 GiB peak on a 24 GB card.

The largest single memory item was never the weights. Gemma's
vocabulary is 262k tokens; a 7k-token pair through the stock trainer
means ~7 GB of logits per sequence in bf16, several times over. Gemma's
forward takes `logits_to_keep`, a tensor of positions, and the loss only
needs the few dozen positions that predict a completion token. That one
argument is the difference between "does not fit on four A10s" and "fits
on two".

The full run went in two halves. The first was cancelled at step five by
the operator's own tool: `modal run` held in a foreground command with a
ten-minute ceiling, the client killed, and an ephemeral app stops with
its client. Checkpoints every 20 % of the steps and `--resume` (adapter,
optimizer, scheduler, data order) made the second half start from
checkpoint 4 and reproduce step five to the digit: loss 0.4094, margin
0.93. Twelve steps in all, 19.6 GiB peak, ~16 minutes of two A10s, about
a dollar. Loss 0.69 → 0.20, margins 0.8–3.5 from the fourth step on: the
adapter separates the training pairs.

## 13. Result of round two

Merged, served as a third endpoint beside the untuned base on the same
card and stack, measured on the same 17 cases, the base measured again in
the same deploy (the harness had changed under it), one blind batch of 43
transcripts with GLM's held-out runs as the anchor.

| model | checks | judges, mean of 10 | a | b | c | d | e |
|---|---|---|---|---|---|---|---|
| GLM 5.3 Flash (anchor) | 9/9 | **9.70** | 2.00 | 1.74 | 1.96 | 2.00 | 2.00 |
| Gemma-IT bf16, untuned | 57/57 | **9.47** | 1.94 | 1.67 | 2.00 | 1.86 | 2.00 |
| Gemma-IT bf16 + DPO | 55/57 | **9.37** | 1.84 | 1.71 | 1.98 | 1.88 | 1.96 |

Judges within a point on 40 of 43. Where the two Gemmas differ by more
than a point it is one case each way: D2 (DPO 9.7, base 8.0) and D4 (DPO
4.0, base 7.7).

**Plus.** The half of the lesson that took: after the guard, the DPO
model answers in words — D4 ended with a right, grounded answer ("the
script fails because of a trailing comma") where round one's Gemmas
answered nothing. Three of the sixteen `chosen` completions were exactly
that kind of answer. And V6, round one's 92-call cycle, was three calls
for base and DPO alike.

**Minus.** The other half did not take. D4 is the loop, on a prompt that
was in the training pairs: the model read the files, saw the comma, and
ran the same `python3 -c "json.loads('{…}')"` six times — it printed `3`
every time — until the guard refused it. The `rejected` completions were
exact repeats and this is an exact repeat, so the signal was the right
one; sixteen pairs, three epochs, β 0.1 at rank 16 did not move it. On
the whole set the DPO model is level with the base and a tenth below.

**The measurement itself had to be done twice.** The two `scenarios`
calls were started at once under the harness's default probe user, so
both wrote into the same threads; the DPO run was a case ahead and in 12
of 17 cases the base's first model call already carried the DPO's
finished turn ("I have already completed this task for you", no tool
call, eleven failed checks). The suite's `--parallel` exists for exactly
this. The base was rerun alone and the batch judged again; the DPO and
GLM scores of the first batch sit within a tenth of the second's.

## 14. What round two leaves

1. **A working recipe for a 12B in bf16 on two 24 GB cards**, with the
   three ideas that made it fit: logits only where the loss needs them,
   one row of a pair per backward with the loss's analytic weight, and
   no mixed precision when everything is bf16 already. Exact resume from
   a checkpoint. About two minutes per step of two pairs.
2. **Sixteen pairs is a pilot, not a training set.** The extractor, the
   teacher and the judges take any number of exports; the harness's
   parallel generation is the way to fifty.
3. **The loop is still the harness's to catch first.** D4's six
   identical calls passed two guard-free repeats before the third was
   refused; a guard that counts an identical *successful* call from the
   second would have cut it to two, for every Gemma, tuned or not.
4. **Two rules for the operator**, learned at ~$1 each: a local timer
   must never hold a training or inference run (`--detach`, detached
   processes, polling); and concurrent measuring runs need their own
   probe users.

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
| total GPU and workers, round one | ≈ $25 across two Modal workspaces |
| round two: loop pairs | 127 repeats → 34 pairs → 23 taught → 16 judged → 13 under the ceiling |
| round two: training | 2 × A10, FSDP, 12 steps, 19.6 GiB peak, ~16 min, ≈ $1 (+ 7 smokes ≈ $1.5) |
| round two: measurement | 2 × 17 turns (+ 17 redone), 43 transcripts, 3 blind judges, ≈ $3 |
| round two, all in | ≈ $6 |
| lines of code here | round one ~1,000; round two: pairs/teach/judge ~600, DPO apps ~450, tests ~250 |

The harness side of the story — the trajectory capture, the seven
families, the judge tooling, the two issues — is in the harness's
repository and its report `reports/2026-09-11_gemma_finetune_experiment.md`.
