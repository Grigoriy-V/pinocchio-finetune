# Loop pairs: the data for item 1 (2026-09-12)

The first step of the fine-tune queue's item 1 (DPO on the loop, two A10s,
FSDP), the part that needs no GPU: preference pairs from the harness's
exports, and the command that fills `chosen` with the teacher's move.

## What a pair is

A student run (Gemma int4, `base`, `tuned`) is read call by call. Where the
model re-issued a call it had already made in the same turn — the same
tool, the same arguments — and nothing had changed in between, the
messages it saw at that call are the prompt and the repeat is `rejected`.

"Nothing changed" is evidence, not a guess, in either of two forms:

- the harness's repeat guard refused the call (its message says the
  earlier result stands), or
- the call ran again and returned the same text as the first time. The
  command runner's timing stamp on its first line (`exit code: 0   (2.8
  s)`) is not a change.

A repeat whose result differs is dropped: `python3 stats/total.py` after
the edit that fixes it is the normal way to work, not a loop. Of 127
repeats in the 55 student runs, 45 were that.

`chosen` is the teacher's reply on the very same prompt with the same
tools, temperature 0 (`tune/teach.py`, OpenRouter, GLM 5.3 Flash). A pair
where the teacher makes the same call the student did is dropped: the
state is then not one the teacher resists either, and the pair would teach
nothing. So is an empty reply.

This is the item's sketch made literal: the earlier note's "GLM's move in
the same state (or its answer)" becomes GLM's actual move on that state,
asked for, because GLM's own runs of the same case never pass through the
student's looped state.

## The set, v1

`tune.pairs --export data/export --export data/export3 --per-run 10`
(the harness's two exports; a run in both is read once):

| | |
|---|---|
| runs seen / student runs | 481 / 55 |
| repeats seen | 127 |
| dropped, result changed | 45 |
| dropped, over the per-run cap of 10 | 48 |
| pairs | **34** from 7 runs |
| by case | V1 16, V6 10, X2 5, D1 1, D2 1, N1 1 |
| by student | tuned 18, int4 14, base 2 |
| by evidence | same result 27, guard 7 |
| prompt tokens to teach (estimate) | ~130k, longest prompt ~8k |

The cap matters: without it V6 alone (the tuned model's 92-call cycle)
gives 58 of 82 pairs. Ten from one run is already generous; the same
prompt with two more tool exchanges appended is nearly the same sample.

Honest about the size: 34 pairs, of which the teacher will keep some
fraction, is a small preference set. It is what the loop contains today.
Item 4 of the queue (more D and X trajectories from the harness) is the
way to more, and the extractor takes any number of exports.

## Checks

`tests/test_pairs.py` (seven): a repeat with the same result is a pair
with the state as prompt and the letter from the thread id; the guard's
refusal counts; a changed result is not a loop and the teacher's runs are
skipped; the per-run cap and a run in two exports read once; the timing
stamp; the teacher's reply filling `chosen`, a repeating or silent teacher
dropping the pair, and a rerun asking nothing; the reply's shape. Suite 23
passed.

## Cost so far

None: no model was called and no worker started. The next step, teaching
the 34 pairs, is ~130k input tokens of GLM 5.3 Flash — cents — and is a
gate; it needs `OPENROUTER_API_KEY` in this repository's own `.env`.

## Next, in order (each on the human's word)

1. `tune.teach --pairs data/pairs/v1/pairs.jsonl --out data/pairs/v1` —
   the teacher's `chosen`; then the count of pairs kept.
2. The DPO training app on `gpu="A10:2"` with FSDP, checkpoints every 20%
   of the steps; one smoke of a few steps.
3. The full run, the measurement through the harness, `base` beside it.
