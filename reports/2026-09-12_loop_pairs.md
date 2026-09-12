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

## The teacher's moves (2026-09-12, later)

`tune.teach` on the 34 pairs: 34 asked, **23 kept**, 11 dropped because
the teacher made the same call the student had (GLM repeats in a third of
these states too: the state is confusing to everyone, not only to Gemma).
242k input tokens (the estimate of 130k was half: tool schemas and the
system prompt are not in the pair's messages), 1.7k output; cents.

Read one by one, the 23 kept are of three kinds:

- **a way out** — an answer in text ("I ran it and checked: `out/app.bin`
  does not exist"), or the one read of `build.sh` that the case wants:
  V1 tuned 8, 9, 11; V6 tuned 8; D1 base 8; V1 int4 7, 13. About seven.
- **a different probe of the same thing** — `find … | wc -l` where the
  student had `find …`, `cat clean.sh` a third time under another
  spelling, `bash build.sh && find` after `bash build.sh && ls`. Different
  by the letter, no better in substance. Most of the rest.
- **the case's right move under another cwd** — X2's `python3 -m unittest
  calc_pkg.test…` for the student's `cd calc_pkg && …`: plausibly better,
  not obviously.

So the mechanical filter (a different call) keeps pairs whose `chosen`
is not a lesson. The blind judge the human asked about is the tool for
exactly this: two moves on one state, anonymised, three Sonnet judges
asked whether the state is a dead end and which move leaves it. A pair
the judges do not prefer `chosen` on is dropped. Proposed as the step
between `teach` and training; a draft until approved.

## The judges (2026-09-12, later still)

Approved by the human ("давай"). `tune/judge_pairs.py pack` (seed 7): the
23 pairs as items, each the state without the system prompt and the two
moves as A and B in a drawn order, no model names; the key apart. Three
Sonnet subagents, each told to read the pack and nothing else, one JSON
line per item. `unblind` keeps a pair when a majority preferred the
teacher's move and no judge scored the state 0.

| | |
|---|---|
| items | 23 |
| kept | **16** |
| dropped: judges called both moves the same | 7 |
| votes for the teacher / same / for the student | 47 / 22 / 0 |
| `dead_end` = 2 | 69 of 69 |
| unanimous items | 22 of 23 |

The seven dropped are exactly the "different probe of the same thing"
kind read out above (V6 `cat` vs `find` a third time, V1 `ls -la` vs
`find`, the guard-refused call vs its restatement): the judges named them
without seeing who was who. No item went to the student. Every state was
judged a dead end, so the extractor's rule of evidence held on every pair.

The set for training: `data/pairs/v1/dpo_judged.jsonl`, 16 pairs — V1 9
(int4 5, tuned 4), X2 3, V6 2, D1 D2 N1 one each; `chosen` is a text
answer in three of them, a read of the script in five, a corrected
command in the rest. Cost of the judging: three Sonnet subagents of ~100k
tokens each.

## Next, in order (each on the human's word)

1. The DPO training app on `gpu="A10:2"` with FSDP, checkpoints every 20%
   of the steps; one smoke of a few steps.
2. The full run, the measurement through the harness, `base` beside it.
