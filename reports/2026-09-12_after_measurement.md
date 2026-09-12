# Before and after, on the same Linux workers, blind-judged — 2026-09-12

The measurement the experiment was for. Both endpoints are the training
repository's own `serve_app` on the second Modal workspace (bf16, L40S,
vLLM 0.26.0, the same flags): `base` is `google/gemma-4-12B-it` untouched,
`tuned` is the same with the v1-r16 LoRA merged in. The harness's
`assistant-control` was deployed to the same workspace so that the
scenarios run on its Linux workers, as the earlier baselines did (the
first workspace's balance being out). `loop_live --deployed --model
<set> D V X`, temperature 0, one sample per case: 17 cases each — the
held-out D1–3 V1–3 X1–3 plus the variants D4–5, V4–7, X4–5 that were in
the training data as prompts.

Beside them in the same blind batch: GLM (the teacher) and the untuned
Gemma int4 QAT endpoint of the first workspace, both from 2026-09-11 on
the same families, held-out cases only (9 each). 52 transcripts, shuffled,
three Sonnet judges on the fixed rubric (a–e, 0–2 each), the key unseen;
`tools/judge_unblind.py` in the harness reads the votes.

## The numbers

| model | n | checks | judges, mean of 10 | per judge | a | b | c | d | e |
|---|---|---|---|---|---|---|---|---|---|
| GLM 5.3 Flash (teacher) | 9 | 9/9 | **9.81** | 9.78 / 10.00 / 9.67 | 2.00 | 1.81 | 2.00 | 2.00 | 2.00 |
| Gemma-IT bf16, untuned (`base`) | 17 | 16/17 | **9.25** | 9.06 / 9.76 / 8.94 | 1.88 | 1.45 | 2.00 | 1.92 | 2.00 |
| Gemma-IT bf16 + v1 LoRA (`tuned`) | 17 | 15/17 | **8.67** | 8.59 / 8.94 / 8.47 | 1.84 | 1.31 | 1.88 | 1.75 | 1.88 |
| Gemma-IT int4 QAT (2026-09-11) | 9 | 7/9 | **8.15** | 8.00 / 8.44 / 8.00 | 1.78 | 1.22 | 1.78 | 1.59 | 1.78 |

Held-out only (D1–3, V1–3, X1–3), across-judge means:

| case | base | tuned | int4 | GLM |
|---|---|---|---|---|
| D1 | 8.3 | 9.0 | 9.3 | 10.0 |
| D2 | 9.3 | 9.3 | 9.7 | 9.3 |
| D3 | 9.7 | 9.0 | 10.0 | 10.0 |
| V1 | 7.7 | **0.0** | 6.3 | 10.0 |
| V2 | 10.0 | 10.0 | 9.3 | 10.0 |
| V3 | 10.0 | 10.0 | 10.0 | 10.0 |
| X1 | 9.3 | 9.3 | 9.3 | 9.7 |
| X2 | 8.7 | 8.0 | **0.0** | 10.0 |
| X3 | 9.3 | 9.3 | 9.3 | 9.3 |
| mean | 9.1 | 8.2 | 8.6 | 9.8 |

Variants (were training prompts): tuned D4 10.0 / D5 10.0 against base
8.3 / 9.3 (base edited `config.json`, which the task forbade); V6 tuned
**6.0** against base 10.0 (92 calls cycling `find`/`cat`/`ls` for six
minutes until the turn check forced a one-line answer); V4 V5 V7 X3 X4
X5 equal.

Judges agreed within a point on 47 of 52; the five outside it are 8–10
splits over `set_goal` and a global `pip install`, none on a loop.

## What it says

1. **The v1 LoRA did not make Gemma more agentic; on this set it is a
   little worse.** 8.67 against 9.25 overall, 8.2 against 9.1 on the
   held-out cases, 15/17 against 16/17 by the checks. The whole gap is two
   turns — V1 (0.0) and V6 (6.0) — and both are the same shape: the model
   re-issues a command whose result it has already seen, until the guard
   or the turn check stops it. Outside those two the tuned model scores
   the base's numbers, and on D4/D5 it beats them.
2. **The loop is the thing, and 34k target tokens from a teacher that
   does not loop did not touch it.** The teacher's trajectories contain
   the *absence* of a loop, not the moment of resisting one; the guard's
   own message occurs in almost no training prompt. Imitation data of
   this kind cannot teach a behaviour the teacher never exhibits the
   correction of.
3. **The quantisation mattered more than the fine-tune.** Untuned bf16
   (9.25) against untuned int4 QAT (8.15) on comparable cases: the
   product's Gemma endpoint has been running the weaker of the two. That
   is a harness finding worth more than the experiment's own result.
4. **The empty answer after the guard (ISS-0069) is the same in all three
   Gemmas**: tuned V1, int4 X2 both 0.0 because the turn ended with
   nothing said. Fix that in the harness and every Gemma's floor rises.
5. The mid-turn message X4: tuned did not answer it on Linux (a check
   failure the judges did not penalise beyond a point); base did.

Cost of the measurement: two `scenarios` calls on the second workspace
(CPU) plus two L40S cold starts and ~25 minutes of GPU: about $2.

## What would be worth doing next, as options

- **Data that contains the correction**: trajectories where a model
  *did* loop and then recovered, or a preference pair (looped turn vs
  the same prompt answered) — DPO/KTO on the loop, not SFT on its
  absence. The harness has the looping turns already (every Gemma V1/V6/X2
  run) and GLM's clean answers to the same prompts.
- **More epochs or a larger rank** would not address 2; the loss curve
  was already at 0.2.
- **The harness side first**: ISS-0069 (the empty answer) and a guard
  that catches a cycle of several commands (V6's 92 calls passed the
  exact-repeat guard). Both lift every Gemma, tuned or not.
- **Serve the product's Gemma in bf16** rather than int4 QAT, if an L40S
  per request is acceptable — a product decision, the human's.
