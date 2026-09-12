# DPO on the loop, measured blind beside the base (2026-09-12)

Item 1 of the queue, closed once: 16 judged loop pairs → DPO on two A10s
(`reports/2026-09-12_dpo_smoke.md`, run `dpo-v1`) → merged → served as
`[model.sets.dpo]` on the second workspace → the harness's D V X
families on its Linux workers at temperature 0, 17 cases each, `base`
measured again in the same deploy → three blind Sonnet judges on one
batch of 43 transcripts (dpo 17, base 17, GLM's 9 held-out runs of
2026-09-11 as the anchor), `tools/judge_unblind.py`.

## The numbers

| model | checks | judges, mean of 10 | per judge | a | b | c | d | e |
|---|---|---|---|---|---|---|---|---|
| GLM 5.3 Flash (anchor, 9 held-out) | 9/9 | **9.70** | 9.56 / 9.89 / 9.67 | 2.00 | 1.74 | 1.96 | 2.00 | 2.00 |
| Gemma-IT bf16, untuned (`base`) | 57/57 | **9.47** | 9.18 / 9.59 / 9.65 | 1.94 | 1.67 | 2.00 | 1.86 | 2.00 |
| Gemma-IT bf16 + DPO `dpo-v1` (`dpo`) | 55/57 | **9.37** | 9.00 / 9.59 / 9.53 | 1.84 | 1.71 | 1.98 | 1.88 | 1.96 |

By case, across-judge means:

| case | base | dpo | GLM |
|---|---|---|---|
| D1 | 9.3 | 9.7 | 10.0 |
| D2 | 8.0 | 9.7 | 8.7 |
| D3 | 10.0 | 10.0 | 10.0 |
| D4 | 7.7 | **4.0** | – |
| D5 | 10.0 | 10.0 | – |
| V1 | 8.7 | 9.3 | 9.7 |
| V2–V6 | 10.0 | 10.0 | 10.0 / – |
| V7 | 9.7 | 9.7 | – |
| X1 | 10.0 | 10.0 | 9.3 |
| X2 | 8.7 | 8.7 | 10.0 |
| X3 | 9.7 | 9.7 | 9.7 |
| X4 | 9.7 | 9.0 | – |
| X5 | 9.7 | 9.7 | – |

Judges within a point on 40 of 43; the three outside are D4 dpo (3 / 5 /
4, all three naming the same loop), D1 base and X4 dpo.

## What it says

1. **DPO on 16 pairs did not make Gemma stop looping; on the whole set it
   is level with the base and a tenth below.** 9.37 against 9.47, 55/57
   against 57/57 by the checks. Where the base and the DPO model differ
   by more than a point it is one case each way: D2 (dpo 9.7, base 8.0 —
   the base re-ran a failing command once) and D4 (dpo 4.0, base 7.7).
2. **D4 is the loop, on a prompt that was in the training pairs.** The
   task: `settings/show.py` fails on a `config.json` with a trailing
   comma, make it work without editing the config. The DPO model read
   the files, saw the comma, and then ran the same
   `python3 -c "json.loads('{…}')"` six times — a command that printed
   `3` every time — until the repeat guard refused it, and only then
   answered in words. Its answer was right and grounded ("the script
   fails because of a trailing comma", c = 2); it never fixed the script
   (a = 1) and looped to get there (b = 0, d = 0).
3. **What the pairs taught is visible, and it is the wrong half.** Three
   of the sixteen `chosen` completions were text answers after a dead
   end, and the DPO model now answers in words after the guard where the
   untuned Gemmas of ISS-0069 answered nothing (D4 ended with a real
   answer, not an empty one). The other half — do not re-issue the call
   whose result you have — did not take: the pairs' `rejected` were
   exact repeats, and the model's D4 loop is an exact repeat, so the
   signal was the right one; sixteen pairs, three epochs, β 0.1 at
   rank 16 did not move it. V6 (the tuned v1's 92-call cycle) was clean
   for both models today, base and dpo alike, at 3 calls.
4. **The measurement itself had to be done twice**, and the first
   version is recorded because the mistake is the harness's operator's:
   I ran `dpo` and `base` as two plain `scenarios` calls at once, both
   under the default probe user, so both wrote into the same threads and
   workspace; `dpo` ran a case ahead and in 12 of 17 cases the base's
   first model call already carried the dpo's finished turn ("I have
   already completed this task for you", no tool call, 11 failed checks).
   The suite's `--parallel` exists for exactly this and gives each call
   its own probe. The base was rerun alone (`deployed-base-40e5ad1a-*`,
   every first call two messages long) and the batch judged again; the
   first batch's dpo and GLM scores (9.41 and 9.89) sit within a tenth of
   the second's.
5. Two things for the harness, recorded there: ISS-0070, the first turn
   on a fresh worker spends 37–155 s building the graph since the MCP
   servers were configured (item 26); and the operator's own rule, that
   no local command timer may hold a run — two runs were lost to it
   before `--detach` and detached processes.

## Cost

Training: ~$1 of two A10s (12 steps with a cancelled half, seven smokes
~$1.5 before it). Merge ~$0.5 CPU. Measurement: four L40S cold starts
(two lost to a cancellation, one to the contaminated pair) and four
`scenarios` containers, ~$3; the teacher's 34 calls, cents; six Sonnet
judge subagents of ~130k tokens each plus one that split itself into five.
About $6 for the whole item after the pairs.

## What would be worth doing next, as options

- **More pairs before more epochs.** Sixteen is the count the loop left
  in the exports; item 4 (more D and X trajectories, `--parallel`) is
  the way to fifty. The extractor, the teacher and the judges take any
  number of exports.
- **KTO on the same states** with every refused repeat as "undesirable"
  (34, not 16) and the teacher's clean turns as "desirable": the loop
  signal without the pairing constraint.
- **The harness's guard for a cycle, not only an exact repeat**, and
  ISS-0069/0070: every Gemma's floor rises with the harness, tuned or
  not (item 3 of the queue). D4's six identical calls passed two
  guard-free repeats before the third was refused; a guard that counts
  an identical *successful* call from the second would have cut the
  loop to two.
- **The 2×A10 FSDP recipe stands** and is the item's other result: bf16
  12B on two 24 GB cards at 19.6 GiB peak, exact resume from a
  checkpoint, ~2 min per step of two pairs. A `TUNE_GPU=A10:4` point is
  one run away.
