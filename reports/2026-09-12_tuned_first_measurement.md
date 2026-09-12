# The tuned model's first measurement: D V X, local harness on Windows — 2026-09-12

`AGENT_TRAJECTORIES=data/trajectories/tuned loop_live --model tuned D V X`
in the harness's **local** profile on the Windows machine, against
`pinocchio-tune-serve` (bf16 merged v1-r16, L40S, second workspace). 17
turns, ~9 minutes of model time after a ~2-minute cold start; the
trajectories are in the harness's `data/trajectories/tuned/` (17 files).
Two earlier "tuned" runs the same morning were GLM: `loop_live` applied
`--model` only on the deployed path (fixed, harness `0078049`) — the
human noticed that no container had woken.

Also the mini set (A B C F W H E M) — the 8/8 reported before the fix was
GLM's, not the tuned model's. Not repeated yet.

## By the checks: 11 of 17 cases, 7 of 9 held-out

| case | result | what happened (from the trajectory) | whose |
|---|---|---|---|
| D1 D2 D3 | pass | | |
| D4 | **fail** | fixed the path, then fought the trailing comma by rewriting the file with the same content again and again; the repeat guard ended the turn; empty answer | model (loop) + the empty answer, below |
| D5 | **fail** | `GREETING=Hello python …` is not cmd syntax; found `set GREETING=Hello && python …` (prints "Hello , Ada!" — cmd keeps the trailing space); then repeated that command until the guard; empty answer | environment first, then model (loop) |
| V1 | **fail** | `sh`, `ls`, `find` all die: `CreateFileMapping … Win32 error 5` (the sandbox breaks Cygwin tools); `dir tools_task/out/` fails on the slash; the model then emitted `command:` as a command, five times | environment, then model (malformed calls) |
| V2 V3 V4 V5 | pass | | |
| V6 | **fail** | `sh clean.sh` dies (Cygwin); the model **deleted the four .tmp files itself** with python, then looped on `os.listdir`; empty answer | model (did the script's job instead of reporting its lie) |
| V7 | **fail** | `python3` absent, `cat` dies (Cygwin); the model read results.txt correctly (3 passed) then looped on `send_file` until the guard; empty answer | environment, then model (loop) |
| X1 | **fail** | `awk` dies (Cygwin); the model summed by hand: 150 + 60 + 40 = **260**; every file made, the mid-turn message answered | model (arithmetic) |
| X2 X3 X4 X5 | pass | | |

## Two harness findings, both older than the fine-tune

1. **The local Windows `run_command` kills every Cygwin tool** (`sh`, `ls`,
   `find`, `cat`, `awk`: `CreateFileMapping … Win32 error 5`), and the
   scenarios are written for a Linux worker (`python3`, env-var prefixes,
   `.sh` scripts). Five of the six failures start there. The measurement
   environment is not the baselines' (deployed Linux workers on the first
   workspace). Recorded in the harness's ISSUES.
2. **After the repeat guard, Gemma's answer is empty** — in every
   guard-ended turn here (D4 D5 V1 V6 V7) *and* in the untuned baseline's
   X2 on the deployed Linux worker (run `deployed-v2-aaa50fb0-820`): the
   request goes out with no tools, the model produces 16–171 output
   tokens, `finish_reason: stop`, and the harness receives no text and no
   call. What those tokens are is not recorded (no `dump_dir` locally); the
   likely shape is a `<|tool_call>` emitted with no tools declared, which
   the vLLM parser drops. Before and after the fine-tune alike; recorded
   in the harness's ISSUES.

## What the fine-tune did and did not do, as far as this run can tell

- Kept: parse rate (every call parsed, none malformed until V1's
  post-failure `command:`), the stop tokens, the mid-turn message, the
  three-deliverable tasks (X2–X5 all pass; X2 looped in the untuned
  baseline).
- Not fixed: the repeat loop. The tuned model still re-issues the same
  call when a result surprises it (D4, D5, V6, V7) — the shape the
  baseline report named. 782 samples with 34k target tokens from a teacher
  that seldom loops did not teach "stop repeating"; the guard's own
  message ("this exact call has already succeeded twice…") appears in
  almost no training prompt, so the model never saw what to do after it.
- New: V6 deleting the files itself is the worst turn of the run — a
  rubric-d failure the untuned baseline did not make on V6.

None of it is a before/after number: the environment differs from the
baselines', and the "before" was never run here. The comparison needs the
same environment for both — either the deployed Linux scenarios on the
first workspace (the baselines' own; needs its balance) or the `base`
endpoint through the same local harness (both sides then share the
Windows defects). The judges have nothing to score until that.
