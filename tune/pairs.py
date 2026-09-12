"""Preference pairs on the loop, from the harness's trajectory export.

A student run (a Gemma, tuned or not) is read call by call. Where the model
re-issued a call it had already made in the same turn — the same tool, the
same arguments — and nothing had changed in between, the state the model
saw at that call is a prompt and the repeat is its `rejected` completion.
"Nothing changed" is evidence, not a guess: either the harness's repeat
guard refused the call (its message names the earlier result as standing),
or the call ran again and returned the same text as the first time.

`chosen` is left empty here; `tune/teach.py` fills it with the teacher's
move on the same prompt, or the pair is dropped when the teacher repeats
too. The shape is TRL's conversational preference format with tools:

    {"run_id", "call_index", "letter", "model", "repeat_of", "evidence",
     "prompt":   [messages the model saw],
     "rejected": [the assistant message it produced],
     "chosen":   null,
     "tools":    [OpenAI function schemas]}

Several exports may be given; a run id seen twice is read once. A run
contributes at most `per_run` pairs (the earliest states), so that one run
of ninety repeats does not become the whole set.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tune.convert import TEACHER, completion_of, messages_of, read_index, text_of

GUARD_PREFIX = "error: this exact call has already"
# The command runner stamps its first line with the wall time it took
# ("exit code: 1   (2.8 s)"); two runs of the same command differ there and
# nowhere else, so the stamp is not a change.
TIMING = re.compile(r"\s*\(\d+(?:\.\d+)? s\)")


def same_text(a: str, b: str) -> bool:
    return TIMING.sub("", a) == TIMING.sub("", b)


def letter_of(entry: dict[str, Any]) -> str:
    """The scenario letter; from the thread id `chat-v1` when the export
    carried no scenario (the int4 baselines were exported that way)."""
    letter = (entry.get("scenario") or {}).get("letter")
    if letter:
        return letter
    match = re.fullmatch(r"chat-([a-z]\d+)", entry.get("thread_id") or "")
    return match.group(1).upper() if match else "?"


@dataclass
class PairStats:
    runs_seen: int = 0
    runs_student: int = 0
    runs_with_pairs: int = 0
    repeats_seen: int = 0
    pairs: int = 0
    dropped: Counter = field(default_factory=Counter)
    by_letter: Counter = field(default_factory=Counter)
    by_model: Counter = field(default_factory=Counter)
    by_evidence: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs_seen": self.runs_seen,
            "runs_student": self.runs_student,
            "runs_with_pairs": self.runs_with_pairs,
            "repeats_seen": self.repeats_seen,
            "pairs": self.pairs,
            "dropped": dict(self.dropped),
            "by_letter": dict(sorted(self.by_letter.items())),
            "by_model": dict(sorted(self.by_model.items())),
            "by_evidence": dict(sorted(self.by_evidence.items())),
        }


def call_key(call: dict[str, Any]) -> tuple[str, str]:
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            pass
    return call["name"], json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def results_of(calls: list[dict[str, Any]]) -> dict[str, str]:
    """Every tool result the model saw in the run, by tool call id."""
    out: dict[str, str] = {}
    for call in calls:
        for message in call.get("messages") or []:
            if message.get("role") == "tool" and message.get("tool_call_id"):
                text = text_of(message.get("content"))
                if text is not None:
                    out[message["tool_call_id"]] = text
    return out


def pairs_of(run: dict[str, Any], letter: str, per_run: int, stats: PairStats) -> Iterator[dict[str, Any]]:
    calls = sorted(run.get("calls") or [], key=lambda c: c["call_index"])
    results = results_of(calls)
    first: dict[tuple[str, str], tuple[int, str]] = {}
    made = 0
    for call in calls:
        completion = call.get("completion") or {}
        tool_calls = completion.get("tool_calls") or []
        if len(tool_calls) != 1:
            for tc in tool_calls:
                first.setdefault(call_key(tc), (call["call_index"], tc.get("id") or ""))
            continue
        tc = tool_calls[0]
        key = call_key(tc)
        if key not in first:
            first[key] = (call["call_index"], tc.get("id") or "")
            continue
        stats.repeats_seen += 1
        earlier_index, earlier_id = first[key]
        result = results.get(tc.get("id") or "")
        if result is None:
            stats.dropped["no result seen"] += 1
            continue
        if result.startswith(GUARD_PREFIX):
            evidence = "guard"
        elif earlier_id in results and same_text(results[earlier_id], result):
            evidence = "same result"
        else:
            stats.dropped["result changed"] += 1
            continue
        if made >= per_run:
            stats.dropped["over per-run cap"] += 1
            continue
        prompt = messages_of(call["messages"])
        if prompt is None:
            stats.dropped["media"] += 1
            continue
        made += 1
        stats.pairs += 1
        stats.by_letter[letter] += 1
        stats.by_model[run.get("model") or "?"] += 1
        stats.by_evidence[evidence] += 1
        yield {
            "run_id": run["run_id"],
            "call_index": call["call_index"],
            "letter": letter,
            "model": run.get("model") or "?",
            "repeat_of": earlier_index,
            "evidence": evidence,
            "prompt": prompt,
            "rejected": [completion_of(completion)],
            "chosen": None,
            "tools": call.get("tools") or [],
        }
    if made:
        stats.runs_with_pairs += 1


def extract(exports: list[Path], out: Path, teacher: str = TEACHER, per_run: int = 3) -> PairStats:
    stats = PairStats()
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as sink:
        for export in exports:
            for entry in read_index(export):
                if entry["run_id"] in seen:
                    continue
                seen.add(entry["run_id"])
                stats.runs_seen += 1
                if teacher in (entry.get("model") or "").lower():
                    continue
                stats.runs_student += 1
                run = json.loads((export / "runs" / f"{entry['run_id']}.json").read_text(encoding="utf-8"))
                letter = letter_of(entry)
                for pair in pairs_of(run, letter, per_run, stats):
                    sink.write(json.dumps(pair, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", required=True, action="append", type=Path,
                        help="a harness export directory; repeat for several")
    parser.add_argument("--out", required=True, type=Path, help="where pairs.jsonl and stats.json go")
    parser.add_argument("--teacher", default=TEACHER, help="substring of the teacher model's name; its runs are skipped")
    parser.add_argument("--per-run", type=int, default=3, help="at most this many pairs from one run")
    args = parser.parse_args(argv)
    stats = extract(args.export, args.out, args.teacher, args.per_run)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
