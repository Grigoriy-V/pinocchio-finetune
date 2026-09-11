"""Turn the harness's trajectory export into SFT samples.

Input: the directory `tools/export_trajectories.py` writes in the harness —
`index.jsonl` (one line per run with its outcome, checks and split) and
`runs/<run_id>.json` (the run's model calls exactly as the model saw them).

Output: one JSON line per model call, in the conversational prompt-completion
shape TRL's SFTTrainer takes, with the tool schemas the call was made with:

    {"run_id", "call_index", "letter",
     "prompt":     [messages the model saw],
     "completion": [the assistant message it produced],
     "tools":      [OpenAI function schemas]}

Nothing is reconstructed: a sample is a rename of fields. Text parts are
joined into a string; a call whose messages carry media (an image, a file)
is dropped, because the export keeps only the media's kind and size.

The filter is the one the harness's report §3f fixed for v1: the teacher's
model, not held out, the scenario's checks passed, the answer delivered. A
check that was dropped after the run (`--ignore-check`) does not count.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

TEACHER = "glm"


@dataclass
class Filter:
    teacher: str = TEACHER
    ignore_checks: tuple[str, ...] = ()
    include_held_out: bool = False

    def passed(self, entry: dict[str, Any]) -> bool:
        scenario = entry.get("scenario") or {}
        checks = scenario.get("checks") or {}
        if not checks:
            return False
        return all(
            ok for name, ok in checks.items() if name not in self.ignore_checks
        )

    def keeps(self, entry: dict[str, Any]) -> tuple[bool, str]:
        """Whether the run trains, and the reason when it does not."""
        if self.teacher not in (entry.get("model") or "").lower():
            return False, "other model"
        if entry.get("held_out") and not self.include_held_out:
            return False, "held out"
        if entry.get("outcome") != "answer_delivered":
            return False, f"outcome {entry.get('outcome')}"
        if not self.passed(entry):
            return False, "checks failed"
        return True, "kept"


@dataclass
class Stats:
    runs_seen: int = 0
    runs_kept: int = 0
    dropped_runs: Counter = field(default_factory=Counter)
    samples: int = 0
    samples_with_media: int = 0
    by_letter: Counter = field(default_factory=Counter)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs_seen": self.runs_seen,
            "runs_kept": self.runs_kept,
            "dropped_runs": dict(self.dropped_runs),
            "samples": self.samples,
            "samples_dropped_for_media": self.samples_with_media,
            "by_letter": dict(sorted(self.by_letter.items())),
            "teacher_prompt_tokens": self.prompt_tokens,
            "teacher_completion_tokens": self.completion_tokens,
        }


def text_of(content: Any) -> str | None:
    """Join the text parts; None when a part is not text (media)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("kind") == "text":
            texts.append(part.get("text") or "")
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text") or "")
        else:
            return None
    return "\n".join(texts)


def tool_call_of(call: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI shape, arguments as a JSON string (what chat templates take)."""
    arguments = call.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": call.get("id") or "",
        "type": "function",
        "function": {"name": call["name"], "arguments": arguments},
    }


def messages_of(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """The prompt as OpenAI messages; None when any part is media."""
    out: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for message in messages:
        role = message["role"]
        text = text_of(message.get("content"))
        if text is None:
            return None
        if role == "assistant":
            calls = [tool_call_of(c) for c in message.get("tool_calls") or []]
            for c in calls:
                names[c["id"]] = c["function"]["name"]
            item: dict[str, Any] = {"role": role, "content": text}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
        elif role == "tool":
            call_id = message.get("tool_call_id") or ""
            item = {"role": "tool", "content": text, "tool_call_id": call_id}
            if call_id in names:
                item["name"] = names[call_id]
            out.append(item)
        else:
            out.append({"role": role, "content": text})
    return out


def completion_of(completion: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {"role": "assistant", "content": completion.get("text") or ""}
    calls = [tool_call_of(c) for c in completion.get("tool_calls") or []]
    if calls:
        item["tool_calls"] = calls
    return item


def samples_of(run: dict[str, Any], letter: str, stats: Stats) -> Iterator[dict[str, Any]]:
    for call in run.get("calls") or []:
        prompt = messages_of(call["messages"])
        if prompt is None:
            stats.samples_with_media += 1
            continue
        usage = (call.get("completion") or {}).get("usage") or {}
        stats.prompt_tokens += usage.get("input_tokens") or 0
        stats.completion_tokens += usage.get("output_tokens") or 0
        stats.samples += 1
        stats.by_letter[letter] += 1
        yield {
            "run_id": run["run_id"],
            "call_index": call["call_index"],
            "letter": letter,
            "prompt": prompt,
            "completion": [completion_of(call["completion"])],
            "tools": call.get("tools") or [],
        }


def read_index(export: Path) -> list[dict[str, Any]]:
    with (export / "index.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert(export: Path, out: Path, rule: Filter) -> Stats:
    stats = Stats()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as sink:
        for entry in read_index(export):
            stats.runs_seen += 1
            keep, why = rule.keeps(entry)
            if not keep:
                stats.dropped_runs[why] += 1
                continue
            stats.runs_kept += 1
            run = json.loads((export / "runs" / f"{entry['run_id']}.json").read_text(encoding="utf-8"))
            letter = (entry.get("scenario") or {}).get("letter") or "?"
            for sample in samples_of(run, letter, stats):
                sink.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", required=True, type=Path, help="the harness's data/export")
    parser.add_argument("--out", required=True, type=Path, help="where train.jsonl and stats.json go")
    parser.add_argument("--teacher", default=TEACHER, help="substring of the teacher model's name")
    parser.add_argument(
        "--ignore-check",
        action="append",
        default=[],
        help="a check name that no longer counts (dropped after the run)",
    )
    parser.add_argument("--include-held-out", action="store_true")
    args = parser.parse_args(argv)
    rule = Filter(args.teacher, tuple(args.ignore_check), args.include_held_out)
    stats = convert(args.export, args.out, rule)
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
