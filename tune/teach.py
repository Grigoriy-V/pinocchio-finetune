"""Fill `chosen` in the loop pairs with the teacher's move on the same state.

Each pair from `tune/pairs.py` is a prompt (the messages a student saw when
it repeated a call) and the repeat as `rejected`. The teacher is asked the
same prompt with the same tools, temperature 0, and its reply is `chosen`.
A pair is dropped when the teacher makes the same call the student did
(then the state is not one the teacher resists either), or answers with
nothing.

Every call here is a paid model request, so the command is a gate: it
prints the count and the token estimate first and asks the same way
everything priced in this repository asks, unless `--yes` is given. Pairs
already taught in the output are not asked again, so a rerun after a
failure costs only what is missing.

Output: `taught.jsonl` (the pairs with `chosen` and the teacher's usage) and
`dpo.jsonl`, the conversational preference format TRL's DPOTrainer takes:

    {"prompt": [...], "chosen": [assistant], "rejected": [assistant], "tools": [...]}

The key is `OPENROUTER_API_KEY` in the environment, or in this repository's
own `.env`; nothing here reads the harness's settings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

from tune.pairs import call_key

TEACHER_MODEL = "z-ai/glm-5.3-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
KEY = "OPENROUTER_API_KEY"

Ask = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def key_from_env(root: Path = Path(".")) -> str:
    value = os.environ.get(KEY, "")
    if value:
        return value
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, sep, rest = line.partition("=")
            if sep and name.strip() == KEY:
                return rest.strip().strip('"').strip("'")
    return ""


def openrouter(model: str, key: str, timeout: float = 120.0) -> Ask:
    def ask(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    return ask


def chosen_of(response: dict[str, Any]) -> dict[str, Any] | None:
    """The teacher's reply as an assistant message; None when it said nothing."""
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    item: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append({
            "id": call.get("id") or "",
            "type": "function",
            "function": {"name": function.get("name") or "", "arguments": function.get("arguments") or "{}"},
        })
    if calls:
        item["tool_calls"] = calls
    if not item["content"] and not calls:
        return None
    return item


def key_of(message: dict[str, Any]) -> tuple[str, str] | None:
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    function = calls[0]["function"]
    return call_key({"name": function["name"], "arguments": function["arguments"]})


def pair_id(pair: dict[str, Any]) -> str:
    return f"{pair['run_id']}#{pair['call_index']}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def teach(pairs: list[dict[str, Any]], out: Path, ask: Ask) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    taught_path = out / "taught.jsonl"
    done = {pair_id(p): p for p in read_jsonl(taught_path)}
    stats: dict[str, Any] = {"asked": 0, "kept": 0, "teacher_repeated": 0, "teacher_silent": 0,
                             "already_taught": 0, "input_tokens": 0, "output_tokens": 0}
    with taught_path.open("a", encoding="utf-8") as sink:
        for pair in pairs:
            if pair_id(pair) in done:
                stats["already_taught"] += 1
                continue
            response = ask(pair["prompt"], pair.get("tools") or [])
            stats["asked"] += 1
            usage = response.get("usage") or {}
            stats["input_tokens"] += usage.get("prompt_tokens") or 0
            stats["output_tokens"] += usage.get("completion_tokens") or 0
            chosen = chosen_of(response)
            record = dict(pair)
            record["teacher"] = response.get("model") or ""
            record["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}
            if chosen is None:
                record["chosen"] = None
                record["dropped"] = "teacher silent"
                stats["teacher_silent"] += 1
            elif key_of(chosen) is not None and key_of(chosen) == key_of(pair["rejected"][0]):
                record["chosen"] = [chosen]
                record["dropped"] = "teacher repeated"
                stats["teacher_repeated"] += 1
            else:
                record["chosen"] = [chosen]
                stats["kept"] += 1
            done[pair_id(record)] = record
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (out / "dpo.jsonl").open("w", encoding="utf-8") as sink:
        for pair in pairs:
            record = done.get(pair_id(pair))
            if not record or record.get("dropped") or not record.get("chosen"):
                continue
            sink.write(json.dumps({
                "run_id": record["run_id"], "call_index": record["call_index"], "letter": record["letter"],
                "prompt": record["prompt"], "chosen": record["chosen"], "rejected": record["rejected"],
                "tools": record.get("tools") or [],
            }, ensure_ascii=False) + "\n")
    stats["dpo_pairs"] = sum(1 for p in pairs if (r := done.get(pair_id(p))) and not r.get("dropped") and r.get("chosen"))
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def estimate(pairs: list[dict[str, Any]]) -> int:
    """Characters of prompt over four: a rough count of input tokens."""
    return sum(len(json.dumps(p["prompt"], ensure_ascii=False)) // 4 for p in pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", required=True, type=Path, help="pairs.jsonl from tune.pairs")
    parser.add_argument("--out", required=True, type=Path, help="where taught.jsonl, dpo.jsonl and stats.json go")
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--yes", action="store_true", help="do not ask before the paid calls")
    args = parser.parse_args(argv)
    pairs = read_jsonl(args.pairs)
    done = {pair_id(p) for p in read_jsonl(args.out / "taught.jsonl")}
    todo = [p for p in pairs if pair_id(p) not in done]
    print(f"{len(todo)} of {len(pairs)} pairs to ask {args.model}; about {estimate(todo):,} input tokens", file=sys.stderr)
    if not todo:
        stats = teach(pairs, args.out, lambda m, t: {})
        print(json.dumps(stats, indent=2))
        return 0
    if not args.yes:
        answer = input("Paid model calls. Run exactly this? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("not run", file=sys.stderr)
            return 1
    key = key_from_env()
    if not key:
        print(f"{KEY} is not set", file=sys.stderr)
        return 2
    stats = teach(pairs, args.out, openrouter(args.model, key))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
