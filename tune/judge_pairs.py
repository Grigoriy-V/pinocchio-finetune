"""Blind judging of the loop pairs: pack, then unblind.

`pack` writes one Markdown file with every taught pair as an item: the
state the model saw (the task, then each call and what came back, long
results cut in the middle) and the two moves as A and B in an order drawn
by the seed, without the models' names. The key (which letter is the
teacher, for each item) goes to a separate JSON file the judges never see.

A judge answers one JSON line per item:

    {"item": 3, "dead_end": 2, "better": "A", "why": "..."}

`dead_end`: 0 = the repeat would have told the model something new; 1 =
unclear; 2 = the state is a dead end, repeating the call cannot help.
`better`: which move is the better next step from this state, or "same".

`unblind` reads the judges' files, matches letters back, and writes
`judged.jsonl` (every pair with its votes) and `dpo_judged.jsonl` (the
pairs kept: a majority of judges preferred the teacher's move and none
scored the state 0), plus `stats.json`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tune.teach import read_jsonl

RUBRIC = """# Two moves from one state

Each item below is the state of an assistant working in a sandbox: the
person's task, then every tool call the assistant made and what the tool
returned. At the end are two candidate next moves, A and B. One of the
assistant's earlier calls has been repeated with exactly the same
arguments; the state is shown up to the point where a next move is due.

Answer for every item, one JSON object per line, nothing else:

    {"item": <n>, "dead_end": <0|1|2>, "better": "A"|"B"|"same", "why": "<one sentence>"}

- `dead_end`: 2 = repeating any of the calls already made, with the same
  arguments, cannot tell the assistant anything new (nothing has changed);
  1 = unclear; 0 = a repeat could reasonably return something different.
- `better`: the move that is the better next step from this exact state —
  the one that makes progress on the task or answers the person honestly
  with what is already known. A move that re-runs what was already run,
  or probes the same fact in another spelling, is not progress. Prefer
  "same" when neither moves the task.

Judge the moves on substance, not on style or length.
"""


def cut(text: str, head: int = 500, tail: int = 200) -> str:
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + f"\n… [{len(text) - head - tail} characters cut] …\n" + text[-tail:]


def render_move(message: dict[str, Any]) -> str:
    parts = []
    if message.get("content"):
        parts.append(message["content"].strip())
    for call in message.get("tool_calls") or []:
        function = call["function"]
        parts.append(f"call `{function['name']}` {function['arguments']}")
    return "\n".join(parts) or "(nothing)"


def render_state(prompt: list[dict[str, Any]]) -> str:
    lines = []
    for message in prompt:
        role = message["role"]
        if role == "system":
            continue
        if role == "user":
            lines.append(f"**Person:** {message['content'].strip()}")
        elif role == "assistant":
            lines.append(f"**Assistant:** {render_move(message)}")
        elif role == "tool":
            name = message.get("name") or "tool"
            lines.append(f"**{name} returned:**\n```\n{cut(message['content'])}\n```")
    return "\n\n".join(lines)


def pack(taught: Path, out: Path, seed: int = 0) -> int:
    out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in read_jsonl(taught) if p.get("chosen") and not p.get("dropped")]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    key: list[dict[str, Any]] = []
    body = [RUBRIC]
    for number, pair in enumerate(pairs, 1):
        teacher_is_a = rng.random() < 0.5
        a, b = (pair["chosen"][0], pair["rejected"][0]) if teacher_is_a else (pair["rejected"][0], pair["chosen"][0])
        body.append(f"\n---\n\n## Item {number}\n\n{render_state(pair['prompt'])}\n\n"
                    f"**Move A:** {render_move(a)}\n\n**Move B:** {render_move(b)}\n")
        key.append({"item": number, "run_id": pair["run_id"], "call_index": pair["call_index"],
                    "teacher": "A" if teacher_is_a else "B"})
    (out / "pack.md").write_text("\n".join(body), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return len(pairs)


def read_votes(path: Path) -> dict[int, dict[str, Any]]:
    votes: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        vote = json.loads(line)
        votes[int(vote["item"])] = vote
    return votes


def unblind(taught: Path, pack_dir: Path, vote_files: list[Path], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    key = {k["item"]: k for k in json.loads((pack_dir / "key.json").read_text(encoding="utf-8"))}
    judges = [read_votes(f) for f in vote_files]
    by_id = {f"{p['run_id']}#{p['call_index']}": p for p in read_jsonl(taught)}
    stats: dict[str, Any] = {"items": len(key), "judges": len(judges), "kept": 0,
                             "dropped_not_preferred": 0, "dropped_not_dead_end": 0,
                             "teacher_preferred_votes": 0, "same_votes": 0, "student_preferred_votes": 0,
                             "dead_end": Counter()}
    kept: list[dict[str, Any]] = []
    with (out / "judged.jsonl").open("w", encoding="utf-8") as sink:
        for item, entry in sorted(key.items()):
            pair = by_id[f"{entry['run_id']}#{entry['call_index']}"]
            votes = [j.get(item) for j in judges if j.get(item)]
            prefer = 0
            dead_zero = False
            rendered = []
            for vote in votes:
                better = vote.get("better")
                who = "teacher" if better == entry["teacher"] else ("same" if better == "same" else "student")
                prefer += who == "teacher"
                stats[f"{who}_preferred_votes" if who != "same" else "same_votes"] += 1
                stats["dead_end"][str(vote.get("dead_end"))] += 1
                dead_zero |= vote.get("dead_end") == 0
                rendered.append({"better": who, "dead_end": vote.get("dead_end"), "why": vote.get("why")})
            keep = votes and prefer * 2 > len(votes) and not dead_zero
            if keep:
                stats["kept"] += 1
                kept.append(pair)
            elif dead_zero:
                stats["dropped_not_dead_end"] += 1
            else:
                stats["dropped_not_preferred"] += 1
            sink.write(json.dumps({"item": item, "run_id": pair["run_id"], "call_index": pair["call_index"],
                                   "letter": pair["letter"], "kept": bool(keep), "votes": rendered},
                                  ensure_ascii=False) + "\n")
    with (out / "dpo_judged.jsonl").open("w", encoding="utf-8") as sink:
        for pair in kept:
            sink.write(json.dumps({k: pair[k] for k in ("run_id", "call_index", "letter", "prompt", "chosen", "rejected", "tools")},
                                  ensure_ascii=False) + "\n")
    stats["dead_end"] = dict(stats["dead_end"])
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--taught", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    u = sub.add_parser("unblind")
    u.add_argument("--taught", required=True, type=Path)
    u.add_argument("--pack", required=True, type=Path)
    u.add_argument("--votes", required=True, nargs="+", type=Path)
    u.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "pack":
        print(f"{pack(args.taught, args.out, args.seed)} items")
    else:
        print(json.dumps(unblind(args.taught, args.pack, args.votes, args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
