import json
from pathlib import Path

from tune.judge_pairs import cut, pack, unblind

TOOLS = [{"type": "function", "function": {"name": "run_command"}}]


def _move(command=None, text=""):
    item = {"role": "assistant", "content": text}
    if command:
        item["tool_calls"] = [{"id": "x", "type": "function",
                               "function": {"name": "run_command", "arguments": json.dumps({"command": command})}}]
    return item


def _taught(tmp_path: Path):
    prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build it"},
              _move("sh build.sh"), {"role": "tool", "name": "run_command", "tool_call_id": "x", "content": "built"}]
    pairs = [
        {"run_id": "g1", "call_index": 3, "letter": "V1", "prompt": prompt, "tools": TOOLS,
         "rejected": [_move("sh build.sh")], "chosen": [_move(text="It is built.")]},
        {"run_id": "g1", "call_index": 5, "letter": "V1", "prompt": prompt, "tools": TOOLS,
         "rejected": [_move("sh build.sh")], "chosen": [_move("ls out")]},
        {"run_id": "g2", "call_index": 2, "letter": "X2", "prompt": prompt, "tools": TOOLS,
         "rejected": [_move("sh build.sh")], "chosen": [_move("sh build.sh")], "dropped": "teacher repeated"},
    ]
    path = tmp_path / "taught.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    return path


def test_pack_hides_names_and_keeps_the_key_apart(tmp_path):
    taught = _taught(tmp_path)
    n = pack(taught, tmp_path / "judge", seed=1)
    text = (tmp_path / "judge" / "pack.md").read_text(encoding="utf-8")
    key = json.loads((tmp_path / "judge" / "key.json").read_text(encoding="utf-8"))
    assert n == 2 and text.count("## Item") == 2 and "sys" not in text
    assert "gemma" not in text.lower() and "teacher" not in text.split("## Item")[1].lower()
    assert "**Person:** build it" in text and "**Move A:**" in text and "**Move B:**" in text
    assert {k["teacher"] for k in key} <= {"A", "B"} and {k["item"] for k in key} == {1, 2}


def test_unblind_keeps_a_majority_for_the_teacher_and_no_zero_dead_end(tmp_path):
    taught = _taught(tmp_path)
    pack(taught, tmp_path / "judge", seed=1)
    key = {k["item"]: k for k in json.loads((tmp_path / "judge" / "key.json").read_text(encoding="utf-8"))}
    other = {"A": "B", "B": "A"}
    t1, t2 = key[1]["teacher"], key[2]["teacher"]
    votes = [
        [{"item": 1, "dead_end": 2, "better": t1, "why": "a"}, {"item": 2, "dead_end": 2, "better": t2, "why": "b"}],
        [{"item": 1, "dead_end": 2, "better": t1, "why": "a"}, {"item": 2, "dead_end": 0, "better": t2, "why": "b"}],
        [{"item": 1, "dead_end": 1, "better": other[t1], "why": "a"}, {"item": 2, "dead_end": 2, "better": "same", "why": "b"}],
    ]
    files = []
    for i, v in enumerate(votes):
        f = tmp_path / f"judge{i}.jsonl"
        f.write_text("```\n" + "\n".join(json.dumps(x) for x in v) + "\n```\n", encoding="utf-8")
        files.append(f)
    stats = unblind(taught, tmp_path / "judge", files, tmp_path / "out")
    assert stats["kept"] == 1 and stats["dropped_not_dead_end"] == 1 and stats["dropped_not_preferred"] == 0
    assert stats["teacher_preferred_votes"] == 4 and stats["student_preferred_votes"] == 1 and stats["same_votes"] == 1
    judged = [json.loads(l) for l in (tmp_path / "out" / "judged.jsonl").read_text(encoding="utf-8").splitlines()]
    kept = [j for j in judged if j["kept"]]
    assert len(kept) == 1 and kept[0]["run_id"] == key[1]["run_id"] and kept[0]["call_index"] == key[1]["call_index"]
    dpo = [json.loads(l) for l in (tmp_path / "out" / "dpo_judged.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dpo) == 1 and set(dpo[0]) == {"run_id", "call_index", "letter", "prompt", "chosen", "rejected", "tools"}


def test_cut_keeps_head_and_tail():
    text = "a" * 1000 + "b" * 1000
    out = cut(text, head=100, tail=50)
    assert out.startswith("a" * 100) and out.endswith("b" * 50) and "characters cut" in out
    assert cut("short") == "short"
