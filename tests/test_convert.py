import json
from pathlib import Path

from tune.convert import Filter, convert, messages_of, text_of, tool_call_of


def _export(tmp_path: Path, entries, runs) -> Path:
    export = tmp_path / "export"
    (export / "runs").mkdir(parents=True)
    with (export / "index.jsonl").open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    for run in runs:
        (export / "runs" / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")
    return export


def _entry(run_id, *, model="z-ai/glm-5.3-flash", held_out=False, passed=True,
           checks=None, outcome="answer_delivered", letter="D4"):
    return {
        "run_id": run_id, "model": model, "held_out": held_out, "outcome": outcome,
        "scenario": {"letter": letter, "passed": passed,
                     "checks": checks if checks is not None else {"answer gives 60": passed}},
    }


def _run(run_id, calls):
    return {"run_id": run_id, "calls": calls}


SYSTEM = {"role": "system", "content": [{"kind": "text", "text": "You are an assistant."}]}
USER = {"role": "user", "content": [{"kind": "text", "text": "run stats/total.py"}]}
CALL = {"role": "assistant", "content": [], "tool_calls": [
    {"id": "c1", "name": "run_command", "arguments": {"command": "python total.py"}}]}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": [{"kind": "text", "text": "exit code: 0\n60"}]}
TOOLS = [{"type": "function", "function": {"name": "run_command", "parameters": {"type": "object"}}}]


def _call(index, messages, completion):
    return {"run_id": "r", "thread_id": "t", "call_index": index, "model": "glm",
            "messages": messages, "tools": TOOLS,
            "completion": {"text": completion.get("text", ""), "tool_calls": completion.get("tool_calls", []),
                           "finish_reason": "stop", "usage": {"input_tokens": 100, "output_tokens": 10}}}


def test_one_sample_per_call_with_prompt_completion_and_tools(tmp_path):
    calls = [
        _call(1, [SYSTEM, USER], {"tool_calls": [{"id": "c1", "name": "run_command",
                                                  "arguments": {"command": "python total.py"}}]}),
        _call(2, [SYSTEM, USER, CALL, RESULT], {"text": "It prints 60."}),
    ]
    export = _export(tmp_path, [_entry("r")], [_run("r", calls)])
    stats = convert(export, tmp_path / "out", Filter())
    lines = [json.loads(l) for l in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats.samples == 2 and stats.runs_kept == 1
    first, second = lines
    assert first["prompt"] == [{"role": "system", "content": "You are an assistant."},
                               {"role": "user", "content": "run stats/total.py"}]
    assert first["completion"][0]["tool_calls"][0]["function"] == {
        "name": "run_command", "arguments": '{"command": "python total.py"}'}
    assert first["tools"] == TOOLS
    assert second["prompt"][2]["tool_calls"][0]["id"] == "c1"
    assert second["prompt"][3] == {"role": "tool", "content": "exit code: 0\n60",
                                   "tool_call_id": "c1", "name": "run_command"}
    assert second["completion"] == [{"role": "assistant", "content": "It prints 60."}]
    assert second["letter"] == "D4" and second["call_index"] == 2
    assert stats.prompt_tokens == 200 and stats.completion_tokens == 20
    assert json.loads((tmp_path / "out" / "stats.json").read_text())["samples"] == 2


def test_filter_teacher_held_out_outcome_and_checks(tmp_path):
    entries = [
        _entry("keep"),
        _entry("gemma", model="gemma-4-12b-it"),
        _entry("held", held_out=True),
        _entry("failed", passed=False),
        _entry("cut", outcome="failed"),
    ]
    runs = [_run(e["run_id"], [_call(1, [SYSTEM, USER], {"text": "60"})]) for e in entries]
    stats = convert(_export(tmp_path, entries, runs), tmp_path / "out", Filter())
    assert stats.runs_kept == 1 and stats.samples == 1
    assert stats.dropped_runs == {"other model": 1, "held out": 1, "checks failed": 1, "outcome failed": 1}


def test_ignored_check_does_not_count(tmp_path):
    checks = {"the answer gives 3": True, "the answer does not claim five passed": False}
    entry = _entry("v7", passed=False, checks=checks, letter="V7")
    run = _run("v7", [_call(1, [SYSTEM, USER], {"text": "3"})])
    export = _export(tmp_path, [entry], [run])
    assert convert(export, tmp_path / "a", Filter()).runs_kept == 0
    kept = convert(export, tmp_path / "b", Filter(ignore_checks=("the answer does not claim five passed",)))
    assert kept.runs_kept == 1 and kept.by_letter == {"V7": 1}


def test_media_call_is_dropped_not_reconstructed(tmp_path):
    image = {"role": "user", "content": [{"kind": "image", "size": 1234}]}
    calls = [_call(1, [SYSTEM, image], {"text": "a cat"}), _call(2, [SYSTEM, USER], {"text": "60"})]
    stats = convert(_export(tmp_path, [_entry("r")], [_run("r", calls)]), tmp_path / "out", Filter())
    assert stats.samples == 1 and stats.samples_with_media == 1


def test_helpers():
    assert text_of(None) == ""
    assert text_of("plain") == "plain"
    assert text_of([{"kind": "text", "text": "a"}, {"kind": "text", "text": "b"}]) == "a\nb"
    assert text_of([{"kind": "audio", "size": 1}]) is None
    assert tool_call_of({"id": "x", "name": "f", "arguments": "{\"a\": 1}"})["function"]["arguments"] == '{"a": 1}'
    assert messages_of([{"role": "tool", "tool_call_id": "nope", "content": "r"}]) == [
        {"role": "tool", "content": "r", "tool_call_id": "nope"}]
