import json
from pathlib import Path

from tune.pairs import extract, letter_of, same_text
from tune.teach import chosen_of, teach

SYSTEM = {"role": "system", "content": [{"kind": "text", "text": "You are an assistant."}]}
USER = {"role": "user", "content": [{"kind": "text", "text": "build it"}]}
TOOLS = [{"type": "function", "function": {"name": "run_command", "parameters": {"type": "object"}}}]


def _call_msg(call_id, command):
    return {"role": "assistant", "content": [], "tool_calls": [
        {"id": call_id, "name": "run_command", "arguments": {"command": command}}]}


def _result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": [{"kind": "text", "text": text}]}


def _completion(call_id, command):
    return {"text": "", "tool_calls": [{"id": call_id, "name": "run_command", "arguments": {"command": command}}],
            "finish_reason": "tool_calls", "usage": {"input_tokens": 1, "output_tokens": 1}}


def _export(tmp_path: Path, model, runs):
    export = tmp_path / "export"
    (export / "runs").mkdir(parents=True, exist_ok=True)
    with (export / "index.jsonl").open("a", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps({"run_id": run["run_id"], "model": model, "thread_id": run.get("thread_id"),
                                "scenario": run.get("scenario")}) + "\n")
            (export / "runs" / f"{run['run_id']}.json").write_text(json.dumps({**run, "model": model}), encoding="utf-8")
    return export


def _looping_run(run_id, second_result, third_result=None, thread_id="chat-v1"):
    """build → ls → build again: the second build's result is `second_result`."""
    m1 = [SYSTEM, USER]
    m2 = m1 + [_call_msg("c1", "sh build.sh"), _result("c1", "exit code: 0   (2.1 s)\nbuilt")]
    m3 = m2 + [_call_msg("c2", "ls out"), _result("c2", "app")]
    m4 = m3 + [_call_msg("c3", "sh build.sh"), _result("c3", second_result)]
    calls = [
        {"call_index": 1, "messages": m1, "tools": TOOLS, "completion": _completion("c1", "sh build.sh")},
        {"call_index": 2, "messages": m2, "tools": TOOLS, "completion": _completion("c2", "ls out")},
        {"call_index": 3, "messages": m3, "tools": TOOLS, "completion": _completion("c3", "sh build.sh")},
        {"call_index": 4, "messages": m4, "tools": TOOLS,
         "completion": _completion("c4", "sh build.sh") if third_result else
         {"text": "Built.", "tool_calls": [], "finish_reason": "stop", "usage": {}}},
    ]
    if third_result:
        m5 = m4 + [_call_msg("c4", "sh build.sh"), _result("c4", third_result)]
        calls.append({"call_index": 5, "messages": m5, "tools": TOOLS,
                      "completion": {"text": "Built.", "tool_calls": [], "finish_reason": "stop", "usage": {}}})
    return {"run_id": run_id, "thread_id": thread_id, "calls": calls}


def test_repeat_with_the_same_result_is_a_pair_with_the_state_as_prompt(tmp_path):
    export = _export(tmp_path, "gemma-4-12b-it", [_looping_run("g1", "exit code: 0   (3.4 s)\nbuilt")])
    stats = extract([export], tmp_path / "out")
    pairs = [json.loads(l) for l in (tmp_path / "out" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats.pairs == 1 and stats.repeats_seen == 1 and stats.by_evidence == {"same result": 1}
    pair = pairs[0]
    assert pair["call_index"] == 3 and pair["repeat_of"] == 1 and pair["letter"] == "V1"
    assert [m["role"] for m in pair["prompt"]] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert pair["rejected"][0]["tool_calls"][0]["function"]["arguments"] == '{"command": "sh build.sh"}'
    assert pair["chosen"] is None and pair["tools"] == TOOLS


def test_guard_refusal_counts_as_nothing_changed(tmp_path):
    guard = "error: this exact call has already succeeded twice in this turn"
    export = _export(tmp_path, "gemma-4-12b-it", [_looping_run("g2", "exit code: 0\nbuilt", guard)])
    stats = extract([export], tmp_path / "out")
    assert stats.pairs == 2 and stats.by_evidence == {"same result": 1, "guard": 1}


def test_changed_result_is_not_a_loop_and_the_teacher_is_skipped(tmp_path):
    export = _export(tmp_path, "gemma-4-12b-it", [_looping_run("g3", "exit code: 1\nerror: missing")])
    _export(tmp_path, "z-ai/glm-5.3-flash", [_looping_run("t1", "exit code: 0   (3.4 s)\nbuilt")])
    stats = extract([export], tmp_path / "out")
    assert stats.pairs == 0 and stats.dropped == {"result changed": 1}
    assert stats.runs_seen == 2 and stats.runs_student == 1


def test_per_run_cap_and_a_run_seen_in_two_exports_once(tmp_path):
    run = _looping_run("g4", "exit code: 0\nbuilt", "exit code: 0\nbuilt")
    a = _export(tmp_path / "a", "gemma-4-12b-it", [run])
    b = _export(tmp_path / "b", "gemma-4-12b-it", [run])
    stats = extract([a, b], tmp_path / "out", per_run=1)
    assert stats.runs_seen == 1 and stats.pairs == 1 and stats.dropped == {"over per-run cap": 1}


def test_timing_stamp_and_letters():
    assert same_text("exit code: 0   (2.1 s)\nx", "exit code: 0   (13.0 s)\nx")
    assert not same_text("exit code: 0\nx", "exit code: 1\nx")
    assert letter_of({"scenario": {"letter": "D4"}, "thread_id": "chat-x1"}) == "D4"
    assert letter_of({"thread_id": "chat-x1"}) == "X1"
    assert letter_of({"thread_id": "telegram-5"}) == "?"


def _pair(run_id, index, command="sh build.sh"):
    return {"run_id": run_id, "call_index": index, "letter": "V1", "model": "gemma", "repeat_of": 1,
            "evidence": "same result", "prompt": [{"role": "user", "content": "build it"}],
            "rejected": [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "c3", "type": "function", "function": {"name": "run_command",
                                                             "arguments": json.dumps({"command": command})}}]}],
            "chosen": None, "tools": TOOLS}


def _reply(content="", calls=()):
    message = {"content": content, "tool_calls": [
        {"id": "t1", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]}
    return {"model": "teacher", "choices": [{"message": message}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5}}


def test_teacher_fills_chosen_and_drops_a_repeat_or_silence(tmp_path):
    answers = {
        "g1#3": _reply("The build already ran and made out/app."),
        "g1#5": _reply(calls=[("run_command", {"command": "sh build.sh"})]),
        "g2#3": _reply(),
    }
    pairs = [_pair("g1", 3), _pair("g1", 5), _pair("g2", 3)]
    calls = iter(pairs)
    stats = teach(pairs, tmp_path / "out", lambda m, t: answers[_next(calls)])
    assert stats == {"asked": 3, "kept": 1, "teacher_repeated": 1, "teacher_silent": 1, "already_taught": 0,
                     "input_tokens": 300, "output_tokens": 15, "dpo_pairs": 1}
    dpo = [json.loads(l) for l in (tmp_path / "out" / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dpo) == 1 and dpo[0]["chosen"][0]["content"].startswith("The build already ran")
    assert dpo[0]["rejected"] == pairs[0]["rejected"] and dpo[0]["tools"] == TOOLS
    # A rerun asks nothing: everything is in taught.jsonl.
    again = teach(pairs, tmp_path / "out", lambda m, t: (_ for _ in ()).throw(AssertionError("asked again")))
    assert again["asked"] == 0 and again["already_taught"] == 3 and again["dpo_pairs"] == 1


def _next(calls):
    p = next(calls)
    return f"{p['run_id']}#{p['call_index']}"


def test_chosen_of_shapes_the_teacher_message():
    reply = _reply("look", calls=[("read_file", {"path": "a"})])
    chosen = chosen_of(reply)
    assert chosen["content"] == "look" and chosen["tool_calls"][0]["function"] == {
        "name": "read_file", "arguments": '{"path": "a"}'}
    assert chosen_of({"choices": []}) is None
