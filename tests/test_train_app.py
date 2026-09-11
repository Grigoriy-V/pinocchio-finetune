from modal_apps.train_app import EMPTY_THOUGHT, _completion_as_gemma, mapping_arguments, split_render


def test_arguments_become_mappings_and_nothing_else_changes():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": {"b": 2}}}]},
    ]
    out = mapping_arguments(messages)
    assert out[0] is messages[0]
    assert out[1]["tool_calls"][0]["function"] == {"name": "f", "arguments": {"a": 1}}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"b": 2}
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_split_first_call_with_empty_thought_channel():
    prompt = "<|turn>user\nhi<turn|>\n<|turn>model\n" + EMPTY_THOUGHT
    call = '<|tool_call>call:f{a:<|"|>1<|"|>}<tool_call|>'
    full = "<|turn>user\nhi<turn|>\n<|turn>model\n" + call + "<|tool_response>"
    assert split_render(prompt, full) == (prompt, call)


def test_split_text_after_tool_response_keeps_turn_end_drops_newline():
    prompt = '...<|tool_response>response:f{value:<|"|>60<|"|>}<tool_response|>'
    full = prompt + "It prints 60.<turn|>\n"
    assert split_render(prompt, full) == (prompt, "It prints 60.<turn|>")


def test_split_closed_turn_is_dropped_with_reason():
    prompt = "...<tool_response|>I'll fix it.<turn|>\n"
    full = "...<tool_response|>I'll fix it.<|tool_call>call:f{}<tool_call|><|tool_response>"
    assert split_render(prompt, full) == "the template closed the turn before the completion"


def test_split_empty_target():
    assert split_render("<|turn>model\n" + EMPTY_THOUGHT, "<|turn>model\n\n") == "empty completion"


def test_call_with_text_becomes_the_call_alone():
    calls = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert _completion_as_gemma([{"role": "assistant", "content": "I'll run it", "tool_calls": calls}]) == [
        {"role": "assistant", "content": "", "tool_calls": calls}]
    assert _completion_as_gemma([{"role": "assistant", "content": "done"}]) == [
        {"role": "assistant", "content": "done"}]
