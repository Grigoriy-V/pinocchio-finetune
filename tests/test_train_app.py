from modal_apps.train_app import (
    EMPTY_THOUGHT,
    _completion_as_gemma,
    mapping_arguments,
    segments_of,
    split_render,
)


def test_segments_of_a_run_mark_calls_stop_tokens_text_and_turn_end():
    call = '<|tool_call>call:f{a:<|"|>1<|"|>}<tool_call|>'
    response = '<|tool_response>response:f{value:<|"|>60<|"|>}<tool_response|>'
    full = ("<bos><|turn>system\nS<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n"
            + call + response + "It prints 60.<turn|>")
    assert segments_of(full) == [
        ("<bos><|turn>system\nS<turn|>\n<|turn>user\nhi<turn|>\n<|turn>model\n" + EMPTY_THOUGHT, False),
        (call + "<|tool_response>", True),
        ('response:f{value:<|"|>60<|"|>}<tool_response|>', False),
        ("It prints 60.<turn|>", True),
    ]
    assert "".join(t for t, _ in segments_of(full)).replace(EMPTY_THOUGHT, "") == full


def test_segments_of_parallel_calls_mark_only_the_first_opener():
    calls = "<|tool_call>call:f{}<tool_call|><|tool_call>call:g{}<tool_call|>"
    r1 = "<|tool_response>response:f{value:<|\"|>1<|\"|>}<tool_response|>"
    r2 = "<|tool_response>response:g{value:<|\"|>2<|\"|>}<tool_response|>"
    full = "<|turn>model\n" + calls + r1 + r2 + "3<turn|>"
    assert segments_of(full) == [
        ("<|turn>model\n" + EMPTY_THOUGHT, False),
        (calls + "<|tool_response>", True),
        (r1[len("<|tool_response>"):] + r2, False),
        ("3<turn|>", True),
    ]


def test_segments_of_two_model_turns_and_a_trailing_call():
    full = ("<|turn>user\na<turn|>\n<|turn>model\nx<turn|>\n<|turn>user\nb<turn|>\n<|turn>model\n"
            "<|tool_call>call:g{}<tool_call|><|tool_response>")
    segments = segments_of(full)
    assert segments == [
        ("<|turn>user\na<turn|>\n<|turn>model\n" + EMPTY_THOUGHT, False),
        ("x<turn|>", True),
        ("\n<|turn>user\nb<turn|>\n<|turn>model\n" + EMPTY_THOUGHT, False),
        ("<|tool_call>call:g{}<tool_call|><|tool_response>", True),
    ]


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
    # `<|tool_response>` stays: it is the stop token the model emits after a call.
    assert split_render(prompt, full) == (prompt, call + "<|tool_response>")


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
