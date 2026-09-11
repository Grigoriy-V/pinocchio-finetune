from modal_apps.train_app import mapping_arguments


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
