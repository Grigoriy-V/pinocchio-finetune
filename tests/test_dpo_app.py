import json

from modal_apps.dpo_app import gpus_of, render_pair
from modal_apps.dpo_train import save_every, steps_of


class Template:
    """A stand-in for Gemma's tokenizer: one token per character, a
    generation prompt that ends the way Gemma's does."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        text = "".join(
            ("<|turn>model\n" if m["role"] == "assistant" else f"<{m['role']}>")
            + f"{m.get('content', '')}{json.dumps(m.get('tool_calls', []))}"
            for m in messages
        )
        return text + ("<|turn>model\n<|channel>thought\n<channel|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_gpus_of_the_spec():
    assert gpus_of("A10:2") == 2 and gpus_of("A10") == 1 and gpus_of("H100:4") == 4


def test_render_pair_shares_the_prompt_and_splits_the_two_completions():
    pair = {
        "prompt": [{"role": "user", "content": "build it"}],
        "chosen": [{"role": "assistant", "content": "It is built."}],
        "rejected": [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "run_command", "arguments": '{"command": "sh b"}'}}]}],
        "tools": [],
    }
    rendered = render_pair(Template(), pair)
    assert isinstance(rendered, dict)
    prompt = "".join(chr(i) for i in rendered["prompt_input_ids"])
    assert prompt.startswith("<user>build it") and prompt.endswith("<channel|>")
    chosen = "".join(chr(i) for i in rendered["chosen_input_ids"])
    rejected = "".join(chr(i) for i in rendered["rejected_input_ids"])
    assert chosen == "It is built.[]"
    assert rejected.startswith("[{") and '"run_command"' in rejected and '"sh b"' in rejected


def test_steps_and_checkpoints_every_fifth():
    assert steps_of(16, 3, gpus=2, per_device=1, accumulation=2) == 12
    assert steps_of(16, 1, gpus=2, per_device=1, accumulation=2) == 4
    assert save_every(12) == 2 and save_every(4) == 1 and save_every(58) == 12
