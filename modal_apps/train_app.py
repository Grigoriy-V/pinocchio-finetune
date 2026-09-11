"""QLoRA of Gemma 4 12B on the harness's trajectories, on Modal.

Every function here is a priced worker and starts only on the human's word.
Three steps, each its own gate:

    modal run modal_apps/train_app.py::fetch_base           # CPU, once
    modal run modal_apps/train_app.py::tokenize --set v1    # CPU, minutes
    modal run modal_apps/train_app.py::train --set v1 --run v1-r16 [--max-steps 5]
    modal run modal_apps/train_app.py::merge --run v1-r16   # CPU, large memory

State lives on the Volume `pinocchio-tune`:

    /vol/hf/                base weights (Hugging Face cache)
    /vol/data/<set>/train.jsonl        uploaded with `modal volume put`
    /vol/data/<set>/tokenized/         Arrow, from `tokenize`
    /vol/runs/<run>/adapter/           the LoRA
    /vol/runs/<run>/merged/            bf16 weights the serving app loads
    /vol/runs/<run>/train.json         what was trained, with what, how long

The base is `google/gemma-4-12B-it` (bf16), not the QAT int4 checkpoint the
harness serves: a LoRA over a compressed-tensors base has a known
correctness bug in vLLM at rank 32 (vllm issue 50059), and QLoRA's 4-bit
base is bitsandbytes NF4, a different quantization anyway. The merge is
therefore bf16, 24 GB, served on an L40S.

Gated repository: a Modal secret `huggingface` with `HF_TOKEN` must exist,
and the account must have accepted Gemma's terms. This file never reads
the token.
"""

from __future__ import annotations

import modal

APP_NAME = "pinocchio-tune"
BASE_REPO = "google/gemma-4-12B-it"
VOLUME = "pinocchio-tune"
VOL = "/vol"
MINUTES = 60

# The tokenizer's ceiling for a sample. The v1 set's longest call is ~8k
# teacher tokens; samples over the ceiling are dropped, never truncated,
# because a cut trajectory teaches a cut answer.
MAX_TOKENS = 12288

# LoRA as the research report settled: r=16, alpha=32, every linear of the
# language model, dropout 0.05, lr 2e-4 cosine, two epochs (report §3).
LORA = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    # Only the text stack. Gemma 4 12B is multimodal; the vision and audio
    # towers are frozen and kept, so the merged checkpoint still serves images.
    "target_modules": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    "task_type": "CAUSAL_LM",
}
TRAIN = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 5,
    "save_strategy": "no",
    "report_to": "none",
    "seed": 3407,
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==5.14.1",
        "trl>=0.29",
        "peft>=0.18",
        "bitsandbytes>=0.48",
        "datasets>=4.0",
        "accelerate>=1.10",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HOME": f"{VOL}/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
)


def _data_dir(set_name: str) -> str:
    return f"{VOL}/data/{set_name}"


def _run_dir(run: str) -> str:
    return f"{VOL}/runs/{run}"


@app.function(image=image, volumes={VOL: volume}, secrets=[hf_secret], cpu=8, timeout=60 * MINUTES)
def fetch_base() -> None:
    """Download the base weights on CPU, once."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASE_REPO)
    volume.commit()
    print(f"base at {path}", flush=True)


def mapping_arguments(messages: list[dict]) -> list[dict]:
    """The same messages with every tool call's arguments as a mapping.

    The samples keep arguments as JSON strings (flat Arrow schemas); Gemma's
    template refuses a string: "arguments must be a JSON object (mapping)".
    """
    import json

    out = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            out.append(message)
            continue
        fixed = []
        for call in calls:
            function = dict(call["function"])
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"] or "{}")
            fixed.append({**call, "function": function})
        out.append({**message, "tool_calls": fixed})
    return out


def _render(tokenizer, sample: dict) -> tuple[list[int], int] | str:
    """Token ids of prompt+completion and the prompt's length, or why not.

    The prompt is rendered with the generation prompt, the whole with the
    completion; the first must be a prefix of the second, otherwise the
    template does something to a finished assistant turn that it does not do
    to an open one and the mask would be wrong.
    """
    tools = sample["tools"] or None
    prompt = mapping_arguments(sample["prompt"])
    completion = mapping_arguments(sample["completion"])
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tools=tools, add_generation_prompt=True, tokenize=True
    )
    full_ids = tokenizer.apply_chat_template(
        prompt + completion, tools=tools, tokenize=True
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
        full_ids = full_ids["input_ids"]
    if full_ids[: len(prompt_ids)] != list(prompt_ids):
        return "prompt is not a prefix of prompt+completion"
    if len(full_ids) > MAX_TOKENS:
        return f"{len(full_ids)} tokens over the ceiling"
    if len(full_ids) == len(prompt_ids):
        return "empty completion"
    return list(full_ids), len(prompt_ids)


@app.function(image=image, volumes={VOL: volume}, secrets=[hf_secret], cpu=4, memory=16384, timeout=60 * MINUTES)
def tokenize(set: str = "v1") -> dict:
    """Render every sample through Gemma's chat template and mask the prompt.

    Writes an Arrow dataset with `input_ids` and `completion_mask`, which
    TRL takes as pre-tokenized input with the loss on the completion only.
    Prints what was dropped and why; a template that rejects a sample shape
    shows up here on CPU, not on the GPU.
    """
    import json
    from collections import Counter

    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
    rows, dropped, lengths = [], Counter(), []
    with open(f"{_data_dir(set)}/train.jsonl", encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            try:
                rendered = _render(tokenizer, sample)
            except Exception as error:  # noqa: BLE001 - the reason is the output
                rendered = f"{type(error).__name__}: {str(error)[:120]}"
            if isinstance(rendered, str):
                dropped[rendered] += 1
                continue
            ids, cut = rendered
            rows.append(
                {
                    "input_ids": ids,
                    "completion_mask": [0] * cut + [1] * (len(ids) - cut),
                    "run_id": sample["run_id"],
                    "call_index": sample["call_index"],
                    "letter": sample["letter"],
                }
            )
            lengths.append(len(ids))
    out = f"{_data_dir(set)}/tokenized"
    Dataset.from_list(rows).save_to_disk(out)
    report = {
        "set": set,
        "samples": len(rows),
        "dropped": dict(dropped),
        "tokens": sum(lengths),
        "completion_tokens": sum(sum(r["completion_mask"]) for r in rows),
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) // len(lengths) if lengths else 0,
    }
    with open(f"{_data_dir(set)}/tokenized.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    volume.commit()
    print(json.dumps(report, indent=2), flush=True)
    if rows:
        print("--- first sample, decoded ---", flush=True)
        print(tokenizer.decode(rows[0]["input_ids"])[:3000], flush=True)
    return report


@app.function(
    image=image,
    volumes={VOL: volume},
    secrets=[hf_secret],
    gpu="L40S",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * MINUTES,
)
def train(set: str = "v1", run: str = "v1-r16", max_steps: int = -1, epochs: float = 0) -> dict:
    """QLoRA on one L40S. `--max-steps 5` is the smoke run; the real one
    goes the epochs. Saves the adapter and a record of the run."""
    import json
    import time

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    started = time.time()
    dataset = load_from_disk(f"{_data_dir(set)}/tokenized")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ("input_ids", "completion_mask")])
    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_REPO,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    config = dict(TRAIN)
    if epochs:
        config["num_train_epochs"] = epochs
    out = _run_dir(run)
    args = SFTConfig(
        output_dir=f"{out}/trainer",
        max_steps=max_steps,
        max_length=MAX_TOKENS,
        completion_only_loss=True,
        use_liger_kernel=False,
        dataset_kwargs={"skip_prepare_dataset": False},
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **config,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**LORA),
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}", flush=True)
    result = trainer.train()
    trainer.model.save_pretrained(f"{out}/adapter")
    tokenizer.save_pretrained(f"{out}/adapter")
    record = {
        "run": run,
        "set": set,
        "base": BASE_REPO,
        "samples": len(dataset),
        "lora": LORA,
        "train": config,
        "max_steps": max_steps,
        "max_tokens": MAX_TOKENS,
        "gpu": "L40S",
        "trainable_parameters": trainable,
        "global_steps": result.global_step,
        "train_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "seconds": round(time.time() - started),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    with open(f"{out}/train.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    volume.commit()
    print(json.dumps({k: v for k, v in record.items() if k != "log_history"}, indent=2), flush=True)
    return {k: v for k, v in record.items() if k != "log_history"}


@app.function(image=image, volumes={VOL: volume}, secrets=[hf_secret], cpu=8, memory=98304, timeout=2 * 60 * MINUTES)
def merge(run: str = "v1-r16") -> str:
    """Fold the adapter into bf16 base weights, on CPU, for vLLM to serve.

    The adapter was trained over an NF4 base and is applied here to the bf16
    one: the usual QLoRA practice, and what the serving stack can load.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = _run_dir(run)
    model = AutoModelForImageTextToText.from_pretrained(BASE_REPO, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, f"{out}/adapter")
    model = model.merge_and_unload()
    model.save_pretrained(f"{out}/merged", safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(BASE_REPO).save_pretrained(f"{out}/merged")
    volume.commit()
    print(f"merged weights at {out}/merged", flush=True)
    return f"{out}/merged"
