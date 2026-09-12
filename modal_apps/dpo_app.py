"""DPO on the loop pairs, two A10s sharded by FSDP, on Modal.

Every function is a priced worker and starts only on the human's word:

    modal volume put pinocchio-tune data/pairs/v1/dpo_judged.jsonl /data/pairs-v1/dpo.jsonl
    modal run modal_apps/dpo_app.py::tokenize_pairs --set pairs-v1          # CPU, minutes
    modal run modal_apps/dpo_app.py::train --set pairs-v1 --run dpo-v1 --max-steps 2   # the smoke
    modal run --detach modal_apps/dpo_app.py::train --set pairs-v1 --run dpo-v1   # the run
    modal run --detach modal_apps/dpo_app.py::train --set pairs-v1 --run dpo-v1 --resume   # after a break
    modal run modal_apps/train_app.py::merge --run dpo-v1                   # CPU, as before

State on the Volume `pinocchio-tune`, beside the SFT runs:

    /vol/data/<set>/dpo.jsonl          the judged pairs (tune/judge_pairs.py)
    /vol/data/<set>/tokenized/         Arrow: prompt, chosen and rejected ids
    /vol/runs/<run>/adapter/           the LoRA; trainer/checkpoint-N every 20 %
    /vol/runs/<run>/train.json         the record

The base, image, LoRA shape and rendering are `train_app.py`'s; `train`
here runs `modal_apps/dpo_train.py` under torchrun, one process per GPU,
so that FSDP shards the bf16 base: 24 GB does not fit one A10's 24 GB,
half of it does. `TUNE_GPU=A10:4` for the optional four-card point.
"""

from __future__ import annotations

import os

import modal

from modal_apps.train_app import (
    BASE_REPO,
    LORA,
    MAX_TOKENS,
    MINUTES,
    VOL,
    _data_dir,
    _render,
    _run_dir,
    hf_secret,
    image,
    volume,
)

APP_NAME = "pinocchio-tune-dpo"
GPU = os.environ.get("TUNE_GPU", "A10:2")

app = modal.App(APP_NAME)
dpo_image = image.add_local_python_source("modal_apps")


def gpus_of(spec: str) -> int:
    """`A10:2` → 2; `A10` → 1."""
    _, _, count = spec.partition(":")
    return int(count) if count else 1


def render_pair(tokenizer, pair: dict) -> dict | str:
    """Prompt, chosen and rejected token ids through Gemma's template, or why not.

    The prompt is rendered with the generation prompt, as vLLM builds it;
    each completion is the target `split_render` cuts after that prompt,
    ending in the model's own stop token, so the two completions the loss
    compares are exactly the two things the model would have emitted.
    """
    prompt_ids = None
    ids = {}
    for side in ("chosen", "rejected"):
        rendered = _render(tokenizer, {"prompt": pair["prompt"], "completion": pair[side], "tools": pair["tools"]})
        if isinstance(rendered, str):
            return f"{side}: {rendered}"
        all_ids, cut = rendered
        if prompt_ids is None:
            prompt_ids = all_ids[:cut]
        elif all_ids[:cut] != prompt_ids:
            return "the prompt rendered differently for the two sides"
        ids[side] = all_ids[cut:]
    return {"prompt_input_ids": prompt_ids, "chosen_input_ids": ids["chosen"], "rejected_input_ids": ids["rejected"]}


@app.function(image=dpo_image, volumes={VOL: volume}, secrets=[hf_secret], cpu=2, memory=8192, timeout=30 * MINUTES)
def tokenize_pairs(set: str = "pairs-v1") -> dict:
    """Render every pair; write the Arrow dataset DPOTrainer takes pre-tokenized."""
    import json
    from collections import Counter

    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_REPO)
    rows, dropped, lengths = [], Counter(), []
    with open(f"{_data_dir(set)}/dpo.jsonl", encoding="utf-8") as handle:
        for line in handle:
            pair = json.loads(line)
            try:
                rendered = render_pair(tokenizer, pair)
            except Exception as error:  # noqa: BLE001 - the reason is the output
                rendered = f"{type(error).__name__}: {str(error)[:120]}"
            if isinstance(rendered, str):
                dropped[rendered] += 1
                continue
            rows.append({**rendered, "run_id": pair["run_id"], "call_index": pair["call_index"], "letter": pair["letter"]})
            lengths.append(len(rendered["prompt_input_ids"]) + max(len(rendered["chosen_input_ids"]), len(rendered["rejected_input_ids"])))
    out = f"{_data_dir(set)}/tokenized"
    Dataset.from_list(rows).save_to_disk(out)
    report = {
        "set": set,
        "pairs": len(rows),
        "dropped": dict(dropped),
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) // len(lengths) if lengths else 0,
        "chosen_tokens": sum(len(r["chosen_input_ids"]) for r in rows),
        "rejected_tokens": sum(len(r["rejected_input_ids"]) for r in rows),
    }
    with open(f"{_data_dir(set)}/tokenized.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    volume.commit()
    print(json.dumps(report, indent=2), flush=True)
    for row in rows[:2]:
        print("--- prompt tail ---", repr(tokenizer.decode(row["prompt_input_ids"][-40:])), flush=True)
        print("--- chosen ---", repr(tokenizer.decode(row["chosen_input_ids"])), flush=True)
        print("--- rejected ---", repr(tokenizer.decode(row["rejected_input_ids"])), flush=True)
    return report


@app.function(
    image=dpo_image,
    volumes={VOL: volume},
    secrets=[hf_secret],
    gpu=GPU,
    # Two cores and 16 GiB, as v1 measured (the human, 2026-09-12). Modal's
    # memory is a reservation, not a cap: the two torchrun processes reading
    # the weights before FSDP shards them may use more for a moment.
    cpu=2,
    memory=16384,
    timeout=4 * 60 * MINUTES,
)
def train(set: str = "pairs-v1", run: str = "dpo-v1", max_steps: int = -1, epochs: float = 3.0,
          lr: float = 5e-5, beta: float = 0.1, accumulation: int = 2, resume: bool = False) -> str:
    """torchrun over every GPU of the container; `--max-steps 2` is the smoke;
    `--resume` continues from the run's last checkpoint (adapter, optimizer,
    scheduler, data order). Start it with `modal run --detach`: the first
    full run was cancelled when the local client went away (2026-09-12)."""
    import json
    import subprocess
    import sys

    import torch

    gpus = torch.cuda.device_count()
    out = _run_dir(run)
    os.makedirs(out, exist_ok=True)
    command = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}", "--standalone",
        "-m", "modal_apps.dpo_train",
        "--base", BASE_REPO, "--data", f"{_data_dir(set)}/tokenized", "--out", out,
        "--epochs", str(epochs), "--max-steps", str(max_steps), "--lr", str(lr), "--beta", str(beta),
        "--accumulation", str(accumulation), "--max-length", str(MAX_TOKENS), "--lora", json.dumps(LORA),
    ] + (["--resume"] if resume else [])
    print(f"{gpus} x {torch.cuda.get_device_name(0)}: {' '.join(command[3:])}", flush=True)
    # Growable segments: the third smoke had 2.7 GB reserved and unusable
    # beside 20 GB allocated on a 23.5 GB card.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    subprocess.run(command, check=True, env=env)
    volume.commit()
    return f"{out}/train.json"
