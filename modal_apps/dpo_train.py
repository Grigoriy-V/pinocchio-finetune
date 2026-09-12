"""DPO of Gemma 4 12B on the loop pairs, sharded by FSDP — the script torchrun runs.

Started by `modal_apps/dpo_app.py::train` as
`torchrun --nproc_per_node=<gpus> -m modal_apps.dpo_train ...`, one process
per GPU. The policy is the bf16 base with a fresh LoRA of the v1 shape; the
reference is the same model with the adapter disabled (PEFT's default in
TRL), so no second copy of the weights is loaded. FSDP full-shards the
base over the GPUs: 12B in bf16 is ~24 GB, more than one A10 holds, and
half of it per card leaves room for the activations.

The dataset is pre-tokenized (`dpo_app.py::tokenize_pairs`) through the
same rendering the SFT path used, so the prompt is what vLLM builds at
inference and the completions end in the model's own stop tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time


def steps_of(samples: int, epochs: float, gpus: int, per_device: int, accumulation: int) -> int:
    """Optimizer steps the run will take: samples per step is the world's batch."""
    per_step = gpus * per_device * accumulation
    return math.ceil(samples * epochs / per_step)


def save_every(total_steps: int, fraction: float = 0.2) -> int:
    """A checkpoint every `fraction` of the run, at least every step."""
    return max(1, round(total_steps * fraction))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True, help="the tokenized Arrow dataset")
    parser.add_argument("--out", required=True, help="the run directory")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--lora", required=True, help="JSON of the LoRA config")
    args = parser.parse_args(argv)

    import torch
    import torch.distributed as dist
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    started = time.time()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dataset = load_from_disk(args.data)
    keep = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")

    total = steps_of(len(dataset), args.epochs, world, 1, args.accumulation) if args.max_steps < 0 else args.max_steps
    every = save_every(total)
    if rank == 0:
        print(f"{len(dataset)} pairs, {world} processes, {total} steps, a checkpoint every {every}", flush=True)

    config = DPOConfig(
        output_dir=f"{args.out}/trainer",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=None,
        truncation_mode="keep_end",
        # Logits only for the completion tokens: Gemma's vocabulary is 262k
        # and full logits over an 8k prompt would not fit beside the shards.
        use_logits_to_keep=True,
        precompute_ref_log_probs=False,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=every,
        save_total_limit=2,
        report_to="none",
        seed=3407,
        remove_unused_columns=False,
        # FSDP over every process torchrun started; the layer class to wrap
        # comes from the model's own `_no_split_modules`. Activation
        # checkpointing through FSDP, not the model's flag, as the Trainer
        # documents for sharded runs.
        fsdp="full_shard auto_wrap",
        fsdp_config={"use_orig_params": True, "activation_checkpointing": True, "cpu_ram_efficient_loading": False},
        ddp_find_unused_parameters=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**json.loads(args.lora)),
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"trainable parameters: {trainable:,}", flush=True)
    result = trainer.train()
    trainer.save_model(f"{args.out}/adapter")
    if rank == 0:
        tokenizer.save_pretrained(f"{args.out}/adapter")
        record = {
            "pairs": len(dataset),
            "processes": world,
            "gpu": torch.cuda.get_device_name(0),
            "steps": result.global_step,
            "planned_steps": total,
            "checkpoint_every": every,
            "train_loss": result.training_loss,
            "epochs": args.epochs,
            "lr": args.lr,
            "beta": args.beta,
            "accumulation": args.accumulation,
            "trainable_parameters": trainable,
            "log_history": trainer.state.log_history,
            "seconds": round(time.time() - started),
            "peak_vram_gib_rank0": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        with open(f"{args.out}/train.json", "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(json.dumps({k: v for k, v in record.items() if k != "log_history"}, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
