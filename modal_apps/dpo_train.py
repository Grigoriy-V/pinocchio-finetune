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

The loss is this file's, not the trainer's. TRL's `_compute_loss` asks the
model for the logits of every position and Gemma's vocabulary is 262k: on
a 7k-token pair that is ~7 GB per sequence in bf16, three times over
(policy, its gradient, the reference), beside a 12 GB shard of the
weights — not on a 24 GB card. Gemma's forward takes `logits_to_keep`, a
tensor of positions, so `LoopDPOTrainer` asks for the few dozen positions
that predict a completion token and computes the sigmoid DPO loss on
those; the reference pass is the same forward with the adapter disabled.
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


def positions_to_keep(completion_mask):
    """The positions whose logits predict a completion token, over the whole
    batch: position p predicts token p+1, so p is kept when any row's mask
    is 1 at p+1. Sorted, unique, as a tensor for `logits_to_keep`."""
    import torch

    return torch.unique(torch.nonzero(completion_mask[:, 1:])[:, 1])


def sequence_logps(logits, input_ids, completion_mask, keep):
    """Sum of the log-probabilities of the completion tokens per row, from
    logits computed at the kept positions only."""
    from trl.trainer.utils import selective_log_softmax

    labels = input_ids[:, keep + 1]
    mask = completion_mask[:, keep + 1]
    per_token = selective_log_softmax(logits, labels)
    return (per_token * mask).sum(dim=1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta: float):
    """The sigmoid DPO loss and its reward margins, mean over the batch."""
    import torch.nn.functional as F

    margins = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margins).mean(), margins


def build_trainer_class():
    """The subclass, built at call time so that the module imports without TRL."""
    import torch
    from trl import DPOTrainer
    from trl.trainer.utils import use_adapter

    class LoopDPOTrainer(DPOTrainer):
        def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
            # Already tokenized through Gemma's template by `tokenize_pairs`.
            return dataset

        def _compute_loss(self, model, inputs, return_outputs):
            mode = "train" if self.model.training else "eval"
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            completion_mask = inputs["completion_mask"]
            keep = positions_to_keep(completion_mask)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False, "logits_to_keep": keep}
            outputs = model(**kwargs)
            logps = sequence_logps(outputs.logits, input_ids, completion_mask, keep)
            chosen, rejected = logps.chunk(2, dim=0)
            if "ref_chosen_logps" in inputs:
                ref_chosen, ref_rejected = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
            else:
                with torch.no_grad():
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    with use_adapter(unwrapped, None):
                        ref_logits = model(**kwargs).logits
                    ref_logps = sequence_logps(ref_logits, input_ids, completion_mask, keep)
                ref_chosen, ref_rejected = ref_logps.chunk(2, dim=0)
            loss, margins = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, self.beta)
            metrics = self._metrics[mode]
            gathered = self.accelerator.gather(margins.detach())
            metrics["rewards/margins"].append(gathered.mean().item())
            metrics["rewards/accuracies"].append((gathered > 0).float().mean().item())
            metrics["logps/chosen"].append(self.accelerator.gather(chosen.detach()).mean().item())
            metrics["logps/rejected"].append(self.accelerator.gather(rejected.detach()).mean().item())
            metrics["logits/positions_kept"].append(float(keep.numel()))
            return (loss, outputs) if return_outputs else loss

    return LoopDPOTrainer


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
    from trl import DPOConfig

    started = time.time()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dataset = load_from_disk(args.data)
    names = {"prompt_input_ids": "prompt_ids", "chosen_input_ids": "chosen_ids", "rejected_input_ids": "rejected_ids"}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in names])
    for old, new in names.items():
        dataset = dataset.rename_column(old, new)
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
        warmup_steps=max(1, round(total * 0.1)),
        beta=args.beta,
        max_length=args.max_length,
        precompute_ref_log_probs=False,
        bf16=True,
        # TRL turns the model's own gradient checkpointing on by default;
        # under FSDP the checkpointing is FSDP's (`activation_checkpointing`
        # below) and transformers refuses both at once.
        gradient_checkpointing=False,
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
    trainer = build_trainer_class()(
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
