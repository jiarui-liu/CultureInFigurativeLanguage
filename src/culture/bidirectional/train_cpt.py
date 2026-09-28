#!/usr/bin/env python3
"""Small-scale continued pretraining on a packed token memmap (one epoch).

All arms of the bidirectional study use this script with identical
hyper-parameters; only ``--data`` differs. Launch with torchrun/deepspeed, e.g.
  torchrun --nproc_per_node 4 -m culture.bidirectional.train_cpt \
      --model <base> --data <prefix> --out <dir> --deepspeed ds_z2.json
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)


class PackedDataset(Dataset):
    def __init__(self, prefix):
        man = json.load(open(prefix + ".json"))
        self.seq_len = man["seq_len"]
        self.n = man["n_blocks"]
        self.arr = np.memmap(prefix + ".bin", dtype=np.uint32, mode="r",
                             shape=(self.n, self.seq_len))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(self.arr[i].astype(np.int64))
        return {"input_ids": x, "labels": x.clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--global_batch", type=int, default=128, help="sequences per step")
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=150)
    ap.add_argument("--ckpt_dir", default=None,
                    help="where intermediate (resumable) checkpoints go; default: --out")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1))
    accum = max(1, a.global_batch // (a.micro_batch * world))
    ds = PackedDataset(a.data)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    total = a.max_steps if a.max_steps > 0 else math.ceil(len(ds) * a.epochs / a.global_batch)
    warmup = max(1, int(0.03 * total))

    ckdir = a.ckpt_dir or a.out
    args = TrainingArguments(
        output_dir=ckdir, per_device_train_batch_size=a.micro_batch,
        gradient_accumulation_steps=accum, learning_rate=a.lr,
        lr_scheduler_type="cosine", warmup_steps=warmup, weight_decay=0.01,
        adam_beta1=0.9, adam_beta2=0.999, max_grad_norm=1.0,
        num_train_epochs=a.epochs, max_steps=a.max_steps, bf16=True,
        gradient_checkpointing=True, logging_steps=10, save_strategy="steps",
        save_steps=a.save_steps, save_total_limit=1, save_only_model=False,
        report_to="none", seed=a.seed, data_seed=a.seed, dataloader_num_workers=2,
        deepspeed=a.deepspeed, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=default_data_collator)
    last = None
    if os.path.isdir(ckdir):
        cks = sorted((d for d in os.listdir(ckdir) if d.startswith("checkpoint-")),
                     key=lambda d: int(d.split("-")[1]))
        last = os.path.join(ckdir, cks[-1]) if cks else None
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    if trainer.is_world_process_zero():
        json.dump({"data": a.data, "blocks": len(ds), "seq_len": ds.seq_len, "lr": a.lr,
                   "global_batch": a.global_batch, "seed": a.seed,
                   "steps": trainer.state.global_step,
                   "log": trainer.state.log_history},
                  open(os.path.join(a.out, "train_manifest.json"), "w"), indent=1)
        import shutil
        for d in os.listdir(ckdir):
            if d.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(ckdir, d), ignore_errors=True)


if __name__ == "__main__":
    main()
