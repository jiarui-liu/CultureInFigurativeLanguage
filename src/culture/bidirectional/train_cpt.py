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
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments,
                          default_data_collator)


MIN_FREE_GB = 55  # one 2B ZeRO-2 checkpoint is ~30 GB; leave room for final model writes


def latest_ckpt(d):
    if not os.path.isdir(d):
        return None
    cks = sorted((x for x in os.listdir(d) if x.startswith("checkpoint-")),
                 key=lambda x: int(x.split("-")[1]))
    return cks[-1] if cks else None


def restore_from_shared(ckdir, resume_dir):
    """Copy the last mirrored checkpoint back to the node-local ckpt dir (local rank 0 only)."""
    tag = "_".join(os.environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURM_RESTART_COUNT", "MASTER_PORT"))
    done = os.path.join(ckdir, f".restore_done_{tag}")
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        while not os.path.exists(done):
            time.sleep(10)
        return
    os.makedirs(ckdir, exist_ok=True)
    marker = os.path.join(resume_dir, "COMPLETE")
    if latest_ckpt(ckdir) is None and os.path.exists(marker):
        name = open(marker).read().strip()
        src = os.path.join(resume_dir, name)
        if os.path.isdir(src):
            print(f"restoring {src} -> {ckdir}", flush=True)
            shutil.copytree(src, os.path.join(ckdir, name + ".tmp"))
            os.rename(os.path.join(ckdir, name + ".tmp"), os.path.join(ckdir, name))
    open(done, "w").close()


class MirrorCallback(TrainerCallback):
    """After each save, copy the new checkpoint to shared storage in the background.

    Node-local /scratch is lost when a preempted job restarts on another node; the
    mirror keeps exactly one complete checkpoint per run on NFS.
    """

    def __init__(self, ckdir, resume_dir):
        self.ckdir, self.resume_dir, self.proc = ckdir, resume_dir, None

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        name = latest_ckpt(self.ckdir)
        if name is None:
            return
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        r, src = self.resume_dir, os.path.join(self.ckdir, name)
        free_gb = shutil.disk_usage(os.path.dirname(os.path.dirname(r))).free / 2**30
        if free_gb < MIN_FREE_GB:
            print(f"skip mirroring {name}: only {free_gb:.0f} GB free", flush=True)
            return
        try:
            cur = open(os.path.join(r, "COMPLETE")).read().strip()
            if int(cur.split("-")[1]) >= int(name.split("-")[1]):
                return
        except (OSError, ValueError, IndexError):
            pass
        cmd = (f"mkdir -p {r} && rm -rf {r}/.tmp && cp -r {src} {r}/.tmp && "
               f"find {r} -maxdepth 1 -name 'checkpoint-*' -exec rm -rf {{}} + && "
               f"mv {r}/.tmp {r}/{name} && echo {name} > {r}/COMPLETE.tmp && "
               f"mv {r}/COMPLETE.tmp {r}/COMPLETE")
        self.proc = subprocess.Popen(["bash", "-c", cmd])

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()


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
    ap.add_argument("--resume_dir", default=None,
                    help="shared-storage mirror of the latest checkpoint; default: --out "
                         "with /ckpt/ replaced by /ckpt_resume/")
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
    resume_dir = a.resume_dir or a.out.replace("/ckpt/", "/ckpt_resume/")
    mirror = ckdir != a.out and resume_dir != a.out
    if mirror:
        restore_from_shared(ckdir, resume_dir)
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
    cb = MirrorCallback(ckdir, resume_dir) if mirror else None
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=default_data_collator,
                      callbacks=[cb] if cb else None)
    last = latest_ckpt(ckdir)
    trainer.train(resume_from_checkpoint=os.path.join(ckdir, last) if last else None)
    if cb:
        cb.stop()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    if trainer.is_world_process_zero():
        json.dump({"data": a.data, "blocks": len(ds), "seq_len": ds.seq_len, "lr": a.lr,
                   "global_batch": a.global_batch, "seed": a.seed,
                   "steps": trainer.state.global_step,
                   "log": trainer.state.log_history},
                  open(os.path.join(a.out, "train_manifest.json"), "w"), indent=1)
        if mirror:
            shutil.rmtree(resume_dir, ignore_errors=True)
        for d in os.listdir(ckdir):
            if d.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(ckdir, d), ignore_errors=True)


if __name__ == "__main__":
    main()
