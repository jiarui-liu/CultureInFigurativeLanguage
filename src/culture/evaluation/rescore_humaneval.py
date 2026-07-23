#!/usr/bin/env python3
"""Re-score stored HumanEval completions with the indentation fix (no regeneration).

HFModel.generate() strips leading whitespace, so the saved completions need one
level of indentation re-added before execution. Reads the stored humaneval.json
for each run, re-assembles + executes against the unit tests, and rewrites the
metrics into humaneval.json and en_gen/summary.json.
"""
import json, os, subprocess, sys, tempfile
from datasets import load_dataset

OUT = "/lustre-storage/fsx_it_0/users/jiaruiliu/culture_pretraining/eval/hi"
NL = "\n"


def run(prog, timeout=12.0):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    ds = {r["task_id"]: r for r in load_dataset("openai/openai_humaneval", split="test",
                                                token=os.environ.get("HF_TOKEN"))}
    for run_name in ["base", "cpt"]:
        hp = f"{OUT}/{run_name}/en_gen/humaneval.json"
        d = json.load(open(hp))
        recs = d["records"]
        passed = 0
        for rec in recs:
            r = ds[rec["task_id"]]
            c = rec["completion"]
            body = c if c[:1] in (" ", "\t") else ("    " + c)
            check = "check(" + r["entry_point"] + ")" + NL
            prog = r["prompt"] + body + NL + NL + r["test"] + NL + NL + check
            ok = run(prog)
            rec["passed"] = ok
            passed += ok
        n = len(recs)
        p1 = round(passed / n, 4)
        d["metrics"]["pass@1"] = p1
        d["metrics"]["primary"] = p1
        json.dump(d, open(hp, "w"), ensure_ascii=False, indent=2)
        sp = f"{OUT}/{run_name}/en_gen/summary.json"
        s = json.load(open(sp))
        s["tasks"]["humaneval"] = d["metrics"]
        json.dump(s, open(sp, "w"), ensure_ascii=False, indent=2)
        print(f"{run_name}: HumanEval pass@1 = {p1}  ({passed}/{n})")


if __name__ == "__main__":
    main()
