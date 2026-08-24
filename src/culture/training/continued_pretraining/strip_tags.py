#!/usr/bin/env python3
"""Build the FILTERED-UNTAGGED corpus (paper point 6): the SAME idiom-bearing
filtered documents as the augmented run, but with the appended meaning/knowledge
block STRIPPED. Comparing augmented vs filtered-untagged isolates the effect of
the meaning tags; comparing filtered-untagged vs unfiltered isolates document
selection.

Per-language strip rule (robust, format-specific):
  hi : text[:original_text_chars]  (the field is stored in each hi tagged doc)
  zh : split on the chengyu-annotation header  \n\n【成语注释】
  ar : split on the amthal knowledge-block header  \n\nالمعاني الاصطلاحية للتعابير الواردة في النص:

Reads tagged_*.json.gz, writes plain train_*.jsonl (LLaMA-Factory reads "text").

Usage:
  python strip_tags.py --lang hi --src_dir <DATA>/hi-proverbs-cpt/data --out_dir <DATA>/train_hi_untagged
  python strip_tags.py --lang zh --src_dir <DATA>/fineweb-edu-zh-chengyu-cpt/data --out_dir <DATA>/train_zh_untagged
  python strip_tags.py --lang ar --src_dir <DATA>/ar-amthal-cpt/data/fineweb2_par --out_dir <DATA>/train_ar_untagged
"""
import argparse, glob, gzip, json, os, sys

AR_HDR = "\n\nالمعاني الاصطلاحية للتعابير الواردة في النص:"
ZH_HDR = "\n\n【成语注释】"


def strip(lang, doc):
    t = doc.get("text")
    if not isinstance(t, str) or not t.strip():
        return None
    if lang == "hi":
        n = doc.get("original_text_chars")
        if isinstance(n, int) and 0 < n <= len(t):
            t = t[:n]
        else:  # fallback: no field -> keep as-is (should not happen for hi)
            pass
    elif lang == "zh":
        t = t.split(ZH_HDR)[0]
    elif lang == "ar":
        t = t.split(AR_HDR)[0]
    else:
        raise ValueError(lang)
    t = t.rstrip()
    return t if t.strip() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hi", "zh", "ar"])
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--glob", default=None, help="shard glob; default tagged_*.json.gz")
    p.add_argument("--docs_per_shard", type=int, default=20000)
    args = p.parse_args()

    pattern = args.glob or "tagged_*.json.gz"
    shards = sorted(glob.glob(os.path.join(args.src_dir, pattern)))
    if not shards:
        print(f"FATAL: no {pattern} under {args.src_dir}", file=sys.stderr); sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.out_dir, "train_*.jsonl")):
        os.remove(old)

    n_in = n_out = out_idx = 0
    orig_chars = strip_chars = 0
    buf = []

    def flush():
        nonlocal out_idx, buf
        if not buf: return
        with open(os.path.join(args.out_dir, f"train_{out_idx:05d}.jsonl"), "w", encoding="utf-8") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out_idx += 1; buf = []

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_in += 1
                orig_chars += len(d.get("text", ""))
                s = strip(args.lang, d)
                if s is None: continue
                strip_chars += len(s)
                buf.append({"text": s, "source": d.get("source", f"{args.lang}-untagged")})
                n_out += 1
                if len(buf) >= args.docs_per_shard:
                    flush()
    flush()
    rep = {"lang": args.lang, "shards_in": len(shards), "docs_in": n_in, "docs_out": n_out,
           "shards_out": out_idx, "orig_chars": orig_chars, "kept_chars": strip_chars,
           "frac_chars_kept": round(strip_chars / max(orig_chars, 1), 4)}
    # IMPORTANT: write the report OUTSIDE out_dir. A stray non-.jsonl file inside the
    # dataset dir makes LLaMA-Factory's loader fail ("File types should be identical").
    rep_path = args.out_dir.rstrip("/") + ".strip_report.json"
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
