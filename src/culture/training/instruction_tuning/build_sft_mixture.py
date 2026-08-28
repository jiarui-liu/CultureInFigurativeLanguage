#!/usr/bin/env python3
"""Build a globally-shuffled SFT mixture: one target-language dataset + English anchor.

Output: LLaMA-Factory `sharegpt` JSONL, one obj/line:
  {"conversations":[{"from":"human","value":...},{"from":"gpt","value":...}],
   "lang":"hi","source":"indic-align/wiki_chat"}

Target (hi -> IndicAlign hin_Deva; ar -> SmolKalam) : English (SmolTalk2) = 60:40, total 300K,
global shuffle seed=42. See ../SFT_PROGRESS.md for decisions/deviations.

Filtering (dependency-free, no fasttext/langdetect/datasketch):
  - normalize to alternating human/gpt sharegpt; system folded into first human
  - structural validity (non-empty, alternating, starts human, ends gpt)
  - strip <think>...</think> from assistant turns (keep data no-think-consistent)
  - per-row language via Unicode script ratio (target turns must be in-script; English must be Latin)
  - degeneration filter (repeated char 3-gram)
  - length cap (char proxy for cutoff_len)
  - near-dedup via normalized first-prompt hash
  - SmolKalam quality gate LR>=0.85 & SCR>=0.95
  - reservoir-sample to quota (fixed memory), then concat + shuffle(seed)
"""
import argparse, glob, gzip, hashlib, json, os, random, re, sys
import pyarrow.parquet as pq

SFT_ROOT_DEFAULT = "/lustre-storage/fsx_0/user/jiaruiliu/culture-sft-data"

# ---- unicode script ratios --------------------------------------------------
def _ratio(text, lo, hi, extra_ranges=()):
    if not text:
        return 0.0
    n = letters = 0
    for ch in text:
        o = ord(ch)
        if ch.isspace() or not ch.isalpha():
            continue
        letters += 1
        if lo <= o <= hi or any(a <= o <= b for a, b in extra_ranges):
            n += 1
    return n / letters if letters else 0.0

def deva_ratio(t):   return _ratio(t, 0x0900, 0x097F)
def arab_ratio(t):   return _ratio(t, 0x0600, 0x06FF, [(0x0750,0x077F),(0x08A0,0x08FF),(0xFB50,0xFDFF),(0xFE70,0xFEFF)])
def latin_ratio(t):  return _ratio(t, 0x0041, 0x007A, [(0x00C0,0x024F)])
def han_ratio(t):    return _ratio(t, 0x4E00, 0x9FFF, [(0x3400,0x4DBF),(0xF900,0xFAFF),(0x20000,0x2A6DF)])

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_think(s):
    return THINK_RE.sub("", s).strip() if s else s

# Neutralize literal media placeholder tokens (from source datasets, e.g. Codeforces
# problems). LlamaFactory's supervised processor treats <image>/<video>/<audio> as
# multimodal tokens and raises ValueError when no matching media is supplied.
MEDIA_RE = re.compile(r"<(image|video|audio)>", re.IGNORECASE)

def strip_media(s):
    return MEDIA_RE.sub(lambda m: "[" + m.group(1).lower() + "]", s) if s else s

def degenerate(text, thr=0.30, minlen=60):
    if not text or len(text) < minlen:
        return False
    grams = [text[i:i+3] for i in range(len(text)-2)]
    if not grams:
        return False
    from collections import Counter
    c = Counter(grams)
    return (c.most_common(1)[0][1] / len(grams)) > thr

def norm_key(s):
    return hashlib.md5(re.sub(r"\s+", " ", (s or "").strip().lower()).encode("utf-8")).hexdigest()

# ---- conversation validation ------------------------------------------------
def clean_conv(turns):
    """turns: list of (role, content) with role in human/gpt/system. Return valid sharegpt or None."""
    out, sys_prefix = [], ""
    for role, content in turns:
        content = strip_media(strip_think(content or "")).strip()
        if role == "system":
            sys_prefix = (sys_prefix + "\n" + content).strip() if content else sys_prefix
            continue
        if not content:
            return None
        out.append((role, content))
    if not out or out[0][0] != "human":
        return None
    if sys_prefix:
        out[0] = ("human", sys_prefix + "\n\n" + out[0][1])
    # enforce strict alternation human,gpt,human,gpt,...
    for i, (role, _) in enumerate(out):
        if role != ("human" if i % 2 == 0 else "gpt"):
            return None
    if len(out) % 2 != 0:  # must end on gpt
        return None
    return [{"from": r, "value": v} for r, v in out]

CHAR_CAP = 24000  # ~ cutoff_len 8192 tokens proxy; drops pathological long rows

def conv_ok_len(conv):
    return sum(len(t["value"]) for t in conv) <= CHAR_CAP

def target_script_ok(conv, ratio_fn, thr=0.60):
    # assistant (gpt) turns must be predominantly in target script
    gpts = [t["value"] for t in conv if t["from"] == "gpt"]
    return gpts and all(ratio_fn(g) >= thr for g in gpts)

# ---- reservoir sampler ------------------------------------------------------
class Reservoir:
    def __init__(self, k, rng):
        self.k, self.rng, self.n, self.buf = k, rng, 0, []
    def offer(self, item):
        self.n += 1
        if len(self.buf) < self.k:
            self.buf.append(item); return
        j = self.rng.randint(0, self.n - 1)
        if j < self.k:
            self.buf[j] = item

# ---- source iterators -------------------------------------------------------
def iter_smoltalk2_english(sft_root):
    files = sorted(glob.glob(f"{sft_root}/smoltalk2/SFT/*.parquet"))
    files = [f for f in files if "multilingual" not in os.path.basename(f).lower()
             and "_no_think" in os.path.basename(f).lower()]
    for f in files:
        src = "smoltalk2/" + re.sub(r"-\d+-of-\d+\.parquet$", "", os.path.basename(f))
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=2000, columns=["messages"]):
            for row in b.to_pylist():
                msgs = row.get("messages") or []
                turns = [( "human" if m["role"]=="user" else "gpt" if m["role"]=="assistant" else "system",
                           m["content"]) for m in msgs]
                yield turns, src

def iter_smolkalam(sft_root):
    files = sorted(glob.glob(f"{sft_root}/smolkalam/*/*.parquet"))
    for f in files:
        src = "smolkalam/" + os.path.basename(os.path.dirname(f))
        pf = pq.ParquetFile(f)
        cols = [c for c in ["messages","LR","SCR"] if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=2000, columns=cols):
            for row in b.to_pylist():
                lr, scr = row.get("LR"), row.get("SCR")
                if lr is not None and lr < 0.85:   continue
                if scr is not None and scr < 0.95: continue
                msgs = row.get("messages") or []
                turns = [( "human" if m["role"]=="user" else "gpt" if m["role"]=="assistant" else "system",
                           m["content"]) for m in msgs]
                yield turns, src

def iter_indicalign_hi(sft_root):
    files = sorted(glob.glob(f"{sft_root}/indic-align/**/*.parquet", recursive=True))
    for f in files:
        base = os.path.basename(f)
        src = "indic-align/" + os.path.basename(os.path.dirname(f))
        pf = pq.ParquetFile(f)
        if "hin_Deva" not in pf.schema_arrow.names:
            continue
        for b in pf.iter_batches(batch_size=1000, columns=["hin_Deva"]):
            for row in b.to_pylist():
                conv = row.get("hin_Deva")
                if not conv:
                    continue
                turns = []
                for pair in conv:                       # each pair = [user, assistant]
                    if not pair or len(pair) < 2:
                        turns = []; break
                    turns.append(("human", pair[0]))
                    turns.append(("gpt", pair[1]))
                if turns:
                    yield turns, src

def _rows_to_turns(row):
    """Map an Infinity-Instruct row to (role,content) turns. Handles sharegpt
    `conversations` (from/value) and OpenAI `messages` (role/content)."""
    conv = row.get("conversations")
    if isinstance(conv, list) and conv and isinstance(conv[0], dict) and "from" in conv[0]:
        role_map = {"human":"human","user":"human","gpt":"gpt","assistant":"gpt",
                    "system":"system","observation":"human","function":"gpt"}
        return [(role_map.get(str(m.get("from","")).lower(),"human"),
                 m.get("value") or m.get("content") or "") for m in conv]
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        return [("human" if m.get("role")=="user" else "gpt" if m.get("role")=="assistant" else "system",
                 m.get("content") or "") for m in msgs]
    return []

def iter_infinity_zh(sft_root):
    root = f"{sft_root}/infinity-instruct-zh"
    files = sorted(glob.glob(f"{root}/**/*.parquet", recursive=True))
    if files:
        for f in files:
            src = "infinity-instruct-zh"
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            cols = [c for c in ("conversations","messages") if c in names] or None
            for b in pf.iter_batches(batch_size=1000, columns=cols):
                for row in b.to_pylist():
                    turns = _rows_to_turns(row)
                    if turns:
                        yield turns, src
        return
    # fallback: jsonl / jsonl.gz shards
    for f in sorted(glob.glob(f"{root}/**/*.jsonl*", recursive=True)):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try: row = json.loads(line)
                except Exception: continue
                turns = _rows_to_turns(row)
                if turns:
                    yield turns, "infinity-instruct-zh"


def iter_quora_ar(sft_root):
    """Quora-Arabic-GPT4 (FreedomIntelligence): native Arabic Quora questions with
    GPT-4 answers. Handles sharegpt `conversations`, OpenAI `messages`, alpaca
    `instruction`/`output`, and `question`/`answer` schemas."""
    files = sorted(glob.glob(f"{sft_root}/quora-ar-gpt4/**/*.parquet", recursive=True))
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        for b in pf.iter_batches(batch_size=2000):
            for row in b.to_pylist():
                turns = _rows_to_turns(row)                       # conversations / messages
                if not turns:
                    instr = (row.get("instruction") or row.get("question") or "").strip()
                    inp = (row.get("input") or "").strip()
                    out = (row.get("output") or row.get("answer") or row.get("response") or "").strip()
                    if instr and out:
                        human = instr if not inp else f"{instr}\n\n{inp}"
                        turns = [("human", human), ("gpt", out)]
                if turns:
                    yield turns, "quora-ar-gpt4"


def iter_cidar(sft_root):
    """CIDAR: human-reviewed Arabic, alpaca-style (instruction[/input]/output).
    Map to a single human turn + single gpt turn."""
    files = sorted(glob.glob(f"{sft_root}/cidar/**/*.parquet", recursive=True))
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        cols = [c for c in ("instruction", "input", "output") if c in names]
        for b in pf.iter_batches(batch_size=2000, columns=cols):
            for row in b.to_pylist():
                instr = (row.get("instruction") or "").strip()
                inp = (row.get("input") or "").strip()
                out = (row.get("output") or "").strip()
                if not instr or not out:
                    continue
                human = instr if not inp else f"{instr}\n\n{inp}"
                yield [("human", human), ("gpt", out)], "cidar"


def build_side(name, it, quota, lang, script_fn, rng, seen):
    res = Reservoir(quota, rng)
    kept = scanned = dropped_struct = dropped_lang = dropped_dup = dropped_len = dropped_degen = 0
    for turns, src in it:
        scanned += 1
        conv = clean_conv(turns)
        if conv is None: dropped_struct += 1; continue
        if not conv_ok_len(conv): dropped_len += 1; continue
        if script_fn is not None and not target_script_ok(conv, script_fn):
            dropped_lang += 1; continue
        if degenerate(conv[-1]["value"]): dropped_degen += 1; continue
        k = norm_key(conv[0]["value"])
        if k in seen: dropped_dup += 1; continue
        seen.add(k)
        res.offer({"conversations": conv, "lang": lang, "source": src})
        kept += 1
        if scanned % 200000 == 0:
            print(f"  [{name}] scanned={scanned} kept={kept} reservoir={len(res.buf)}", flush=True)
    stats = dict(scanned=scanned, kept_unique=kept, sampled=len(res.buf),
                 dropped_struct=dropped_struct, dropped_lang=dropped_lang,
                 dropped_dup=dropped_dup, dropped_len=dropped_len, dropped_degen=dropped_degen)
    return res.buf, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["hi","ar","zh"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sft_root", default=SFT_ROOT_DEFAULT)
    ap.add_argument("--target_total", type=int, default=180000)
    ap.add_argument("--english_total", type=int, default=120000)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cidar_repeat", type=int, default=0,
                    help="(legacy) ar/smolkalam recipe: mix in CIDAR this many times.")
    ap.add_argument("--ar_recipe", choices=["smolkalam", "native"], default="smolkalam",
                    help="ar target half: 'smolkalam' (translated, 1st run) or 'native' "
                         "(Quora-Arabic-GPT4 + CIDAR, no upsampling; runbook §2).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()

    if args.lang == "hi":
        tgt_it, tgt_script = iter_indicalign_hi(args.sft_root), deva_ratio
        tgt, tgt_stats = build_side("target", tgt_it, args.target_total, args.lang, tgt_script, rng, seen)
    elif args.lang == "zh":
        tgt_it, tgt_script = iter_infinity_zh(args.sft_root), han_ratio
        tgt, tgt_stats = build_side("target", tgt_it, args.target_total, args.lang, tgt_script, rng, seen)
    elif args.lang == "ar" and args.ar_recipe == "native":
        # Native-Arabic recipe (runbook §2): Quora-Arabic-GPT4 (volume, native prompts)
        # + CIDAR (human-reviewed cultural overlay). 1x each, NO upsampling; SmolKalam dropped.
        print("== building CIDAR (native, all) ==", flush=True)
        cid, cid_stats = build_side("cidar", iter_cidar(args.sft_root), 20000, "ar", arab_ratio, rng, seen)
        print("== building Quora-Arabic-GPT4 (native, all) ==", flush=True)
        quo, quo_stats = build_side("quora", iter_quora_ar(args.sft_root), args.target_total, "ar", arab_ratio, rng, seen)
        tgt = quo + cid
        tgt_stats = {"quora": quo_stats, "cidar": cid_stats, "recipe": "native"}
    elif args.lang == "ar" and args.cidar_repeat > 0:
        print(f"== building CIDAR overlay (repeat={args.cidar_repeat}) ==", flush=True)
        def _cidar_repeated():
            for _ in range(args.cidar_repeat):
                yield from iter_cidar(args.sft_root)
        cid, cid_stats = build_side("cidar", _cidar_repeated(), 10000 * args.cidar_repeat,
                                    "ar", arab_ratio, rng, seen)
        sk_quota = max(0, args.target_total - len(cid))
        print(f"== building TARGET (ar/SmolKalam) quota={sk_quota} (+{len(cid)} CIDAR) ==", flush=True)
        sk, sk_stats = build_side("smolkalam", iter_smolkalam(args.sft_root), sk_quota,
                                  "ar", arab_ratio, rng, seen)
        tgt = sk + cid
        tgt_stats = {"smolkalam": sk_stats, "cidar": cid_stats, "cidar_repeat": args.cidar_repeat}
    else:
        tgt_it, tgt_script = iter_smolkalam(args.sft_root), arab_ratio
        tgt, tgt_stats = build_side("target", tgt_it, args.target_total, args.lang, tgt_script, rng, seen)

    print(f"== building ENGLISH quota={args.english_total} ==", flush=True)
    eng, eng_stats = build_side("english", iter_smoltalk2_english(args.sft_root),
                                args.english_total, "en", latin_ratio, rng, seen)

    mixture = tgt + eng
    rng.shuffle(mixture)                                  # <-- global shuffle, seed
    print(f"mixture size = {len(mixture)} (target {len(tgt)} + english {len(eng)})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    # out_dir must contain ONLY .jsonl shards (LLaMA-Factory dir loader requires identical types)
    for old in glob.glob(f"{args.out_dir}/*.jsonl"):
        os.remove(old)
    per = (len(mixture) + args.shards - 1) // args.shards
    for si in range(args.shards):
        chunk = mixture[si*per:(si+1)*per]
        if not chunk: continue
        with open(f"{args.out_dir}/part-{si:05d}.jsonl", "w", encoding="utf-8") as w:
            for r in chunk:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")

    realized = len(tgt) / max(1, len(mixture))
    manifest = dict(lang=args.lang, total=len(mixture), target_n=len(tgt), english_n=len(eng),
                    realized_target_ratio=round(realized, 4), seed=args.seed,
                    target_stats=tgt_stats, english_stats=eng_stats,
                    out_dir=args.out_dir, char_cap=CHAR_CAP)
    with open(args.out_dir.rstrip("/") + ".manifest.json", "w") as w:  # SIBLING, not inside out_dir
        json.dump(manifest, w, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if len(tgt) < args.target_total:
        print(f"WARN: target under quota ({len(tgt)}/{args.target_total})", flush=True)
    if len(eng) < args.english_total:
        print(f"WARN: english under quota ({len(eng)}/{args.english_total})", flush=True)

if __name__ == "__main__":
    main()
