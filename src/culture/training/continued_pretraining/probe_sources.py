from datasets import load_dataset
def probe(name, kwargs):
    try:
        ds = load_dataset(streaming=True, **kwargs)
        r = next(iter(ds))
        tlen = len(str(r.get("text", "")))
        print("OK  %s: keys=%s textlen=%d" % (name, list(r.keys())[:8], tlen))
    except Exception as e:
        print("ERR %s: %s: %s" % (name, type(e).__name__, str(e)[:150]))
probe("fineweb-2 hin_Deva", dict(path="HuggingFaceFW/fineweb-2", name="hin_Deva", split="train"))
probe("allenai/c4 hi", dict(path="allenai/c4", name="hi", split="train"))
probe("legacy-datasets/mc4 hi", dict(path="legacy-datasets/mc4", name="hi", split="train"))
probe("allenai/mc4 hi", dict(path="allenai/mc4", name="hi", split="train"))
probe("ai4bharat/IndicCorpV2", dict(path="ai4bharat/IndicCorpV2", split="train"))
