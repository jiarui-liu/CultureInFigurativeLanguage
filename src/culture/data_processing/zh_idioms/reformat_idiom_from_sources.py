import argparse
import json
import os
from collections import defaultdict


def normalize_example(example, idiom):
    if example is None or example == "无":
        return None

    def _replace(s):
        return s.replace("～", idiom).replace("~", idiom)

    if isinstance(example, list):
        return [_replace(e) for e in example]
    else:
        return [_replace(example)]


def load_xinhua(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    result = defaultdict(list)

    for item in data:
        idiom = item["word"]
        definition = (
            f'{item.get("derivation", "")}{item.get("explanation", "")}'.strip()
            or None
        )
        example = normalize_example(item.get("example"), idiom)

        result[idiom].append({
            "definition": definition,
            "example": example,
            "source_input": "xinhua"
        })

    return result


def load_chengyu(folder):
    def_path = os.path.join(folder, "chengyu_definition.txt")
    sent_path = os.path.join(folder, "chengyu_sentence.txt")

    definitions = {}
    examples = defaultdict(list)

    with open(def_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            idiom, definition = line.split(",", 1)
            definitions[idiom] = definition.strip()

    with open(sent_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            idiom, sentence = line.split(",", 1)
            sentence = sentence.replace("～", idiom).replace("~", idiom)
            examples[idiom].append(sentence)

    result = defaultdict(list)
    all_idioms = set(definitions) | set(examples)

    for idiom in all_idioms:
        result[idiom].append({
            "definition": definitions.get(idiom),
            "example": examples.get(idiom) or None,
            "source_input": "chengyu"
        })

    return result


def load_fuxi(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    result = defaultdict(list)

    for item in data:
        idiom = item["input"]
        result[idiom].append({
            "definition": item.get("output"),
            "example": None,
            "source_input": "fuxi"
        })

    return result


def load_idiomkb(path):
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    result = defaultdict(list)

    for item in data:
        idiom = item["idiom"]
        result[idiom].append({
            "definition": item.get("zh_meaning"),
            "example": None,
            "source_input": "idiomkb"
        })

    return result


def main(args):
    xinhua = load_xinhua(args.input_xinhua)
    chengyu = load_chengyu(args.input_chengyu)
    fuxi = load_fuxi(args.input_fuxi)
    idiomkb = load_idiomkb(args.input_idiomkb)

    all_idioms = sorted(set(xinhua) | set(chengyu) | set(fuxi) | set(idiomkb))

    with open(args.output, "w", encoding="utf8") as fout:
        for idx, idiom in enumerate(all_idioms):
            record = {
                "idiom": idiom,
                "index": idx,
                "source1": xinhua.get(idiom),
                "source2": chengyu.get(idiom),
                "source3": fuxi.get(idiom),
                "source4": idiomkb.get(idiom),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xinhua", type=str, required=True)
    parser.add_argument("--input_chengyu", type=str, required=True)
    parser.add_argument("--input_fuxi", type=str, required=True)
    parser.add_argument("--input_idiomkb", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    main(args)
