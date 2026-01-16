import json
import argparse
import re
from pathlib import Path


def parse_definition(definition_text: str):
    """
    Parse a definition string into structured meanings.
    Handles numbered senses, Lit./Fig. usage, and examples marked by '_'.
    """
    meanings = []

    # Match numbered senses like "1. ... 2. ..."
    matches = re.findall(
        r'\d+\.\s*(.*?)(?=\s*\d+\.|$)',
        definition_text,
        flags=re.DOTALL
    )

    if not matches:
        matches = [definition_text]

    for sense in matches:
        sense = sense.strip()
        
        # Skip empty senses
        if not sense:
            continue

        parts = [p.strip() for p in sense.split('_') if p.strip()]
        
        # Handle edge case where sense might only contain underscores
        if not parts:
            continue
            
        explanation = parts[0]
        examples = parts[1:]

        usage = None
        if "Lit." in explanation:
            usage = "literal"
            explanation = explanation.replace("Lit.", "").strip()
        elif "Fig." in explanation:
            usage = "figurative"
            explanation = explanation.replace("Fig.", "").strip()

        meanings.append({
            "unknown": {
                "explanation": explanation,
                "usage": usage,
                "example": examples
            }
        })

    return meanings


def convert_phrases_to_jsonl(input_file: str, output_file: str):
    seen_idioms = set()

    # FORCE overwrite to avoid contamination from old runs
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as out:
        added = 0
        skipped = 0

        for entry in data.get("dictionary", []):

            # Skip explicitly marked duplicates
            if entry.get("duplicate") is True:
                skipped += 1
                continue

            idiom = entry.get("phrase", "").strip()
            definition_text = entry.get("definition", "").strip()
            patterns = entry.get("patterns", [])

            if not idiom or idiom in seen_idioms:
                skipped += 1
                continue

            parsed_definitions = parse_definition(definition_text)

            output_entry = {
                "idiom": idiom,
                "definition": parsed_definitions,
                "patterns": patterns
            }

            out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            seen_idioms.add(idiom)
            added += 1

        print(f"Added {added} idioms, skipped {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_phrases_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
