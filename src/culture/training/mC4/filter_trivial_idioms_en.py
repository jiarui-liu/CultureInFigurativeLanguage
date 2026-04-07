#!/usr/bin/env python3
"""
Filter trivial idioms from English idiom dataset using GPT-5.2-chat-latest.

A trivial idiom is defined as:
1. A single common word that is not truly an idiom (e.g., "would", "the", "is")
2. A basic grammatical construction without figurative meaning
3. A simple phrase that can be understood literally without any cultural or figurative interpretation

Non-trivial idioms are expressions that:
1. Have figurative or metaphorical meanings beyond their literal interpretation
2. Carry cultural significance or context
3. Are recognized phrases or sayings with established meanings
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from culture.models.llm_utils import ChatModel


SYSTEM_PROMPT = """You are an expert linguist tasked with identifying trivial phrases vs. idioms.

A TRIVIAL phrase is:
1. A single common word that is NOT actually an idiom (e.g., "would", "the", "is", "can", "may", "shall")
2. A basic grammatical word or modal verb being misclassified as an idiom
3. A word or phrase that is just a regular part of speech, not an idiomatic expression


Examples of TRIVIAL entries (answer YES):
- "would" (just a modal verb)
- "the" (just an article)
- "is" (just a verb)
- "very" (just an adverb)

Examples of idioms (answer NO):
- "kick the bucket" (means to die)
- "break a leg" (means good luck)
- "raining cats and dogs" (means raining heavily)
- "spill the beans" (means to reveal a secret)
- "piece of cake" (means something easy)

You must respond with ONLY "YES" or "NO":
- YES = This is a TRIVIAL entry (should be filtered out)
- NO = This is an idiom (should be kept)"""


USER_PROMPT_TEMPLATE = """Is the following entry a TRIVIAL phrase that should be filtered out?

Phrase: "{idiom}"
Literal meanings: {literal_meanings}
Figurative meanings: {figurative_meanings}

Remember:
- Answer YES if this is just a common word, modal verb, or basic grammatical construction (not a real idiom)
- Answer NO if this is a genuine idiom

Your answer (YES or NO only):"""


def is_trivial_idiom(model: ChatModel, idiom_data: dict) -> bool:
    """
    Use GPT to determine if an idiom is trivial.

    Args:
        model: ChatModel instance
        idiom_data: Dictionary containing idiom information

    Returns:
        True if the idiom is trivial, False otherwise
    """
    output = idiom_data.get("output", idiom_data)
    idiom = output.get("idiom", "")
    print(idiom)
    literal_meanings = output.get("literal_meanings", [])
    figurative_meanings = output.get("figurative_meanings", [])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        idiom=idiom,
        literal_meanings=json.dumps(literal_meanings, ensure_ascii=False),
        figurative_meanings=json.dumps(figurative_meanings, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = model.generate(messages)
        response = response.strip().upper()
        print(response)

        # Parse the response
        if "YES" in response:
            return True
        elif "NO" in response:
            return False
        else:
            # Default to keeping the idiom if response is unclear
            print(f"Warning: Unclear response '{response}' for idiom '{idiom}', keeping it.")
            return False
    except Exception as e:
        print(f"Error processing idiom '{idiom}': {e}")
        # Default to keeping the idiom on error
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter trivial idioms from English idiom dataset using GPT-5.2-chat-latest"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted_nontrivial.jsonl",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-chat-latest",
        help="OpenAI model to use for classification"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for async processing"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for processing (useful for resuming)"
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Ending index for processing (-1 for all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to file"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls; only filter entries with empty figurative meanings"
    )

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Initialize the model (skip if --skip-llm is set)
    model = None
    if not args.skip_llm:
        print(f"Initializing model: {args.model}")
        model = ChatModel(model=args.model, provider="openai")

    # Read all lines from input
    print(f"Reading input file: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total idioms in input: {total_lines}")

    # Determine processing range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index != -1 else total_lines
    lines_to_process = lines[start_idx:end_idx]

    print(f"Processing idioms from index {start_idx} to {end_idx}")

    # Process each idiom
    non_trivial_idioms = []
    trivial_count = 0

    for i, line in enumerate(tqdm(lines_to_process, desc="Filtering idioms")):
        line = line.strip()
        if not line:
            continue

        try:
            idiom_data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON at line {start_idx + i}: {e}")
            continue

        output = idiom_data.get("output", idiom_data)
        if output is None or "idiom" not in output:
            print(f"Skipping '{idiom_data}': output is None or 'idiom' not in output")
            continue
        idiom_name = output["idiom"]
        figurative_meanings = output.get("figurative_meanings", [])

        # Skip if figurative meanings is empty
        if not figurative_meanings:
            # print(f"Skipping '{idiom_name}': empty figurative meanings")
            continue

        # Skip LLM call if --skip-llm is set; keep all entries with non-empty figurative meanings
        if args.skip_llm:
            non_trivial_idioms.append(idiom_data)
            if args.dry_run:
                print(f"KEPT (no LLM): {idiom_name}")
        elif is_trivial_idiom(model, idiom_data):
            trivial_count += 1
            if args.dry_run:
                print(f"TRIVIAL: {idiom_name}")
        else:
            non_trivial_idioms.append(idiom_data)
            if args.dry_run:
                print(f"NON-TRIVIAL: {idiom_name}")

    print(f"\nResults:")
    print(f"  Total processed: {len(lines_to_process)}")
    print(f"  Trivial (filtered out): {trivial_count}")
    print(f"  Non-trivial (kept): {len(non_trivial_idioms)}")

    # Write output
    if not args.dry_run:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Writing non-trivial idioms to: {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            for idiom_data in non_trivial_idioms:
                f.write(json.dumps(idiom_data, ensure_ascii=False) + "\n")

        print(f"Done! Wrote {len(non_trivial_idioms)} non-trivial idioms to {args.output}")
    else:
        print("Dry run complete. No files written.")


if __name__ == "__main__":
    main()
