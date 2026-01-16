import json
import argparse
from pathlib import Path


def lowercase_first_char(text: str) -> str:
    """
    Lowercase the first character of a string if it's a letter.
    
    Args:
        text: Input string
        
    Returns:
        String with first character lowercased if it's a letter
    """
    if not text:
        return text
    if text[0].isalpha():
        return text[0].lower() + text[1:]
    return text


def convert_idiom_kb_to_jsonl(input_file: str, output_file: str):
    """
    Convert en_idiom_meaning.json to JSONL format where each line contains an idiom entry.
    Skips duplicates by checking existing idioms in the output file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
    """
    # Track seen idioms to avoid duplicates
    seen_idioms = set()
    
    # Check if output file exists and read existing idioms
    output_path = Path(output_file)
    if output_path.exists():
        print(f"Reading existing idioms from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_entry = json.loads(line)
                        idiom = existing_entry.get('idiom', '')
                        if idiom:
                            seen_idioms.add(idiom)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
        print(f"Found {len(seen_idioms)} existing idioms")
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open output file for appending
    mode = 'a' if output_path.exists() else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        skipped_count = 0
        added_count = 0
        
        # Iterate through each entry in the array
        for entry in data:
            # Extract relevant fields
            idiom_raw = entry.get('idiom', '')
            en_meaning = entry.get('en_meaning', '')
            
            # Lowercase the first character of the idiom
            idiom = lowercase_first_char(idiom_raw)
            
            # Skip if idiom is empty or already seen
            if not idiom or idiom in seen_idioms:
                skipped_count += 1
                continue
            
            # Mark as seen
            seen_idioms.add(idiom)
            
            # Create the output structure
            output_entry = {
                "idiom": idiom,
                "definition": [
                    {
                        "unknown": {
                            "explanation": en_meaning,
                            "usage": None,
                            "example": []
                        }
                    }
                ],
                "patterns": []
            }
            
            # Write as a single line in JSONL format
            f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
            added_count += 1
        
        print(f"Added {added_count} new idioms, skipped {skipped_count} duplicates")


def main():
    parser = argparse.ArgumentParser(
        description='Convert en_idiom_meaning.json to JSONL format'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input JSON file (en_idiom_meaning.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    
    args = parser.parse_args()
    
    # Convert the file
    convert_idiom_kb_to_jsonl(args.input, args.output)
    print(f"Conversion complete. Output written/appended to {args.output}")


if __name__ == '__main__':
    main()

