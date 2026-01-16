import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional


def normalize_idiom_for_comparison(idiom: str) -> str:
    """
    Normalize idiom for case-insensitive comparison.
    Lowercases the first character if it's a letter, otherwise returns as-is.
    
    Args:
        idiom: Input idiom string
        
    Returns:
        Normalized idiom for comparison
    """
    if not idiom:
        return idiom
    if idiom[0].isalpha():
        return idiom[0].lower() + idiom[1:]
    return idiom


def idioms_differ_only_by_case(idiom1: str, idiom2: str) -> bool:
    """
    Check if two idioms differ only by the first character's case.
    
    Args:
        idiom1: First idiom
        idiom2: Second idiom
        
    Returns:
        True if idioms differ only by first character case
    """
    if not idiom1 or not idiom2:
        return False
    
    # Normalize both for comparison
    norm1 = normalize_idiom_for_comparison(idiom1)
    norm2 = normalize_idiom_for_comparison(idiom2)
    
    # They must be identical after normalization
    if norm1 != norm2:
        return False
    
    # And the original strings must differ only in first character case
    if len(idiom1) != len(idiom2):
        return False
    
    # Check if only first character differs (and rest is identical)
    if idiom1[1:] != idiom2[1:]:
        return False
    
    # Check if first characters differ only by case
    if idiom1[0].isalpha() and idiom2[0].isalpha():
        return idiom1[0].lower() == idiom2[0].lower() and idiom1[0] != idiom2[0]
    
    return False


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read all entries from a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries parsed from JSONL
    """
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num} in {file_path}: {e}")
    return entries


def merge_idioms(input_files: List[str], output_file: str, wiktionary_file: Optional[str] = None):
    """
    Merge idioms from multiple JSONL files into a single output file.
    Each unique idiom will have entries from all sources.
    Idioms that differ only by first character case will be merged.
    
    Args:
        input_files: List of paths to input JSONL files
        output_file: Path to output JSONL file
        wiktionary_file: Optional path to wiktionary file (for case preference)
    """
    # Dictionary to store merged idioms: normalized_idiom -> (canonical_idiom, list of source entries)
    # canonical_idiom is the preferred case variant
    idiom_dict: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
    
    # Track which file is the wiktionary file
    wiktionary_file_name = None
    if wiktionary_file:
        wiktionary_file_name = Path(wiktionary_file).name
    
    # Read all input files
    for file_path in input_files:
        file_name = Path(file_path).name
        is_wiktionary = (wiktionary_file and file_name == wiktionary_file_name)
        print(f"Reading {file_path}...")
        
        entries = read_jsonl_file(file_path)
        
        for entry in entries:
            idiom = entry.get('idiom', '')
            if not idiom:
                continue
            
            # Normalize idiom for comparison
            normalized = normalize_idiom_for_comparison(idiom)
            
            # Create source entry with original data plus source filename
            source_entry = {
                "definition": entry.get('definition', []),
                "patterns": entry.get('patterns', []),
                "source": file_name
            }
            
            # Check if we already have this idiom (case-insensitive)
            if normalized in idiom_dict:
                canonical_idiom, source_entries = idiom_dict[normalized]
                
                # If this is from wiktionary file, prefer its case
                if is_wiktionary:
                    # Update canonical idiom to wiktionary's case
                    idiom_dict[normalized] = (idiom, source_entries)
                    source_entries.append(source_entry)
                else:
                    # Keep existing canonical, just add source entry
                    source_entries.append(source_entry)
            else:
                # New idiom, add it
                idiom_dict[normalized] = (idiom, [source_entry])
        
        print(f"  Found {len(entries)} entries in {file_name}")
    
    # Create merged output
    print(f"\nMerging {len(idiom_dict)} unique idioms...")
    
    merged_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, (normalized, (canonical_idiom, source_entries)) in enumerate(sorted(idiom_dict.items())):
            # Create output entry
            output_entry = {
                "idiom": canonical_idiom,
                "index": index,
            }
            
            # Add each source as source1, source2, etc.
            for idx, source_entry in enumerate(source_entries, 1):
                source_key = f"source{idx}"
                output_entry[source_key] = source_entry
            
            # Write as a single line in JSONL format
            f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
            merged_count += 1
    
    # Print statistics
    total_sources = sum(len(entries) for _, (_, entries) in idiom_dict.items())
    print(f"Total unique idioms: {len(idiom_dict)}")
    print(f"Total source entries: {total_sources}")
    print(f"Average sources per idiom: {total_sources / len(idiom_dict):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge idioms from multiple JSONL files into a single output file'
    )
    parser.add_argument(
        '--inputs',
        type=str,
        nargs='+',
        required=True,
        help='List of input JSONL files to merge'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for input_file in args.inputs:
        if not Path(input_file).exists():
            parser.error(f"Input file does not exist: {input_file}")
    
    # Detect wiktionary file (check if any input file matches the wiktionary pattern)
    wiktionary_file = None
    for input_file in args.inputs:
        input_path = Path(input_file)
        # Check if it's the wiktionary file based on path or filename
        if 'wiktionary_outputs' in str(input_path) and 'english_idioms_reformatted.jsonl' in input_path.name:
            wiktionary_file = input_file
            print(f"Detected wiktionary file: {wiktionary_file}")
            break
    
    # Merge the files
    merge_idioms(args.inputs, args.output, wiktionary_file)
    print(f"\nMerge complete. Output written to {args.output}")


if __name__ == '__main__':
    main()

