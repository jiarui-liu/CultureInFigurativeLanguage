import json
import numpy as np
from collections import Counter
from typing import Dict, Any


def print_idiom_statistics(file_path: str):
    """
    Print basic statistics about idioms in a JSONL file.
    
    Statistics include:
    - Number of lines/idioms
    - Idiom length statistics (word count): mean, std, min, max
    - Idiom length statistics (character count): mean, std, min, max
    - Entity statistics: avg per idiom, std, total unique entities
    - Literal meaning statistics: avg per idiom, std
    - Figurative meaning statistics: avg per idiom, std
    
    Args:
        file_path: Path to the JSONL file containing idioms
    """
    idioms = []
    idiom_lengths_words = []
    idiom_lengths_chars = []
    entities_per_idiom = []
    literal_meanings_per_idiom = []
    figurative_meanings_per_idiom = []
    all_entities = []
    
    line_count = 0
    valid_idiom_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                idiom_text = obj.get("idiom", "")
                
                if not idiom_text:
                    continue
                
                valid_idiom_count += 1
                idioms.append(idiom_text)
                
                # Word count (split by whitespace)
                words = idiom_text.split()
                word_count = len(words)
                idiom_lengths_words.append(word_count)
                
                # Character count
                char_count = len(idiom_text)
                idiom_lengths_chars.append(char_count)
                
                # Entity statistics
                output = obj.get("output", {})
                if output and isinstance(output, dict):
                    entities = output.get("entities", [])
                    entities_per_idiom.append(len(entities))
                    all_entities.extend(entities)
                    
                    literal_meanings = output.get("literal_meanings", [])
                    # Handle case where literal_meanings might be a list of lists
                    if literal_meanings and isinstance(literal_meanings[0], list):
                        literal_meanings = [item for sublist in literal_meanings for item in sublist]
                    literal_meanings_per_idiom.append(len(literal_meanings))
                    
                    figurative_meanings = output.get("figurative_meanings", [])
                    # Handle case where figurative_meanings might be a list of lists
                    if figurative_meanings and isinstance(figurative_meanings[0], list):
                        figurative_meanings = [item for sublist in figurative_meanings for item in sublist]
                    figurative_meanings_per_idiom.append(len(figurative_meanings))
                else:
                    entities_per_idiom.append(0)
                    literal_meanings_per_idiom.append(0)
                    figurative_meanings_per_idiom.append(0)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_count}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_count}: {e}")
                continue
    
    # Convert to numpy arrays for statistics
    idiom_lengths_words = np.array(idiom_lengths_words)
    idiom_lengths_chars = np.array(idiom_lengths_chars)
    entities_per_idiom = np.array(entities_per_idiom)
    literal_meanings_per_idiom = np.array(literal_meanings_per_idiom)
    figurative_meanings_per_idiom = np.array(figurative_meanings_per_idiom)
    
    # Print statistics
    print("=" * 80)
    print(f"Statistics for: {file_path}")
    print("=" * 80)
    print(f"\nBasic Counts:")
    print(f"  Total lines in file: {line_count:,}")
    print(f"  Valid idioms: {valid_idiom_count:,}")
    print(f"  Unique idioms: {len(set(idioms)):,}")
    
    print(f"\nIdiom Length Statistics (Word Count):")
    print(f"  Mean: {np.mean(idiom_lengths_words):.2f} words")
    print(f"  Std:  {np.std(idiom_lengths_words):.2f} words")
    print(f"  Min:  {np.min(idiom_lengths_words)} words")
    print(f"  Max:  {np.max(idiom_lengths_words)} words")
    print(f"  Median: {np.median(idiom_lengths_words):.2f} words")
    
    print(f"\nIdiom Length Statistics (Character Count):")
    print(f"  Mean: {np.mean(idiom_lengths_chars):.2f} characters")
    print(f"  Std:  {np.std(idiom_lengths_chars):.2f} characters")
    print(f"  Min:  {np.min(idiom_lengths_chars)} characters")
    print(f"  Max:  {np.max(idiom_lengths_chars)} characters")
    print(f"  Median: {np.median(idiom_lengths_chars):.2f} characters")
    
    print(f"\nEntity Statistics:")
    print(f"  Average entities per idiom: {np.mean(entities_per_idiom):.2f}")
    print(f"  Std entities per idiom: {np.std(entities_per_idiom):.2f}")
    print(f"  Min entities per idiom: {np.min(entities_per_idiom)}")
    print(f"  Max entities per idiom: {np.max(entities_per_idiom)}")
    print(f"  Total unique entities: {len(set(all_entities)):,}")
    print(f"  Total entity mentions: {len(all_entities):,}")
    
    print(f"\nLiteral Meaning Statistics:")
    print(f"  Average literal meanings per idiom: {np.mean(literal_meanings_per_idiom):.2f}")
    print(f"  Std literal meanings per idiom: {np.std(literal_meanings_per_idiom):.2f}")
    print(f"  Min literal meanings per idiom: {np.min(literal_meanings_per_idiom)}")
    print(f"  Max literal meanings per idiom: {np.max(literal_meanings_per_idiom)}")
    print(f"  Idioms with literal meanings: {np.sum(literal_meanings_per_idiom > 0):,}")
    
    print(f"\nFigurative Meaning Statistics:")
    print(f"  Average figurative meanings per idiom: {np.mean(figurative_meanings_per_idiom):.2f}")
    print(f"  Std figurative meanings per idiom: {np.std(figurative_meanings_per_idiom):.2f}")
    print(f"  Min figurative meanings per idiom: {np.min(figurative_meanings_per_idiom)}")
    print(f"  Max figurative meanings per idiom: {np.max(figurative_meanings_per_idiom)}")
    print(f"  Idioms with figurative meanings: {np.sum(figurative_meanings_per_idiom > 0):,}")
    
    print("=" * 80)


def get_idiom_statistics(file_path: str) -> Dict[str, Any]:
    """
    Get basic statistics about idioms in a JSONL file as a dictionary.
    
    Args:
        file_path: Path to the JSONL file containing idioms
        
    Returns:
        Dictionary containing all statistics
    """
    idioms = []
    idiom_lengths_words = []
    idiom_lengths_chars = []
    entities_per_idiom = []
    literal_meanings_per_idiom = []
    figurative_meanings_per_idiom = []
    all_entities = []
    
    line_count = 0
    valid_idiom_count = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                idiom_text = obj.get("idiom", "")
                
                if not idiom_text:
                    continue
                
                valid_idiom_count += 1
                idioms.append(idiom_text)
                
                # Word count (split by whitespace)
                words = idiom_text.split()
                word_count = len(words)
                idiom_lengths_words.append(word_count)
                
                # Character count
                char_count = len(idiom_text)
                idiom_lengths_chars.append(char_count)
                
                # Entity statistics
                output = obj.get("output", {})
                if output and isinstance(output, dict):
                    entities = output.get("entities", [])
                    entities_per_idiom.append(len(entities))
                    all_entities.extend(entities)
                    
                    literal_meanings = output.get("literal_meanings", [])
                    # Handle case where literal_meanings might be a list of lists
                    if literal_meanings and isinstance(literal_meanings[0], list):
                        literal_meanings = [item for sublist in literal_meanings for item in sublist]
                    literal_meanings_per_idiom.append(len(literal_meanings))
                    
                    figurative_meanings = output.get("figurative_meanings", [])
                    # Handle case where figurative_meanings might be a list of lists
                    if figurative_meanings and isinstance(figurative_meanings[0], list):
                        figurative_meanings = [item for sublist in figurative_meanings for item in sublist]
                    figurative_meanings_per_idiom.append(len(figurative_meanings))
                else:
                    entities_per_idiom.append(0)
                    literal_meanings_per_idiom.append(0)
                    figurative_meanings_per_idiom.append(0)
                    
            except json.JSONDecodeError as e:
                continue
            except Exception as e:
                continue
    
    # Convert to numpy arrays for statistics
    idiom_lengths_words = np.array(idiom_lengths_words)
    idiom_lengths_chars = np.array(idiom_lengths_chars)
    entities_per_idiom = np.array(entities_per_idiom)
    literal_meanings_per_idiom = np.array(literal_meanings_per_idiom)
    figurative_meanings_per_idiom = np.array(figurative_meanings_per_idiom)
    
    stats = {
        "file_path": file_path,
        "line_count": line_count,
        "valid_idiom_count": valid_idiom_count,
        "unique_idiom_count": len(set(idioms)),
        "word_length": {
            "mean": float(np.mean(idiom_lengths_words)),
            "std": float(np.std(idiom_lengths_words)),
            "min": int(np.min(idiom_lengths_words)),
            "max": int(np.max(idiom_lengths_words)),
            "median": float(np.median(idiom_lengths_words))
        },
        "char_length": {
            "mean": float(np.mean(idiom_lengths_chars)),
            "std": float(np.std(idiom_lengths_chars)),
            "min": int(np.min(idiom_lengths_chars)),
            "max": int(np.max(idiom_lengths_chars)),
            "median": float(np.median(idiom_lengths_chars))
        },
        "entities": {
            "avg_per_idiom": float(np.mean(entities_per_idiom)),
            "std_per_idiom": float(np.std(entities_per_idiom)),
            "min_per_idiom": int(np.min(entities_per_idiom)),
            "max_per_idiom": int(np.max(entities_per_idiom)),
            "total_unique": len(set(all_entities)),
            "total_mentions": len(all_entities)
        },
        "literal_meanings": {
            "avg_per_idiom": float(np.mean(literal_meanings_per_idiom)),
            "std_per_idiom": float(np.std(literal_meanings_per_idiom)),
            "min_per_idiom": int(np.min(literal_meanings_per_idiom)),
            "max_per_idiom": int(np.max(literal_meanings_per_idiom)),
            "idioms_with_meanings": int(np.sum(literal_meanings_per_idiom > 0))
        },
        "figurative_meanings": {
            "avg_per_idiom": float(np.mean(figurative_meanings_per_idiom)),
            "std_per_idiom": float(np.std(figurative_meanings_per_idiom)),
            "min_per_idiom": int(np.min(figurative_meanings_per_idiom)),
            "max_per_idiom": int(np.max(figurative_meanings_per_idiom)),
            "idioms_with_meanings": int(np.sum(figurative_meanings_per_idiom > 0))
        }
    }
    
    return stats

