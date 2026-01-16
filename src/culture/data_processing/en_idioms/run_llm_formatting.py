import argparse
import json
import os
import asyncio
from typing import List, Dict, Set, Optional, Tuple, Any

# Import ChatModel from llm_utils
import sys
from culture.models.llm_utils import ChatModel

prompt = """
You are given 10 JSON idiom entries. Each entry may contain one or more sources and multiple definitions.

Your task is pure extraction, clustering, and classification only. You must NOT generate, infer, paraphrase, or modify meanings.

For each idiom entry:
1. Meaning clustering and classification from the "definition" field
- Cluster definitions when they clearly express the same meaning by choosing one of the definitions.
- If a definition has a "usage" value of "literal" or "figurative", use it directly to classify the definition.
- If "usage" is null or missing, classify manually based only on the definition text.
- You should use the original definition string verbatim, without any modifications, except for formatting errors.
2. Entity extraction from the idiom string itself
- Should be concrete nouns or noun phrases in their singular form.


Output format (JSON only):
```
[
{{
"idiom": "",
"entities": [],
"literal_meanings": [],
"figurative_meanings": []
}},
...
]
```

Here are the 10 idiom entries:
```
{input_idioms}
```

Now produce the output strictly following the format above. Do not include any additional text.
"""


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_meanings(entry: Dict) -> Dict:
    """
    Converts one raw idiom entry into:
    {
        "idiom": "...",
        "meanings": [
            {"usage": "literal"/"figurative"/None, "definition": "..."},
            ...
        ]
    }

    This function is robust to arbitrary definition keys such as
    "unknown", "verb", etc.
    """
    idiom = entry.get("idiom", "")
    meanings = []

    for key, source in entry.items():
        if not key.startswith("source"):
            continue

        definitions = source.get("definition", [])
        for definition_obj in definitions:
            # Each definition_obj has exactly one key like "unknown", "verb", etc.
            for _, content in definition_obj.items():
                explanation = content.get("explanation")
                if explanation is None:
                    continue

                meanings.append({
                    "usage": content.get("usage"),
                    "definition": explanation
                })

    return {
        "idiom": idiom,
        "meanings": meanings
    }


def chunk_data(data: List, chunk_size: int = 10) -> List[List]:
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_processed_indices(output_path: str) -> Set[int]:
    """Returns a set of indices that are already in the output file."""
    processed = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        idx = entry.get("index")
                        if idx is not None:
                            processed.add(idx)
                    except json.JSONDecodeError:
                        continue
    return processed

def parse_llm_output(output: str, chunk: List[tuple]) -> Optional[List[Dict]]:
    """
    Parse LLM output and match results to the chunk.
    Returns a list of parsed results for each idiom, or None if parsing fails.
    """
    try:
        # Try to parse the JSON output
        parsed_output = json.loads(output)

        if not isinstance(parsed_output, list):
            raise ValueError(f"Expected a list, got {type(parsed_output)}")

        # Create a mapping from idiom to its result
        idiom_to_result = {}
        for item in parsed_output:
            idiom_str = item.get("idiom", "")
            if idiom_str:
                idiom_to_result[idiom_str] = item

        # Match each idiom in the chunk to its result
        results = []
        for idx, entry in chunk:
            idiom = entry.get("idiom", "")
            if idiom in idiom_to_result:
                results.append({
                    "idiom": idiom,
                    "index": idx,
                    "output": idiom_to_result[idiom]
                })
            else:
                # If idiom not found in output, store None or empty dict
                print(f"Warning: Idiom '{idiom}' not found in LLM output")
                results.append({
                    "idiom": idiom,
                    "index": idx,
                    "output": None
                })

        return results

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Failed to parse LLM output: {e}")
        return None


async def process_chunks_batch(
    model: ChatModel,
    chunks: List[List[tuple]],
    output_path: str,
    batch_size: int = 10,
    max_retries: int = 3
) -> Tuple[int, int]:
    """
    Process multiple chunks in parallel using batch API calls.
    Writes results to output file immediately after each batch completes.

    Args:
        model: ChatModel instance
        chunks: List of chunks, where each chunk is a list of (idx, entry) tuples
        output_path: Path to output file for writing results
        batch_size: Number of chunks to process in parallel
        max_retries: Maximum number of retries for failed chunks

    Returns:
        Tuple of (processed_count, skipped_count)
    """
    processed_count = 0
    skipped_count = 0

    # Process chunks in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} "
              f"(chunks {batch_start + 1}-{batch_end} of {len(chunks)})")

        # Prepare messages for each chunk in the batch
        batch_messages = []
        for chunk_idx, chunk in enumerate(batch_chunks, start=batch_start):
            chunk_data_for_prompt = [entry for _, entry in chunk]
            formatted_prompt = prompt.format(input_idioms=json.dumps(chunk_data_for_prompt, ensure_ascii=False, indent=2))

            messages = [
                {"role": "system", "content": "You are a strict information extraction assistant."},
                {"role": "user", "content": formatted_prompt}
            ]
            batch_messages.append((chunk_idx, messages))

        # Try with retries
        remaining_messages = batch_messages.copy()
        chunk_results = {}  # chunk_idx -> results

        for attempt in range(max_retries):
            if not remaining_messages:
                break

            if attempt > 0:
                print(f"  Retry attempt {attempt + 1} for {len(remaining_messages)} failed chunks...")

            # Call batch API
            responses = await model.batch_generate_with_indices(
                remaining_messages
            )

            # Process responses and identify failures
            failed_messages = []
            for chunk_idx, response, error in responses:
                if error is not None:
                    print(f"  Chunk {chunk_idx + 1} failed with error: {error}")
                    # Find the original message for retry
                    for msg_tuple in remaining_messages:
                        if msg_tuple[0] == chunk_idx:
                            failed_messages.append(msg_tuple)
                            break
                    continue

                # Parse the response
                chunk = chunks[chunk_idx]
                parsed_results = parse_llm_output(response, chunk)

                if parsed_results is None:
                    print(f"  Chunk {chunk_idx + 1} failed to parse, will retry...")
                    # Find the original message for retry
                    for msg_tuple in remaining_messages:
                        if msg_tuple[0] == chunk_idx:
                            failed_messages.append(msg_tuple)
                            break
                else:
                    chunk_results[chunk_idx] = parsed_results
                    print(f"  Chunk {chunk_idx + 1} processed successfully")

            remaining_messages = failed_messages

        # Mark any remaining failures as None
        for chunk_idx, _ in remaining_messages:
            print(f"  Chunk {chunk_idx + 1} failed after all retries, skipping...")
            chunk_results[chunk_idx] = None

        # Write results to file immediately after batch completes
        with open(output_path, "a", encoding="utf-8") as f:
            for chunk_idx in range(batch_start, batch_end):
                results = chunk_results.get(chunk_idx)
                if results is None:
                    skipped_count += len(chunks[chunk_idx])
                    print(f"  Skipped chunk {chunk_idx + 1} with {len(chunks[chunk_idx])} idioms due to failures")
                else:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        processed_count += 1
            f.flush()

        print(f"  Batch complete. Total processed: {processed_count}, skipped: {skipped_count}")

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o-mini)")
    parser.add_argument("--provider", default="openai", choices=["openai", "togetherai", "bedrock"],
                        help="Model provider (default: openai)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for failed JSON parsing (default: 3)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of chunks to process in parallel (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Number of idioms per chunk (default: 10)")
    args = parser.parse_args()

    # Get already processed indices
    processed_indices = get_processed_indices(args.output)
    print(f"Found {len(processed_indices)} already processed items (by index)")

    # Read input and filter out processed items by index
    raw_entries = read_jsonl(args.input)

    # Create list of (index, entry) tuples, filtering out processed indices
    filtered_entries = []
    for idx, entry in enumerate(raw_entries):
        if idx not in processed_indices:
            filtered_entries.append((idx, entry))

    print(f"Processing {len(filtered_entries)} idioms (skipped {len(raw_entries) - len(filtered_entries)} already processed)")

    if not filtered_entries:
        print("No idioms to process. Exiting.")
        return

    # Convert to format needed for LLM
    converted_entries = [(idx, extract_meanings(e)) for idx, e in filtered_entries]
    chunks = chunk_data(converted_entries, chunk_size=args.chunk_size)

    print(f"Created {len(chunks)} chunks of up to {args.chunk_size} idioms each")
    print(f"Will process in batches of {args.batch_size} chunks in parallel")

    # Initialize ChatModel
    model = ChatModel(model=args.model, provider=args.provider)

    # Process all chunks using batch API (writes to file after each batch)
    async def run_batch_processing():
        return await process_chunks_batch(
            model=model,
            chunks=chunks,
            output_path=args.output,
            batch_size=args.batch_size,
            max_retries=args.max_retries
        )

    # Run async batch processing
    print("\nStarting batch processing...")
    processed_count, skipped_count = asyncio.run(run_batch_processing())

    print(f"\nFinished. Processed {processed_count} idiom entries, skipped {skipped_count} entries")


if __name__ == "__main__":
    main()
