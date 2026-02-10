#!/usr/bin/env python3
"""
Iteratively download and filter mC4/C4 dataset for documents containing idioms.

Uses chunked streaming processing:
- Downloads documents one at a time (streaming)
- Filters each document immediately
- Only saves matching documents (no raw data stored)
- Outputs chunked, gzip-compressed files (like allenai/c4 format)
- Memory efficient for large-scale pretraining corpora

Output format:
- mc4_filtered/zh/mc4_zh_filtered_00000.json.gz
- mc4_filtered/zh/mc4_zh_filtered_00001.json.gz
- ... (each file contains --chunk-size documents)

Supports multiple filtering modes:
1. Local regex/string matching (default)
2. Remote infini-gram API (English only, pre-built indexes)
3. Local infini-gram engine (any language, builds index from filtered docs)

Usage:
    # Basic filtering (outputs chunked .json.gz files)
    python download_and_filter_mc4.py --lang zh --chunk-size 10000

    # Build local infini-gram index from filtered docs
    python download_and_filter_mc4.py --lang zh --build-index --index-dir ./index_zh

    # Use existing local index for idiom counting
    python download_and_filter_mc4.py --lang zh --use-infinigram-local --index-dir ./index_zh
"""

import argparse
import json
import re
import gc
import gzip
import time
import subprocess
from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from datasets import load_dataset
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Infini-gram API endpoint (for remote queries)
INFINIGRAM_API_URL = "https://api.infini-gram.io/"

# Available remote infini-gram indexes (English only)
INFINIGRAM_REMOTE_INDEXES = [
    "v4_pileval_llama",      # Pile validation (small, for testing)
    "v4_dolma-v1_7_llama",   # Dolma v1.7 (2.6T tokens)
    "v4_rpj_llama_s4",       # RedPajama (1.4T tokens)
    "v4_c4train_llama",      # C4 train
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and filter mC4/C4 dataset for idiom-containing documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Language selection
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en", "zh"],
        choices=["en", "zh", "ar", "hi"],
        help="Languages to process",
    )

    # Path arguments
    parser.add_argument(
        "--base-dir",
        type=Path,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data",
        help="Base directory for data",
    )
    parser.add_argument(
        "--idiom-dir",
        type=Path,
        default=None,
        help="Directory containing idiom files (default: <base-dir>/idioms)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for filtered documents (default: <base-dir>/mc4_filtered)",
    )
    parser.add_argument(
        "--idiom-file",
        type=Path,
        default=None,
        help="Specific idiom file to use (overrides idiom-dir for single language)",
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        default="allenai/c4",
        help="HuggingFace dataset name (use allenai/mc4 for multilingual, allenai/c4 for English-only)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode (recommended for large datasets)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming mode",
    )

    # Remote infini-gram API options (English only)
    parser.add_argument(
        "--use-infinigram",
        action="store_true",
        help="Use remote infini-gram API for pre-filtering (English only)",
    )
    parser.add_argument(
        "--infinigram-index",
        default="v4_dolma-v1_7_llama",
        choices=INFINIGRAM_REMOTE_INDEXES,
        help="Remote infini-gram index to use (English only)",
    )

    # Local infini-gram options (any language)
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build a local infini-gram index from downloaded mC4 data",
    )
    parser.add_argument(
        "--use-infinigram-local",
        action="store_true",
        help="Use local infini-gram index for filtering (supports any language)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Directory for local infini-gram index (default: <output-dir>/infinigram_index_<lang>)",
    )
    parser.add_argument(
        "--tokenizer",
        default="llama",
        choices=["llama", "gpt2", "olmo"],
        help="Tokenizer for local infini-gram index",
    )
    parser.add_argument(
        "--index-cpus",
        type=int,
        default=8,
        help="Number of CPUs for building index",
    )
    parser.add_argument(
        "--index-mem",
        type=int,
        default=64,
        help="Memory (GB) for building index",
    )
    parser.add_argument(
        "--index-shards",
        type=int,
        default=1,
        help="Number of shards for index",
    )

    # Common infini-gram options
    parser.add_argument(
        "--infinigram-min-count",
        type=int,
        default=1,
        help="Minimum count in infini-gram to keep an idiom",
    )
    parser.add_argument(
        "--infinigram-only",
        action="store_true",
        help="Only run infini-gram analysis, skip mC4 filtering",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of documents to process between progress logs and memory cleanup",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of filtered documents per output chunk file (e.g., mc4_zh_filtered_00000.json.gz)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (for testing)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable gzip compression (use plain .jsonl instead of .json.gz)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file instead of overwriting",
    )
    args = parser.parse_args()

    # Set derived defaults
    if args.idiom_dir is None:
        args.idiom_dir = args.base_dir / "idioms"
    if args.output_dir is None:
        args.output_dir = args.base_dir / "mc4_filtered"

    return args


def load_idioms(idiom_file: Path) -> Set[str]:
    """Load idioms from a JSONL file."""
    idioms = set()
    with open(idiom_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                idiom = data.get("idiom", "")
                if idiom:
                    idioms.add(idiom)
    logger.info(f"Loaded {len(idioms)} idioms from {idiom_file}")
    return idioms


def get_idiom_file(args, lang: str) -> Optional[Path]:
    """Get the idiom file path for a language."""
    if args.idiom_file:
        return args.idiom_file
    idiom_file = args.idiom_dir / lang / "idioms_merged_llm_formatted.jsonl"
    if idiom_file.exists():
        return idiom_file
    logger.warning(f"Idiom file not found: {idiom_file}")
    return None


# =============================================================================
# Remote Infini-gram API Functions (English only)
# =============================================================================

def infinigram_remote_count(query: str, index: str, max_retries: int = 3) -> Dict[str, Any]:
    """Count n-gram occurrences using remote infini-gram API."""
    payload = {
        "index": index,
        "query_type": "count",
        "query": query,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(INFINIGRAM_API_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                logger.warning(f"Failed to query infini-gram for '{query}': {e}")
                return {"count": -1, "error": str(e)}
    return {"count": -1, "error": "Max retries exceeded"}


def filter_idioms_with_remote_infinigram(
    idioms: Set[str],
    index: str,
    min_count: int = 1,
) -> Dict[str, int]:
    """Use remote infini-gram API to filter idioms and get their counts."""
    logger.info(f"Querying remote infini-gram for {len(idioms)} idioms...")
    filtered = {}
    total = len(idioms)

    for i, idiom in enumerate(idioms):
        result = infinigram_remote_count(idiom, index)
        count = result.get("count", 0)

        if count >= min_count:
            filtered[idiom] = count

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{total}, Found: {len(filtered)}")

        # Rate limiting
        time.sleep(0.05)

    logger.info(f"Remote infini-gram: {len(filtered)}/{total} idioms have count >= {min_count}")
    return filtered


# =============================================================================
# Local Infini-gram Functions (any language)
# =============================================================================

def get_local_index_dir(args, lang: str) -> Path:
    """Get the local index directory for a language."""
    if args.index_dir:
        return args.index_dir
    return args.output_dir / f"infinigram_index_{lang}"


def build_local_infinigram_index(
    data_dir: Path,
    index_dir: Path,
    tokenizer: str = "llama",
    cpus: int = 8,
    mem: int = 64,
    shards: int = 1,
) -> bool:
    """
    Build a local infini-gram index from JSONL data.

    The data_dir should contain JSONL files with "text" field.
    """
    logger.info(f"Building local infini-gram index...")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Index dir: {index_dir}")
    logger.info(f"  Tokenizer: {tokenizer}")
    logger.info(f"  CPUs: {cpus}, Memory: {mem}GB, Shards: {shards}")

    index_dir.mkdir(parents=True, exist_ok=True)

    # Get current ulimit to avoid permission errors on shared clusters
    import resource
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Use the current soft limit to avoid permission issues
        ulimit_value = min(soft_limit, 65536)
    except Exception:
        ulimit_value = 1024  # Safe default

    cmd = [
        "python", "-m", "infini_gram.indexing",
        "--data_dir", str(data_dir),
        "--save_dir", str(index_dir),
        "--tokenizer", tokenizer,
        "--cpus", str(cpus),
        "--mem", str(mem),
        "--shards", str(shards),
        "--add_metadata",
        "--ulimit", str(ulimit_value),
    ]

    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Index building completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Index building failed: {e}")
        logger.error(e.stderr)
        return False
    except FileNotFoundError:
        logger.error("infini-gram package not installed. Run: pip install infini-gram")
        return False


def load_local_infinigram_engine(index_dir: Path, tokenizer_name: str = "llama"):
    """Load a local infini-gram engine."""
    try:
        from infini_gram.engine import InfiniGramEngine
        from transformers import AutoTokenizer

        # Map tokenizer name to HuggingFace model
        # Note: meta-llama models are gated; use alternatives if not logged in
        tokenizer_map = {
            "llama": "meta-llama/Llama-2-7b-hf",
            "gpt2": "gpt2",
            "olmo": "allenai/OLMo-7B",
        }

        hf_tokenizer = tokenizer_map.get(tokenizer_name, tokenizer_name)

        logger.info(f"Loading tokenizer: {hf_tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_tokenizer,
            add_bos_token=False,
            add_eos_token=False,
        )

        logger.info(f"Loading infini-gram engine from: {index_dir}")
        engine = InfiniGramEngine(
            index_dir=str(index_dir),
            eos_token_id=tokenizer.eos_token_id,
        )

        return engine, tokenizer

    except ImportError:
        logger.error("infini-gram package not installed. Run: pip install infini-gram")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load infini-gram engine: {e}")
        return None, None


def infinigram_local_count(engine, tokenizer, query: str) -> Dict[str, Any]:
    """Count n-gram occurrences using local infini-gram engine."""
    try:
        input_ids = tokenizer.encode(query)
        result = engine.count(input_ids=input_ids)
        return result
    except Exception as e:
        logger.warning(f"Failed to count '{query}': {e}")
        return {"count": 0, "error": str(e)}


def infinigram_local_find(engine, tokenizer, query: str) -> Dict[str, Any]:
    """Find documents containing an n-gram using local infini-gram engine."""
    try:
        input_ids = tokenizer.encode(query)
        result = engine.find(input_ids=input_ids)
        return result
    except Exception as e:
        logger.warning(f"Failed to find '{query}': {e}")
        return {"error": str(e)}


def filter_idioms_with_local_infinigram(
    engine,
    tokenizer,
    idioms: Set[str],
    min_count: int = 1,
) -> Dict[str, int]:
    """Use local infini-gram engine to filter idioms and get their counts."""
    logger.info(f"Querying local infini-gram for {len(idioms)} idioms...")
    filtered = {}
    total = len(idioms)

    for i, idiom in enumerate(idioms):
        result = infinigram_local_count(engine, tokenizer, idiom)
        count = result.get("count", 0)

        if count >= min_count:
            filtered[idiom] = count

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i+1}/{total}, Found: {len(filtered)}")

    logger.info(f"Local infini-gram: {len(filtered)}/{total} idioms have count >= {min_count}")
    return filtered


def extract_documents_with_local_infinigram(
    engine,
    tokenizer,
    idioms: Set[str],
    output_file: Path,
    max_docs_per_idiom: int = 100,
):
    """
    Extract documents containing idioms using local infini-gram engine.

    This directly retrieves documents from the indexed corpus.
    """
    logger.info(f"Extracting documents for {len(idioms)} idioms...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    seen_docs = set()

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for i, idiom in enumerate(idioms):
            input_ids = tokenizer.encode(idiom)

            try:
                find_result = engine.find(input_ids=input_ids)
                docs_found = 0

                for s, (start, end) in enumerate(find_result.get('segment_by_shard', [])):
                    for rank in range(start, min(end, start + max_docs_per_idiom)):
                        try:
                            doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=10000)
                            doc_text = doc.get('doc', '')

                            # Deduplicate
                            doc_hash = hash(doc_text[:500])
                            if doc_hash in seen_docs:
                                continue
                            seen_docs.add(doc_hash)

                            output_doc = {
                                "text": doc_text,
                                "matched_idiom": idiom,
                                "source": "infinigram-local",
                            }
                            out_f.write(json.dumps(output_doc, ensure_ascii=False) + '\n')
                            total_docs += 1
                            docs_found += 1

                        except Exception as e:
                            logger.debug(f"Failed to get doc: {e}")
                            break

                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i+1}/{len(idioms)} idioms, {total_docs} docs extracted")

            except Exception as e:
                logger.warning(f"Failed to find docs for '{idiom}': {e}")

    logger.info(f"Extracted {total_docs} documents to {output_file}")
    return total_docs


# =============================================================================
# Local Matching Functions
# =============================================================================

def create_english_matcher(idioms: Set[str]):
    """Create a loose matcher for English idioms that handles morphological variants."""
    patterns = []

    for idiom in idioms:
        # Remove content in parentheses for the base pattern
        base_idiom = re.sub(r'\([^)]*\)', '', idiom).strip()
        if not base_idiom:
            base_idiom = idiom

        # Escape special regex characters
        escaped = re.escape(base_idiom)

        # Replace escaped spaces with flexible whitespace
        pattern = escaped.replace(r'\ ', r'\s+')

        # Handle common verb endings
        words = pattern.split(r'\s+')
        flexible_words = []
        for word in words:
            # Add optional common suffixes
            flexible_word = word + r"(?:'?s|ed|ing|er|est|n't|'d|'ll|'ve|'re|ies|ied)?"
            flexible_words.append(flexible_word)

        flexible_pattern = r'\s+'.join(flexible_words)
        patterns.append(flexible_pattern)

    # Combine patterns with word boundaries
    combined_pattern = r'\b(?:' + '|'.join(patterns) + r')\b'

    try:
        compiled = re.compile(combined_pattern, re.IGNORECASE)
        logger.info(f"Created English matcher with {len(patterns)} patterns")
        return compiled
    except re.error as e:
        logger.error(f"Regex compilation error: {e}")
        return create_batched_english_matcher(patterns)


def create_batched_english_matcher(patterns: List[str], batch_size: int = 500):
    """Create multiple regex matchers in batches."""
    matchers = []
    for i in range(0, len(patterns), batch_size):
        batch = patterns[i:i + batch_size]
        combined = r'\b(?:' + '|'.join(batch) + r')\b'
        try:
            matchers.append(re.compile(combined, re.IGNORECASE))
        except re.error as e:
            logger.warning(f"Batch {i//batch_size} failed: {e}")
    logger.info(f"Created {len(matchers)} batched English matchers")
    return matchers


def contains_english_idiom(text: str, matcher) -> bool:
    """Check if text contains any English idiom."""
    if isinstance(matcher, list):
        for m in matcher:
            if m.search(text):
                return True
        return False
    else:
        return matcher.search(text) is not None


def contains_chinese_idiom(text: str, idioms: Set[str]) -> bool:
    """Check if text contains any Chinese idiom (exact substring match)."""
    for idiom in idioms:
        if idiom in text:
            return True
    return False


def create_matcher(lang: str, idioms: Set[str]):
    """Create appropriate matcher for language."""
    if lang == "en":
        return create_english_matcher(idioms)
    else:
        return idioms


def check_contains_idiom(text: str, lang: str, matcher) -> bool:
    """Check if text contains idiom based on language."""
    if lang == "en":
        return contains_english_idiom(text, matcher)
    else:
        return contains_chinese_idiom(text, matcher)


# =============================================================================
# Main Processing
# =============================================================================

class ChunkedGzipWriter:
    """
    Writes documents to chunked, gzip-compressed files.

    Files are named: {base_name}_{chunk_id:05d}.json.gz
    A new chunk is created every chunk_size documents.
    """

    def __init__(
        self,
        output_dir: Path,
        base_name: str,
        chunk_size: int = 10000,
        compress: bool = True,
    ):
        self.output_dir = output_dir
        self.base_name = base_name
        self.chunk_size = chunk_size
        self.compress = compress
        self.current_chunk_id = 0
        self.current_chunk_count = 0
        self.current_file = None
        self.total_written = 0
        self.chunk_files = []

        output_dir.mkdir(parents=True, exist_ok=True)

    def _get_chunk_path(self) -> Path:
        ext = ".json.gz" if self.compress else ".jsonl"
        return self.output_dir / f"{self.base_name}_{self.current_chunk_id:05d}{ext}"

    def _open_new_chunk(self):
        if self.current_file:
            self.current_file.close()

        chunk_path = self._get_chunk_path()
        self.chunk_files.append(chunk_path)

        if self.compress:
            self.current_file = gzip.open(chunk_path, 'wt', encoding='utf-8')
        else:
            self.current_file = open(chunk_path, 'w', encoding='utf-8')

        logger.info(f"Writing to chunk: {chunk_path}")

    def write(self, doc_json: str):
        if self.current_file is None or self.current_chunk_count >= self.chunk_size:
            if self.current_file:
                self.current_chunk_id += 1
            self.current_chunk_count = 0
            self._open_new_chunk()

        self.current_file.write(doc_json + '\n')
        self.current_chunk_count += 1
        self.total_written += 1

    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def download_and_filter_chunked(
    args,
    lang: str,
    idioms: Set[str],
    output_dir: Path,
    index_data_dir: Optional[Path] = None,
) -> tuple:
    """
    Download and filter mC4 data in chunks.

    Processes documents in streaming fashion:
    - Downloads and filters documents one at a time
    - Saves only matching documents to chunked, compressed files
    - Creates files like: mc4_zh_filtered_00000.json.gz, mc4_zh_filtered_00001.json.gz, etc.
    - Optionally saves filtered docs for index building
    - Does NOT keep unfiltered raw data

    Returns (total_processed, total_matched)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create chunked writer for filtered output
    writer = ChunkedGzipWriter(
        output_dir=output_dir,
        base_name=f"mc4_{lang}_filtered",
        chunk_size=args.chunk_size,
        compress=not args.no_compress,
    )

    # If building index, we'll save filtered docs for indexing
    if index_data_dir:
        index_data_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_data_dir / f"mc4_{lang}.jsonl"
        index_f = open(index_file, 'w', encoding='utf-8')
    else:
        index_f = None

    matcher = create_matcher(lang, idioms)

    logger.info(f"Starting chunked processing of {args.dataset} {lang}...")
    logger.info(f"  Batch size (logging): {args.batch_size}")
    logger.info(f"  Chunk size (output): {args.chunk_size} docs per file")
    logger.info(f"  Compression: {'gzip' if not args.no_compress else 'none'}")
    logger.info(f"  Max docs: {args.max_docs or 'unlimited'}")

    dataset = load_dataset(
        args.dataset,
        lang,
        split=args.split,
        streaming=args.streaming,
    )

    total_processed = 0
    total_matched = 0

    with writer:
        for doc in dataset:
            total_processed += 1
            text = doc.get("text", "")

            if check_contains_idiom(text, lang, matcher):
                total_matched += 1
                timestamp = doc.get("timestamp", "")
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()

                output_doc = {
                    "text": text,
                    "url": doc.get("url", ""),
                    "timestamp": timestamp,
                    "source": f"mc4-{lang}",
                }
                writer.write(json.dumps(output_doc, ensure_ascii=False))

                # Also write to index file if building index
                if index_f:
                    index_f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

            # Log progress at batch_size intervals
            if total_processed % args.batch_size == 0:
                logger.info(
                    f"[{lang}] Processed: {total_processed:,}, "
                    f"Matched: {total_matched:,} "
                    f"({100*total_matched/total_processed:.4f}%)"
                )
                gc.collect()

            if args.max_docs and total_processed >= args.max_docs:
                logger.info(f"Reached max_docs limit: {args.max_docs}")
                break

    if index_f:
        index_f.close()
        logger.info(f"Saved {total_matched} filtered docs for indexing to {index_data_dir}")

    logger.info(
        f"[{lang}] Completed! Total: {total_processed:,}, "
        f"Matched: {total_matched:,} ({100*total_matched/max(1,total_processed):.4f}%)"
    )
    logger.info(f"Output files: {len(writer.chunk_files)} chunks in {output_dir}")

    return total_processed, total_matched


def filter_and_save_documents(
    args,
    lang: str,
    idioms: Set[str],
    output_file: Path,
):
    """Download, filter, and save documents containing idioms."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    matcher = create_matcher(lang, idioms)

    logger.info(f"Starting to process {args.dataset} {lang} subset...")

    try:
        dataset = load_dataset(
            args.dataset,
            lang,
            split=args.split,
            streaming=args.streaming,
            )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    total_processed = 0
    total_matched = 0
    batch_docs = []

    mode = 'a' if args.resume else 'w'
    if args.resume and output_file.exists():
        logger.info(f"Resuming from existing file: {output_file}")

    with open(output_file, mode, encoding='utf-8') as out_f:
        for doc in dataset:
            total_processed += 1
            text = doc.get("text", "")

            if check_contains_idiom(text, lang, matcher):
                total_matched += 1
                timestamp = doc.get("timestamp", "")
                if hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                output_doc = {
                    "text": text,
                    "url": doc.get("url", ""),
                    "timestamp": timestamp,
                    "source": f"mc4-{lang}",
                }
                batch_docs.append(json.dumps(output_doc, ensure_ascii=False))

            if len(batch_docs) >= 100:
                out_f.write('\n'.join(batch_docs) + '\n')
                batch_docs = []

            if total_processed % args.batch_size == 0:
                logger.info(
                    f"[{lang}] Processed: {total_processed:,}, "
                    f"Matched: {total_matched:,} "
                    f"({100*total_matched/total_processed:.4f}%)"
                )
                gc.collect()

            if args.max_docs and total_processed >= args.max_docs:
                logger.info(f"Reached max_docs limit: {args.max_docs}")
                break

        if batch_docs:
            out_f.write('\n'.join(batch_docs) + '\n')

    logger.info(
        f"[{lang}] Completed! Total: {total_processed:,}, "
        f"Matched: {total_matched:,} ({100*total_matched/max(1,total_processed):.4f}%)"
    )

    return total_processed, total_matched


def run_remote_infinigram_analysis(args, lang: str, idioms: Set[str]) -> Dict[str, Any]:
    """Run remote infini-gram analysis on idioms (English only)."""
    if lang != "en":
        logger.warning(f"Remote infini-gram only supports English. Skipping {lang}.")
        return {"skipped": True, "reason": "Remote infini-gram only supports English"}

    logger.info(f"\n{'='*60}")
    logger.info(f"Running remote infini-gram analysis for {lang}")
    logger.info(f"Index: {args.infinigram_index}")
    logger.info(f"{'='*60}")

    idiom_counts = filter_idioms_with_remote_infinigram(
        idioms,
        args.infinigram_index,
        args.infinigram_min_count,
    )

    output_file = args.output_dir / f"infinigram_remote_{lang}_counts.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "mode": "remote",
        "index": args.infinigram_index,
        "min_count": args.infinigram_min_count,
        "total_idioms": len(idioms),
        "filtered_idioms": len(idiom_counts),
        "idiom_counts": idiom_counts,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to: {output_file}")
    return result


def run_local_infinigram_analysis(args, lang: str, idioms: Set[str]) -> Dict[str, Any]:
    """Run local infini-gram analysis on idioms."""
    index_dir = get_local_index_dir(args, lang)

    if not index_dir.exists():
        logger.error(f"Local index not found: {index_dir}")
        logger.error("Build it first with: --build-index --index-dir <path>")
        return {"error": f"Index not found: {index_dir}"}

    logger.info(f"\n{'='*60}")
    logger.info(f"Running local infini-gram analysis for {lang}")
    logger.info(f"Index: {index_dir}")
    logger.info(f"{'='*60}")

    engine, tokenizer = load_local_infinigram_engine(index_dir, args.tokenizer)
    if engine is None:
        return {"error": "Failed to load infini-gram engine"}

    idiom_counts = filter_idioms_with_local_infinigram(
        engine, tokenizer, idioms, args.infinigram_min_count
    )

    output_file = args.output_dir / f"infinigram_local_{lang}_counts.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "mode": "local",
        "index_dir": str(index_dir),
        "tokenizer": args.tokenizer,
        "min_count": args.infinigram_min_count,
        "total_idioms": len(idioms),
        "filtered_idioms": len(idiom_counts),
        "idiom_counts": idiom_counts,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to: {output_file}")
    return result


def main():
    args = parse_args()

    logger.info(f"Configuration:")
    logger.info(f"  Languages: {args.lang}")
    logger.info(f"  Base dir: {args.base_dir}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Build index: {args.build_index}")
    logger.info(f"  Use remote infini-gram: {args.use_infinigram}")
    logger.info(f"  Use local infini-gram: {args.use_infinigram_local}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for lang in args.lang:
        idiom_file = get_idiom_file(args, lang)
        if not idiom_file:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing language: {lang}")
        logger.info(f"{'='*60}")

        idioms = load_idioms(idiom_file)

        # Run remote infini-gram analysis first (pre-filter idioms)
        if args.use_infinigram:
            infinigram_result = run_remote_infinigram_analysis(args, lang, idioms)
            results[f"{lang}_infinigram_remote"] = infinigram_result

            if infinigram_result.get("idiom_counts"):
                idioms = set(infinigram_result["idiom_counts"].keys())
                logger.info(f"Using {len(idioms)} idioms filtered by remote infini-gram")

        # Skip mC4 filtering if infini-gram only mode
        if args.infinigram_only:
            continue

        # Process mC4 with chunked filtering
        # Output will be chunked files: mc4_{lang}_filtered_00000.json.gz, etc.
        output_subdir = args.output_dir / lang

        # Clean up existing chunk files if not resuming
        if not args.resume and output_subdir.exists():
            ext = ".jsonl" if args.no_compress else ".json.gz"
            existing_chunks = list(output_subdir.glob(f"mc4_{lang}_filtered_*{ext}"))
            if existing_chunks:
                logger.info(f"Removing {len(existing_chunks)} existing chunk files...")
                for chunk_file in existing_chunks:
                    chunk_file.unlink()

        # Determine if we need to save data for index building
        index_data_dir = None
        if args.build_index:
            index_data_dir = args.output_dir / f"mc4_{lang}_raw"

        try:
            # Chunked download + filter: only keeps matching documents
            total_processed, total_matched = download_and_filter_chunked(
                args,
                lang=lang,
                idioms=idioms,
                output_dir=output_subdir,
                index_data_dir=index_data_dir,
            )
            results[lang] = {
                "total_processed": total_processed,
                "total_matched": total_matched,
                "output_dir": str(output_subdir),
            }
        except Exception as e:
            logger.error(f"Error processing {lang}: {e}")
            results[lang] = {"error": str(e)}
            continue

        # Build local index from FILTERED docs only (much smaller!)
        if args.build_index and index_data_dir:
            index_dir = get_local_index_dir(args, lang)

            logger.info(f"Building index from {total_matched} filtered documents...")
            success = build_local_infinigram_index(
                data_dir=index_data_dir,
                index_dir=index_dir,
                tokenizer=args.tokenizer,
                cpus=args.index_cpus,
                mem=args.index_mem,
                shards=args.index_shards,
            )
            results[f"{lang}_index"] = {"success": success, "index_dir": str(index_dir)}

            if not success:
                continue

        # Run local infini-gram analysis (after building index)
        if args.use_infinigram_local:
            infinigram_result = run_local_infinigram_analysis(args, lang, idioms)
            results[f"{lang}_infinigram_local"] = infinigram_result

            if infinigram_result.get("idiom_counts"):
                logger.info(f"Local infini-gram found {infinigram_result['filtered_idioms']} idioms in corpus")

    # Save summary
    summary_file = args.output_dir / "filtering_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info("All processing completed!")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"{'='*60}")

    for key, result in results.items():
        if "error" in result:
            logger.info(f"  {key}: ERROR - {result['error']}")
        elif "total_matched" in result:
            logger.info(f"  {key}: {result['total_matched']:,} / {result['total_processed']:,} matched")
        elif "filtered_idioms" in result:
            logger.info(f"  {key}: {result['filtered_idioms']} / {result['total_idioms']} idioms in corpus")
        elif "success" in result:
            logger.info(f"  {key}: {'SUCCESS' if result['success'] else 'FAILED'}")


if __name__ == "__main__":
    main()


"""
# Step 1: Basic filtering (outputs chunked .json.gz files)
python download_and_filter_mc4.py --lang zh --chunk-size 10000
# Output: mc4_filtered/zh/mc4_zh_filtered_00000.json.gz, mc4_zh_filtered_00001.json.gz, ...

# Step 2: Build local infini-gram index from filtered docs
python download_and_filter_mc4.py --lang zh --build-index \
    --index-dir ./index_zh \
    --index-cpus 16 --index-mem 128 --index-shards 2

# Step 3: Use local index for idiom counting
python download_and_filter_mc4.py --lang zh --use-infinigram-local \
    --index-dir ./index_zh

New arguments:

Argument	Description
--build-index	Build local infini-gram index from filtered docs
--use-infinigram-local	Use local index for filtering (any language)
--index-dir	Directory for local index
--tokenizer	Tokenizer: llama, gpt2, olmo
--index-cpus	CPUs for indexing (default: 8)
--index-mem	Memory GB for indexing (default: 64)
--index-shards	Number of shards (default: 1)
--chunk-size	Documents per output chunk file (default: 10000)
--no-compress	Disable gzip compression (use .jsonl instead of .json.gz)
--resume	Resume from existing output file instead of overwriting
Storage requirement: ~7 bytes per token indexed.
"""
