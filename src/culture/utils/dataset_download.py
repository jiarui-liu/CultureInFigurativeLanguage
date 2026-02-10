import argparse
from huggingface_hub import snapshot_download

# Usage:
#   python dataset_download.py /path/to/save/folder
#   python dataset_download.py /path/to/save/folder --repo_name my-dataset

def main():
    parser = argparse.ArgumentParser(description='Download dataset from Hugging Face Hub')
    parser.add_argument('dataset_path', type=str, help='Local path to save the downloaded dataset')
    parser.add_argument('--repo_name', type=str, default='loop-llm',
                        help='Repository name on Hugging Face (default: loop-llm)')
    parser.add_argument('--username', type=str, default='Jerry999',
                        help='Hugging Face username (default: Jerry999)')
    parser.add_argument('--allow_patterns', type=str, nargs='*', default=None,
                        help='Only download files matching these patterns (e.g. "*.json" "*.parquet")')
    parser.add_argument('--ignore_patterns', type=str, nargs='*', default=None,
                        help='Skip files matching these patterns (e.g. "*.bin" "*.safetensors")')
    args = parser.parse_args()

    # Configuration
    token = "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"
    full_repo_id = f"{args.username}/{args.repo_name}"

    # Download dataset
    print(f"Downloading {full_repo_id} to {args.dataset_path}...")
    snapshot_download(
        repo_id=full_repo_id,
        repo_type="dataset",
        local_dir=args.dataset_path,
        token=token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )

    print("Download complete!")


if __name__ == '__main__':
    main()
