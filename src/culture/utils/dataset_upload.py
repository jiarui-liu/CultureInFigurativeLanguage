import argparse
from huggingface_hub import HfApi, create_repo, upload_folder, upload_large_folder

# Usage:
#   python data_upload.py /path/to/your/folder
#   python data_upload.py /path/to/your/folder --large

def main():
    parser = argparse.ArgumentParser(description='Upload dataset to Hugging Face Hub')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset folder to upload')
    parser.add_argument('--large', action='store_true',
                        help='Use upload_large_folder for large datasets (>50GB or >10k files)')
    parser.add_argument('--sync', action='store_true',
                        help='Delete remote files not present locally (makes remote mirror local exactly). Not supported with --large.')
    parser.add_argument('--repo_name', type=str, default='loop-llm',
                        help='Repository name on Hugging Face (default: loop-llm)')
    parser.add_argument('--username', type=str, default='Jerry999',
                        help='Hugging Face username (default: Jerry999)')
    args = parser.parse_args()

    # Configuration
    token = "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"
    full_repo_id = f"{args.username}/{args.repo_name}"

    api = HfApi()

    if args.sync and args.large:
        print("Warning: --sync is not supported with --large. Remote files not present locally will NOT be deleted.")

    if args.large:
        # Upload large folder (for datasets >50GB or >10k files)
        print(f"Uploading large folder to {full_repo_id}...")
        upload_large_folder(
            folder_path=args.dataset_path,
            repo_id=full_repo_id,
            repo_type="dataset",
            token=token,
        )
    else:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=full_repo_id, repo_type="dataset", token=token, exist_ok=True)

        # Upload dataset folder
        print(f"Uploading folder to {full_repo_id}...")
        upload_kwargs = {
            "repo_id": full_repo_id,
            "repo_type": "dataset",
            "folder_path": args.dataset_path,
            "token": token,
        }
        if args.sync:
            upload_kwargs["delete_patterns"] = "*"
            print("Sync mode enabled: remote files not present locally will be deleted.")
        upload_folder(**upload_kwargs)

    print("Upload complete!")


if __name__ == '__main__':
    main()
