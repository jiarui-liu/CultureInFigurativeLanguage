import os
import argparse
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--local_dir", type=str, default="/home/jiaruil5/latent_reasoning/recursive_multistep/recursive_multistep/models")
    args = parser.parse_args()

    local_dir = os.path.join(args.local_dir, args.model_name)

    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        # Download model to a specific local directory without symlinks or hash folders
        local_dir = snapshot_download(
            repo_id=args.model_name,       # Replace with the model you want
            repo_type="model",
            local_dir=local_dir,  # Custom local folder
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN")
        )

    print(f"Model saved to: {local_dir}")
