import argparse
import json
import os
import numpy as np
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from sentence_transformers import SentenceTransformer


def get_chinese_font():
    """
    Try to find and return a font that supports Chinese characters.
    Returns a tuple (font_name, font_path) if found, (None, None) otherwise.
    font_path can be None if using a system font.
    """
    # First, check local fonts directory (project-specific fonts)
    local_fonts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'culture', 'data', 'fonts'
    )
    
    # Prefer local fonts first
    if os.path.exists(local_fonts_dir):
        # Preferred order: static fonts first, then variable fonts
        preferred_files = [
            'LxgwWenKai-Regular.ttf',
            'NotoSansCJKsc-Regular.ttf',
            'NotoSansCJKsc-VF.ttf',
        ]
        
        for font_file in preferred_files:
            font_path = os.path.join(local_fonts_dir, font_file)
            if os.path.exists(font_path):
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    if font_name:
                        # Register the font with matplotlib
                        fm.fontManager.addfont(font_path)
                        return (font_name, font_path)
                except Exception as e:
                    continue
        
        # If preferred files not found, search for any Chinese font in the directory
        for file in os.listdir(local_fonts_dir):
            if file.endswith(('.ttf', '.otf', '.ttc')):
                file_lower = file.lower()
                if any(keyword in file_lower for keyword in ['noto', 'cjk', 'han', 'chinese', 'simhei', 'simsun', 'wenkai', 'lxgw']):
                    font_path = os.path.join(local_fonts_dir, file)
                    try:
                        font_prop = fm.FontProperties(fname=font_path)
                        font_name = font_prop.get_name()
                        if font_name:
                            fm.fontManager.addfont(font_path)
                            return (font_name, font_path)
                    except Exception:
                        continue
    
    # List of common Chinese font names to try (in order of preference)
    chinese_fonts = [
        'Noto Sans CJK SC',
        'Noto Sans CJK TC',
        'Source Han Sans SC',
        'Source Han Sans TC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'STHeiti',
        'STSong',
        'AR PL UMing CN',
        'AR PL UKai CN',
        'LXGW WenKai',
    ]
    
    # Get all available fonts with their full paths
    available_fonts = {f.name: f for f in fm.fontManager.ttflist}
    
    # Try to find a Chinese font from our preferred list
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            # Double check: exclude DejaVu Sans which doesn't support Chinese
            if font_name != 'DejaVu Sans':
                return (font_name, None)
    
    # If no specific font found, try to find any font with CJK-related keywords
    for font_name, font_prop in available_fonts.items():
        font_lower = font_name.lower()
        # Exclude DejaVu Sans and other non-Chinese fonts
        if font_name == 'DejaVu Sans' or 'dejavu' in font_lower:
            continue
        if any(keyword in font_lower for keyword in ['cjk', 'han', 'chinese', 'simhei', 'simsun', 'ming', 'kai', 'wenkai', 'lxgw']):
            return (font_name, None)
    
    # Last resort: try to find fonts by checking system font directories
    try:
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts'),
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if file.endswith(('.ttf', '.otf', '.ttc')):
                            file_lower = file.lower()
                            # Skip DejaVu fonts
                            if 'dejavu' in file_lower:
                                continue
                            if any(keyword in file_lower for keyword in ['noto', 'cjk', 'han', 'chinese', 'simhei', 'simsun', 'wenkai', 'lxgw']):
                                # Try to get font name from file
                                try:
                                    font_path = os.path.join(root, file)
                                    font_prop = fm.FontProperties(fname=font_path)
                                    font_name = font_prop.get_name()
                                    if font_name and font_name != 'DejaVu Sans':
                                        return (font_name, font_path)
                                except:
                                    continue
    except Exception:
        pass
    
    return (None, None)


def load_entities(jsonl_path):
    """
    Load unique entities from a JSONL file.
    """
    entities = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                ents = obj.get("output", {}).get("entities", [])
            except:
                continue
            entities.extend(ents)

    # Deduplicate while preserving order
    seen = set()
    unique_entities = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    return unique_entities


def compute_embeddings(entities, model_name):
    """
    Compute embeddings for a list of entities.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        entities,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def save_embeddings(entities, embeddings, output_path):
    """
    Save entity embeddings to disk as JSONL.
    """
    with open(output_path, "w", encoding="utf8") as f:
        for entity, emb in zip(entities, embeddings):
            f.write(json.dumps({
                "entity": entity,
                "embedding": emb.tolist()
            }, ensure_ascii=False) + "\n")


def cluster_embeddings(embeddings, n_clusters):
    """
    Cluster embeddings using KMeans.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def find_central_entities(entities, embeddings, labels, centroids):
    """
    Identify the most central entity in each cluster.
    """
    cluster_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_to_indices[label].append(idx)

    central_entities = {}

    for cluster_id, indices in cluster_to_indices.items():
        cluster_embs = embeddings[indices]
        centroid = centroids[cluster_id].reshape(1, -1)

        distances = pairwise_distances(cluster_embs, centroid, metric="cosine")
        min_idx = indices[int(distances.argmin())]

        central_entities[cluster_id] = entities[min_idx]

    return central_entities


def plot_clusters_2d(embeddings, labels, entities, central_entities, output_path):
    """
    Plot clusters in 2D using PCA, with different colors per cluster.
    Save the plot as PDF with bbox_inches='tight'.
    """
    # Try to set a Chinese font BEFORE creating the figure
    font_name, font_path = get_chinese_font()
    if font_name:
        # Set font for matplotlib
        if font_path:
            # Use FontProperties with the font file path for local fonts
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            # Also set it in sans-serif list
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial', 'sans-serif']
        else:
            # Use system font by name
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial', 'sans-serif']
        
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
        print(f"Using font: {font_name}" + (f" (from {font_path})" if font_path else ""))
    else:
        print("Warning: No Chinese font found. Chinese characters may not display correctly.")
        print("To install a Chinese font on Linux, try:")
        print("  sudo apt-get install fonts-noto-cjk  # For Debian/Ubuntu")
        print("  sudo yum install google-noto-cjk-fonts  # For RHEL/CentOS")
    
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            reduced[idxs, 0],
            reduced[idxs, 1],
            label=f"Cluster {label}",
            alpha=0.6
        )

    # Mark central entities
    for cluster_id, entity in central_entities.items():
        idx = entities.index(entity)
        plt.scatter(
            reduced[idx, 0],
            reduced[idx, 1],
            color="black",
            marker="x",
            s=100
        )
        # Use FontProperties for text rendering if we have a Chinese font
        if font_name:
            if font_path:
                text_font = fm.FontProperties(fname=font_path)
            else:
                text_font = fm.FontProperties(family=font_name)
            plt.text(
                reduced[idx, 0],
                reduced[idx, 1],
                entity,
                fontsize=10,
                weight="bold",
                fontproperties=text_font
            )
        else:
            plt.text(
                reduced[idx, 0],
                reduced[idx, 1],
                entity,
                fontsize=10,
                weight="bold"
            )

    plt.title("Entity Clusters (PCA 2D)")
    plt.legend()
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def print_central_entities(central_entities):
    """
    Print the most central entity for each cluster.
    """
    print("\nMost central entity per cluster:\n")
    for cluster_id in sorted(central_entities):
        print(f"Cluster {cluster_id}: {central_entities[cluster_id]}")


def main(args):
    entities = load_entities(args.input)
    print(f"Loaded {len(entities)} unique entities")

    embeddings = compute_embeddings(entities, args.model)
    save_embeddings(entities, embeddings, args.embedding_output)

    labels, centroids = cluster_embeddings(embeddings, args.n_clusters)
    central_entities = find_central_entities(
        entities, embeddings, labels, centroids
    )

    print_central_entities(central_entities)

    if args.plot:
        # Generate output path for plot
        plots_dir = "/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/src/culture/analysis/plots"
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        plot_output_path = os.path.join(plots_dir, f"{input_basename}_clusters.pdf")
        
        plot_clusters_2d(
            embeddings, labels, entities, central_entities, plot_output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--embedding_output", type=str, required=True, help="Output embedding JSONL file")
    parser.add_argument("--model", type=str, default="/home/jiaruil5/math_rl/mix_teachers/r3lit_rl/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    main(args)
