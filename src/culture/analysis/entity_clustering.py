import argparse
import json
import time
import asyncio
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from googletrans import Translator


def get_chinese_font():
    """
    Try to find and return a font that supports Chinese characters.
    Returns a tuple (font_name, font_path) if found, (None, None) otherwise.
    font_path can be None if using a system font.
    """
    import os
    
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


def _run_async_sync(coro):
    """
    Run an async coroutine synchronously.
    Handles both cases: when event loop exists and when it doesn't.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running (e.g., in Jupyter), we need to use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(coro)
            except ImportError:
                # Fallback: create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
        else:
            # Loop exists but not running, we can use it
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(coro)


def translate_entities(entities, src='zh', dest='en'):
    """
    Translate a list of entities from source language to destination language.
    Uses synchronous translation with retry logic.
    
    Args:
        entities: List of entity strings to translate
        src: Source language code (default: 'zh' for Chinese)
        dest: Destination language code (default: 'en' for English)
        
    Returns:
        List of translated strings
    """
    translator = Translator()
    translated_list = []
    max_tries = 5
    
    for entity in entities:
        curr_tries = 0
        while True:
            try:
                # translate() returns a coroutine in googletrans 4.x
                translate_coro = translator.translate(
                    entity,
                    dest=dest,
                    src=src
                )
                # Run the coroutine synchronously
                translated_text = _run_async_sync(translate_coro)
                translated_list.append(translated_text.text)
                break
            except Exception as e:
                print(f"Translation error for '{entity}': {e}")
                if curr_tries < max_tries:
                    curr_tries += 1
                    time.sleep(5)
                else:
                    # If translation fails after max tries, use original entity
                    print(f"Failed to translate '{entity}' after {max_tries} tries. Using original.")
                    translated_list.append(entity)
                    break
    
    return translated_list


def load_entities_from_file(file_path):
    """
    Load entities from a JSONL file and return a Counter of entity frequencies.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        Counter: A Counter object with entity frequencies
    """
    entity_counter = Counter()
    
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                entities = obj.get("output", {}).get("entities", [])
            except:
                print(obj)
            for e in entities:
                entity_counter[e] += 1
    
    return entity_counter


def plot_entity_distribution(entity_counter, top_k=30, src_lang='zh'):
    """
    Plot the frequency distribution of entities using matplotlib pyplot.
    For Chinese entities, displays labels in format "Chinese + English translation".
    For English entities, displays labels without translation.
    
    Args:
        entity_counter: Counter object with entity frequencies
        top_k: Number of top entities to plot
        src_lang: Source language code (default: 'zh' for Chinese, 'en' for English)
    """
    most_common = entity_counter.most_common(top_k)
    if not most_common:
        print("No entities found. Nothing to plot.")
        return

    entities, counts = zip(*most_common)
    
    # Only translate if source language is not English
    if src_lang == 'en':
        # Use original entity names directly for English
        entity_labels = list(entities)
    else:
        # Translate entities to English
        print("Translating entities to English...")
        try:
            translated_entities = translate_entities(entities, src=src_lang, dest='en')
            # Format labels as "Original + English"
            entity_labels = [f"{original} + {english}" for original, english in zip(entities, translated_entities)]
        except Exception as e:
            print(f"Warning: Translation failed: {e}. Using original entity names.")
            entity_labels = list(entities)

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
        print("\nOr check available fonts with:")
        print("  import matplotlib.font_manager as fm")
        print("  [f.name for f in fm.fontManager.ttflist if 'CJK' in f.name or 'Chinese' in f.name]")

    plt.figure(figsize=(12, 6))
    plt.bar(entities, counts)
    plt.xticks(range(len(entities)), entity_labels, rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Entity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(f"Top {top_k} Entity Frequency Distribution", fontsize=14)
    plt.tight_layout()
    plt.show()


def main(args):
    entity_counter = Counter()

    with open(args.input, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            entities = obj.get("output", {}).get("entities", [])
            for e in entities:
                entity_counter[e] += 1

    plot_entity_distribution(entity_counter, top_k=args.top_k, src_lang=args.src_lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top entities to plot")
    parser.add_argument("--src_lang", type=str, default='zh', help="Source language code (default: 'zh' for Chinese, 'en' for English)")
    args = parser.parse_args()

    main(args)
