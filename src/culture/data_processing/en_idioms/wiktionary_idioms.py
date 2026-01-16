import requests
import time
import re
import json
import argparse

API_ENDPOINT = "https://en.wiktionary.org/w/api.php"
CATEGORY = "Category:English_idioms"
MAX_RETRIES = 3
HEADERS = {
    "User-Agent": "IdiomCrawler/1.0 (Academic research; contact: jiaruil5@andrew.cmu.edu)"
}


def safe_request(params, retries=MAX_RETRIES):
    """Make a request with retry logic and error handling."""
    for attempt in range(retries):
        try:
            response = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                print(f"  All retries failed, skipping...")
                return None
    return None


def fetch_all_idioms():
    idioms = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": CATEGORY,
        "cmlimit": "500",
        "cmnamespace": 0
    }

    while True:
        r = safe_request(params)
        if r is None:
            print("Failed to fetch idiom list, aborting.")
            break
        members = r.get("query", {}).get("categorymembers", [])
        for m in members:
            idioms.append(m["title"])
        if "continue" in r:
            params.update(r["continue"])
            time.sleep(0.1)
        else:
            break

    return idioms

def fetch_idiom_wikitext(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "titles": title
    }
    r = safe_request(params)
    if r is None:
        return ""
    try:
        pages = r["query"]["pages"]
        page = list(pages.values())[0]
        return page["revisions"][0]["slots"]["main"]["*"] if "revisions" in page else ""
    except (KeyError, IndexError) as e:
        print(f"  Error parsing response for '{title}': {e}")
        return ""

def parse_english_section(wikitext):
    # Split on English section header
    parts = re.split(r"==\s*English\s*==", wikitext)
    if len(parts) < 2:
        return None

    # Take text after English
    english_text = parts[1]

    # Stop at next language section (== ...)
    english_text = re.split(r"==\s*[A-Za-z]+\s*==", english_text)[0]

    return english_text

def extract_definitions(english_wikitext):
    meanings = []
    lines = english_wikitext.split("\n")

    current_def = None

    for line in lines:
        # Detect definition lines starting with "# "
        if line.startswith("# "):
            text = line[2:].strip()
            current_def = {"definition": text, "examples": []}
            meanings.append(current_def)

        # Detect example lines that start with "#*: "
        elif line.startswith("#*") and current_def:
            ex_text = line.lstrip("#* ").strip()
            current_def["examples"].append(ex_text)

    return meanings

def crawl_idioms(output_path):
    all_idioms = fetch_all_idioms()
    print(f"Found {len(all_idioms)} idioms")

    with open(output_path, "a", encoding="utf-8") as f:
        for i, idiom in enumerate(all_idioms):
            print(f"[{i+1}/{len(all_idioms)}] {idiom}")

            wikitext = fetch_idiom_wikitext(idiom)
            
            if wikitext:
                record = {"idiom": idiom, "wikitext": wikitext}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # ensure data is written immediately

            time.sleep(0.2)  # be polite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl English idioms from Wiktionary")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="english_idioms_full.jsonl",
        help="Output path for the JSONL file (default: english_idioms_full.jsonl)"
    )
    args = parser.parse_args()

    crawl_idioms(args.output)
    print("Done!")
