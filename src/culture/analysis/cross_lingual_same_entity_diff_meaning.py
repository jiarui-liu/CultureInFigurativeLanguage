"""
Cross-lingual Entity Analysis for Idioms

This script analyzes how the same entities convey different meanings across
English and Chinese idioms. It:
1. Extracts top-k entities from each language
2. Translates entities to the other language with high recall
3. Finds idioms containing these entities in both languages
4. Uses GPT-5.2 to analyze cultural/meaning differences
"""

import argparse
import json
import os
import re
import asyncio
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from tqdm import tqdm
import time

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llm_utils import ChatModel


@dataclass
class EntityTranslation:
    """Represents an entity and its translations."""
    source_entity: str
    source_lang: str
    translations: List[str]  # Multiple translations for high recall
    target_lang: str


@dataclass
class IdiomMatch:
    """Represents an idiom that contains a specific entity."""
    idiom: str
    entity: str
    literal_meanings: List[str]
    figurative_meanings: List[str]
    lang: str


@dataclass
class CrossLingualEntityPair:
    """Represents a pair of entities across languages with their idioms."""
    entity_en: str
    entity_zh: str
    translation_direction: str  # "en_to_zh" or "zh_to_en"
    idioms_en: List[IdiomMatch]
    idioms_zh: List[IdiomMatch]
    cultural_analysis: Optional[str] = None


def load_idioms_data(file_path: str) -> List[Dict[str, Any]]:
    """Load idioms data from JSONL file."""
    idioms = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("output") is not None:
                idioms.append(obj)
    return idioms


def get_entity_counter(idioms_data: List[Dict[str, Any]]) -> Counter:
    """Get entity frequency counter from idioms data."""
    entity_counter = Counter()
    for item in idioms_data:
        entities = item.get("output", {}).get("entities", [])
        for e in entities:
            entity_counter[e] += 1
    return entity_counter


def get_top_entities(entity_counter: Counter, top_k: int = 200) -> List[str]:
    """Get top-k most frequent entities."""
    return [entity for entity, _ in entity_counter.most_common(top_k)]


def build_entity_to_idioms_index(idioms_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build index from entity to list of idioms containing that entity."""
    index = defaultdict(list)
    for item in idioms_data:
        output = item.get("output", {})
        entities = output.get("entities", [])
        for entity in entities:
            index[entity].append(item)
    return index


async def translate_entities_with_llm(
    entities: List[str],
    source_lang: str,
    target_lang: str,
    chat_model: ChatModel,
    batch_size: int = 10
) -> Dict[str, List[str]]:
    """
    Translate entities using LLM with high recall (multiple translations).

    Returns a dict mapping source entity to list of possible translations.
    """
    translations = {}

    lang_names = {"en": "English", "zh": "Chinese"}
    source_name = lang_names[source_lang]
    target_name = lang_names[target_lang]

    prompt_template = """You are a linguistic expert specializing in translating ENTITIES (nouns, concrete objects, animals, natural phenomena, body parts, etc.) that appear in idioms.

Given the {source_name} entity "{entity}", provide ALL possible {target_name} entity translations.
Focus on high recall - include:
1. Direct entity translations
2. Synonymous entities
3. Related entities that could substitute in different contexts
4. Different forms (e.g., singular/plural, different characters for same concept)

IMPORTANT CONSTRAINTS:
- All translations must be ENTITIES (nouns/concrete concepts), not verbs or adjectives
- For English to Chinese: Each translation should be 1-2 Chinese characters only (since Chinese idioms are typically 4 characters, entities within them are short). For example: "dragon" -> ["龙"], "water" -> ["水"], "river" -> ["河", "江"], "heart" -> ["心"]
- For Chinese to English: Include different English nouns that could represent the concept. For example: "龙" -> ["dragon"], "水" -> ["water"], "人" -> ["person", "people", "human", "guy"]

Return ONLY a JSON array of entity translations, nothing else. Example: ["entity1", "entity2", "entity3"]
If the entity has no good translation, return the original in an array: ["{entity}"]"""

    all_messages = []
    for entity in entities:
        messages = [
            {"role": "user", "content": prompt_template.format(
                source_name=source_name,
                target_name=target_name,
                entity=entity
            )}
        ]
        all_messages.append((entity, messages))

    # Process in batches
    for i in tqdm(range(0, len(all_messages), batch_size), desc=f"Translating {source_lang} -> {target_lang}"):
        batch = all_messages[i:i+batch_size]
        results = await chat_model.batch_generate_with_indices(batch)

        for entity, response, error in results:
            if error or response is None:
                print(f"Error translating '{entity}': {error}")
                translations[entity] = [entity]
                continue

            try:
                # Parse JSON response
                # Try to extract JSON array from response
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```\w*\n?', '', response)
                    response = re.sub(r'\n?```$', '', response)

                trans_list = json.loads(response)
                if isinstance(trans_list, list) and len(trans_list) > 0:
                    translations[entity] = trans_list
                else:
                    translations[entity] = [entity]
            except json.JSONDecodeError:
                # Try to extract from malformed response
                match = re.search(r'\[([^\]]+)\]', response)
                if match:
                    try:
                        trans_list = json.loads(f"[{match.group(1)}]")
                        translations[entity] = trans_list
                    except:
                        translations[entity] = [entity]
                else:
                    translations[entity] = [entity]

    return translations


def find_idioms_with_entity(
    entity: str,
    entity_to_idioms: Dict[str, List[Dict[str, Any]]],
    lang: str,
    max_idioms: int = 20
) -> List[IdiomMatch]:
    """Find idioms containing the given entity."""
    matches = []
    idioms = entity_to_idioms.get(entity, [])

    for item in idioms[:max_idioms]:
        output = item.get("output", {})
        match = IdiomMatch(
            idiom=output.get("idiom", item.get("idiom", "")),
            entity=entity,
            literal_meanings=output.get("literal_meanings", []),
            figurative_meanings=output.get("figurative_meanings", []),
            lang=lang
        )
        matches.append(match)

    return matches


def find_idioms_containing_translations(
    translations: List[str],
    entity_to_idioms: Dict[str, List[Dict[str, Any]]],
    idioms_data: List[Dict[str, Any]],
    lang: str,
    max_idioms: int = 20
) -> Tuple[List[IdiomMatch], str]:
    """
    Find idioms containing any of the translated entities.
    Also does substring matching for better recall.
    Returns matches and the matched entity.
    """
    all_matches = []
    matched_entity = None

    # First try exact entity match
    for trans in translations:
        if trans in entity_to_idioms:
            matches = find_idioms_with_entity(trans, entity_to_idioms, lang, max_idioms)
            if matches:
                all_matches.extend(matches)
                if matched_entity is None:
                    matched_entity = trans

    # If no exact matches, try substring matching in idiom text
    if not all_matches:
        for trans in translations:
            for item in idioms_data:
                output = item.get("output", {})
                idiom_text = output.get("idiom", item.get("idiom", ""))
                if trans in idiom_text:
                    match = IdiomMatch(
                        idiom=idiom_text,
                        entity=trans,
                        literal_meanings=output.get("literal_meanings", []),
                        figurative_meanings=output.get("figurative_meanings", []),
                        lang=lang
                    )
                    all_matches.append(match)
                    if matched_entity is None:
                        matched_entity = trans
                    if len(all_matches) >= max_idioms:
                        break
            if len(all_matches) >= max_idioms:
                break

    # Deduplicate by idiom text
    seen = set()
    unique_matches = []
    for m in all_matches:
        if m.idiom not in seen:
            seen.add(m.idiom)
            unique_matches.append(m)

    return unique_matches[:max_idioms], matched_entity


async def analyze_cultural_differences(
    entity_pair: CrossLingualEntityPair,
    chat_model: ChatModel
) -> str:
    """
    Use LLM to analyze cultural/meaning differences between idioms
    in two languages that share the same entity.
    """
    # Format idioms for analysis
    en_idioms_text = "\n".join([
        f"- {m.idiom}: {'; '.join(m.figurative_meanings)}"
        for m in entity_pair.idioms_en[:10]
    ])

    zh_idioms_text = "\n".join([
        f"- {m.idiom}: {'; '.join(m.figurative_meanings)}"
        for m in entity_pair.idioms_zh[:10]
    ])

    prompt = f"""You are a cultural linguistics expert analyzing how the same entity/concept conveys different meanings across English and Chinese idioms.

Entity in English: "{entity_pair.entity_en}"
Entity in Chinese: "{entity_pair.entity_zh}"

English idioms containing this entity:
{en_idioms_text}

Chinese idioms containing this entity:
{zh_idioms_text}

Analyze the cultural and semantic differences:
1. What are the PRIMARY figurative meanings/connotations associated with this entity in English idioms?
2. What are the PRIMARY figurative meanings/connotations associated with this entity in Chinese idioms?
3. What cultural values, beliefs, or perspectives might explain these differences?
4. Are there any SHARED meanings across both languages?
5. What unique cultural aspects does each language capture that the other doesn't?

Provide a structured analysis in JSON format:
{{
    "entity": "{entity_pair.entity_en} / {entity_pair.entity_zh}",
    "english_primary_meanings": ["meaning1", "meaning2"],
    "chinese_primary_meanings": ["meaning1", "meaning2"],
    "shared_meanings": ["shared1", "shared2"],
    "english_unique_aspects": ["aspect1", "aspect2"],
    "chinese_unique_aspects": ["aspect1", "aspect2"],
    "cultural_explanation": "Brief explanation of why these differences exist",
    "summary": "One paragraph summary of the cross-cultural comparison"
}}"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await chat_model.async_generate(messages)
        return response
    except Exception as e:
        print(f"Error analyzing entity pair: {e}")
        return json.dumps({"error": str(e)})


async def run_analysis(
    en_idioms_file: str,
    zh_idioms_file: str,
    output_dir: str,
    top_k: int = 200,
    model: str = "gpt-5.2-chat",
    provider: str = "openai",
    max_idioms_per_entity: int = 20,
    batch_size: int = 10
):
    """Run the full cross-lingual entity analysis."""

    os.makedirs(output_dir, exist_ok=True)

    print("Loading idioms data...")
    en_idioms = load_idioms_data(en_idioms_file)
    zh_idioms = load_idioms_data(zh_idioms_file)
    print(f"Loaded {len(en_idioms)} English idioms, {len(zh_idioms)} Chinese idioms")

    # Get entity counters and top entities
    print(f"\nExtracting top-{top_k} entities...")
    en_counter = get_entity_counter(en_idioms)
    zh_counter = get_entity_counter(zh_idioms)

    top_en_entities = get_top_entities(en_counter, top_k)
    top_zh_entities = get_top_entities(zh_counter, top_k)

    print(f"Top English entities (sample): {top_en_entities[:10]}")
    print(f"Top Chinese entities (sample): {top_zh_entities[:10]}")

    # Save top entities
    with open(os.path.join(output_dir, "top_entities_en.json"), "w", encoding="utf8") as f:
        json.dump(top_en_entities, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "top_entities_zh.json"), "w", encoding="utf8") as f:
        json.dump(top_zh_entities, f, ensure_ascii=False, indent=2)

    # Build entity-to-idioms indices
    print("\nBuilding entity-to-idioms indices...")
    en_entity_to_idioms = build_entity_to_idioms_index(en_idioms)
    zh_entity_to_idioms = build_entity_to_idioms_index(zh_idioms)

    # Initialize LLM
    print(f"\nInitializing {model} for translation and analysis...")
    chat_model = ChatModel(model=model, provider=provider)

    # Translate entities
    print("\nTranslating English entities to Chinese...")
    en_to_zh_translations = await translate_entities_with_llm(
        top_en_entities, "en", "zh", chat_model, batch_size
    )

    print("\nTranslating Chinese entities to English...")
    zh_to_en_translations = await translate_entities_with_llm(
        top_zh_entities, "zh", "en", chat_model, batch_size
    )

    # Save translations
    with open(os.path.join(output_dir, "translations_en_to_zh.json"), "w", encoding="utf8") as f:
        json.dump(en_to_zh_translations, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "translations_zh_to_en.json"), "w", encoding="utf8") as f:
        json.dump(zh_to_en_translations, f, ensure_ascii=False, indent=2)

    # Find cross-lingual entity pairs
    print("\nFinding cross-lingual entity pairs...")
    entity_pairs = []

    # English -> Chinese direction
    for en_entity in tqdm(top_en_entities, desc="EN->ZH matching"):
        translations = en_to_zh_translations.get(en_entity, [en_entity])

        # Find idioms in English
        en_idiom_matches = find_idioms_with_entity(
            en_entity, en_entity_to_idioms, "en", max_idioms_per_entity
        )

        # Find idioms in Chinese using translations
        zh_idiom_matches, matched_zh = find_idioms_containing_translations(
            translations, zh_entity_to_idioms, zh_idioms, "zh", max_idioms_per_entity
        )

        if en_idiom_matches and zh_idiom_matches:
            pair = CrossLingualEntityPair(
                entity_en=en_entity,
                entity_zh=matched_zh or translations[0] if translations else en_entity,
                translation_direction="en_to_zh",
                idioms_en=en_idiom_matches,
                idioms_zh=zh_idiom_matches
            )
            entity_pairs.append(pair)

    # Chinese -> English direction
    for zh_entity in tqdm(top_zh_entities, desc="ZH->EN matching"):
        translations = zh_to_en_translations.get(zh_entity, [zh_entity])

        # Find idioms in Chinese
        zh_idiom_matches = find_idioms_with_entity(
            zh_entity, zh_entity_to_idioms, "zh", max_idioms_per_entity
        )

        # Find idioms in English using translations
        en_idiom_matches, matched_en = find_idioms_containing_translations(
            translations, en_entity_to_idioms, en_idioms, "en", max_idioms_per_entity
        )

        if zh_idiom_matches and en_idiom_matches:
            # Check if this pair already exists from EN->ZH
            existing = any(
                p.entity_zh == zh_entity or
                (p.entity_en == matched_en and p.entity_zh == zh_entity)
                for p in entity_pairs
            )
            if not existing:
                pair = CrossLingualEntityPair(
                    entity_en=matched_en or translations[0] if translations else zh_entity,
                    entity_zh=zh_entity,
                    translation_direction="zh_to_en",
                    idioms_en=en_idiom_matches,
                    idioms_zh=zh_idiom_matches
                )
                entity_pairs.append(pair)

    print(f"\nFound {len(entity_pairs)} cross-lingual entity pairs with idioms in both languages")

    # Save intermediate results
    pairs_data = []
    for pair in entity_pairs:
        pairs_data.append({
            "entity_en": pair.entity_en,
            "entity_zh": pair.entity_zh,
            "translation_direction": pair.translation_direction,
            "idioms_en": [asdict(m) for m in pair.idioms_en],
            "idioms_zh": [asdict(m) for m in pair.idioms_zh],
            "num_en_idioms": len(pair.idioms_en),
            "num_zh_idioms": len(pair.idioms_zh)
        })

    with open(os.path.join(output_dir, "entity_pairs.json"), "w", encoding="utf8") as f:
        json.dump(pairs_data, f, ensure_ascii=False, indent=2)

    # Analyze cultural differences for each pair
    print("\nAnalyzing cultural differences...")
    analyzed_pairs = []

    for i, pair in enumerate(tqdm(entity_pairs, desc="Cultural analysis")):
        try:
            analysis = await analyze_cultural_differences(pair, chat_model)
            pair.cultural_analysis = analysis
            analyzed_pairs.append(pair)

            # Save progress periodically
            if (i + 1) % 20 == 0:
                save_analyzed_pairs(analyzed_pairs, output_dir)
        except Exception as e:
            print(f"Error analyzing pair {pair.entity_en}/{pair.entity_zh}: {e}")
            pair.cultural_analysis = json.dumps({"error": str(e)})
            analyzed_pairs.append(pair)

    # Save final results
    save_analyzed_pairs(analyzed_pairs, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return analyzed_pairs


def save_analyzed_pairs(pairs: List[CrossLingualEntityPair], output_dir: str):
    """Save analyzed pairs to file."""
    results = []
    for pair in pairs:
        result = {
            "entity_en": pair.entity_en,
            "entity_zh": pair.entity_zh,
            "translation_direction": pair.translation_direction,
            "idioms_en": [asdict(m) for m in pair.idioms_en],
            "idioms_zh": [asdict(m) for m in pair.idioms_zh],
            "cultural_analysis": pair.cultural_analysis
        }
        results.append(result)

    with open(os.path.join(output_dir, "cultural_analysis_results.json"), "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also save as JSONL for easier processing
    with open(os.path.join(output_dir, "cultural_analysis_results.jsonl"), "w", encoding="utf8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-lingual entity analysis for idioms")
    parser.add_argument(
        "--en_idioms",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/en/idioms_merged_llm_formatted.jsonl",
        help="Path to English idioms JSONL file"
    )
    parser.add_argument(
        "--zh_idioms",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/zh/idioms_merged_llm_formatted.jsonl",
        help="Path to Chinese idioms JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaruil5/culture_pretrain/CultureInFigurativeLanguage/culture/data/idioms/cross_lingual_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Number of top entities to analyze per language"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-chat",
        help="LLM model to use for translation and analysis"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "togetherai", "bedrock"],
        help="LLM provider"
    )
    parser.add_argument(
        "--max_idioms",
        type=int,
        default=20,
        help="Maximum idioms to retrieve per entity"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for LLM API calls"
    )

    args = parser.parse_args()

    # Run async analysis
    asyncio.run(run_analysis(
        en_idioms_file=args.en_idioms,
        zh_idioms_file=args.zh_idioms,
        output_dir=args.output_dir,
        top_k=args.top_k,
        model=args.model,
        provider=args.provider,
        max_idioms_per_entity=args.max_idioms,
        batch_size=args.batch_size
    ))


if __name__ == "__main__":
    main()
