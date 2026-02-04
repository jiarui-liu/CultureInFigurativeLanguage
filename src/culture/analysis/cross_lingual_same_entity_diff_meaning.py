"""
Cross-lingual Entity Analysis for Idioms

This script analyzes how the same entities convey different meanings across
English and Chinese idioms. It:
1. Extracts top-k entities from each language
2. Translates entities to the other language with high recall
3. Uses embeddings to filter and expand entity clusters
4. Finds idioms containing these entities in both languages
5. Uses GPT-5.2 to analyze cultural/meaning differences
"""

import argparse
import json
import os
import re
import asyncio
import random
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import time
import numpy as np

from sentence_transformers import SentenceTransformer

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
class EntityCluster:
    """Represents a cluster of similar entities in one language."""
    primary_entity: str  # The original/seed entity
    cluster_entities: List[str]  # All entities in the cluster (including primary)
    lang: str


@dataclass
class CrossLingualEntityClusterPair:
    """Represents a pair of entity clusters across languages with their idioms."""
    en_cluster: EntityCluster
    zh_cluster: EntityCluster
    translation_direction: str  # "en_to_zh" or "zh_to_en"
    idioms_en: List[IdiomMatch]
    idioms_zh: List[IdiomMatch]
    cultural_analysis: Optional[str] = None


# Keep old dataclass for backward compatibility
@dataclass
class CrossLingualEntityPair:
    """Represents a pair of entities across languages with their idioms."""
    entity_en: str
    entity_zh: str
    translation_direction: str  # "en_to_zh" or "zh_to_en"
    idioms_en: List[IdiomMatch]
    idioms_zh: List[IdiomMatch]
    matched_translations: List[str] = None  # All translations that matched idioms
    cultural_analysis: Optional[str] = None


# ============================================================================
# Embedding-related functions
# ============================================================================

class EntityEmbeddingManager:
    """Manages entity embeddings for similarity computation."""

    def __init__(self, model_name: str = "/home/jiaruil5/math_rl/mix_teachers/r3lit_rl/models/Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)
        self._entity_embeddings_cache: Dict[str, np.ndarray] = {}

    def compute_embeddings(self, entities: List[str], show_progress: bool = False) -> np.ndarray:
        """Compute normalized embeddings for a list of entities."""
        embeddings = self.model.encode(
            entities,
            batch_size=64,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
        return embeddings

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(emb1, emb2))

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        # Embeddings are normalized, so dot product = cosine similarity
        return np.dot(embeddings, embeddings.T)

    def build_entity_embedding_index(
        self,
        entities: List[str],
        show_progress: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """Build embedding index for a list of entities."""
        print(f"Computing embeddings for {len(entities)} entities...")
        embeddings = self.compute_embeddings(entities, show_progress=show_progress)
        return entities, embeddings


async def filter_translations_by_semantic_equivalence(
    source_entity: str,
    candidate_translations: List[str],
    source_lang: str,
    target_lang: str,
    chat_model: ChatModel
) -> List[str]:
    """
    Use LLM to filter translations that are NOT semantically equivalent (synonyms) to the source entity.

    This addresses the limitation of embedding similarity, which may capture related but
    non-synonymous concepts (e.g., "dog" and "cat" are related but not synonyms).

    Args:
        source_entity: The original entity in the source language
        candidate_translations: List of candidate translations to filter
        source_lang: Source language code ("en" or "zh")
        target_lang: Target language code ("en" or "zh")
        chat_model: ChatModel instance for LLM calls

    Returns:
        List of translations that are true synonyms/semantic equivalents
    """
    if not candidate_translations:
        return []

    if len(candidate_translations) == 1:
        return candidate_translations

    lang_names = {"en": "English", "zh": "Chinese"}
    source_name = lang_names[source_lang]
    target_name = lang_names[target_lang]

    # Format candidates as a numbered list for clarity
    candidates_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidate_translations)])

    prompt = f"""You are a linguistic expert. Your task is to identify which candidate translations are TRUE TRANSLATIONS (synonyms or semantic equivalents) of the source entity.

Source entity ({source_name}): "{source_entity}"

Candidate translations ({target_name}) - numbered starting from 1:
{candidates_formatted}

A candidate is a TRUE TRANSLATION if:
- It refers to the EXACT SAME concept/object as the source entity
- It could replace the source entity in a translation without changing the meaning

REJECT candidates that are:
- Hypernyms (more general): "animal" is NOT a translation of "dog"
- Hyponyms (more specific): "puppy" is NOT a translation of "dog"
- Related but different: "cat", "wolf", "pet" are NOT translations of "dog"
- Associated concepts: "bone", "bark" are NOT translations of "dog"
- Part-whole relations: "leg" is NOT a translation of "body"

CROSS-LINGUAL EXAMPLES:
- "dog" (EN) → TRUE: "狗", "犬" | FALSE: "猫"(cat), "动物"(animal), "宠物"(pet)
- "water" (EN) → TRUE: "水" | FALSE: "河"(river), "雨"(rain), "海"(sea), "冰"(ice)
- "龙" (ZH) → TRUE: "dragon" | FALSE: "snake", "lizard", "monster", "creature"
- "月" (ZH) → TRUE: "moon" | FALSE: "sun", "night", "sky", "star"
- "heart" (EN) → TRUE: "心" | FALSE: "爱"(love), "情"(emotion), "胸"(chest)

OUTPUT FORMAT:
Return a JSON array of the 1-indexed numbers of TRUE TRANSLATIONS only.
- If candidates 1 and 3 are true translations: [1, 3]
- If only candidate 2 is a true translation: [2]
- If NONE are true translations: []

Your response (JSON array only):"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await chat_model.async_generate(messages)
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            response = re.sub(r'^```\w*\n?', '', response)
            response = re.sub(r'\n?```$', '', response)

        # Parse JSON response
        indices = json.loads(response)

        if isinstance(indices, list):
            # Convert 1-indexed to 0-indexed and filter valid indices
            filtered = []
            for idx in indices:
                if isinstance(idx, int) and 1 <= idx <= len(candidate_translations):
                    filtered.append(candidate_translations[idx - 1])
            return filtered if filtered else []
        else:
            return []

    except (json.JSONDecodeError, Exception) as e:
        print(f"Error in semantic filtering for '{source_entity}': {e}")
        # On error, return all candidates (fail open to preserve recall)
        return candidate_translations


async def batch_filter_translations_by_semantic_equivalence(
    entity_to_candidates: Dict[str, List[str]],
    source_lang: str,
    target_lang: str,
    chat_model: ChatModel,
    batch_size: int = 10
) -> Dict[str, List[str]]:
    """
    Batch version of semantic equivalence filtering for multiple entities.

    Args:
        entity_to_candidates: Dict mapping source entities to their candidate translations
        source_lang: Source language code
        target_lang: Target language code
        chat_model: ChatModel instance
        batch_size: Number of entities to process in parallel

    Returns:
        Dict mapping source entities to filtered (synonym-only) translations
    """
    filtered_results = {}
    entities = list(entity_to_candidates.keys())

    lang_names = {"en": "English", "zh": "Chinese"}
    source_name = lang_names[source_lang]
    target_name = lang_names[target_lang]

    # Build all messages
    all_messages = []
    for entity in entities:
        candidates = entity_to_candidates[entity]
        if not candidates or len(candidates) <= 1:
            # Skip filtering for empty or single-candidate lists
            filtered_results[entity] = candidates
            continue

        candidates_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        prompt = f"""You are a linguistic expert. Your task is to identify which candidate translations are TRUE TRANSLATIONS (synonyms or semantic equivalents) of the source entity.

Source entity ({source_name}): "{entity}"

Candidate translations ({target_name}) - numbered starting from 1:
{candidates_formatted}

A candidate is a TRUE TRANSLATION if:
- It refers to the EXACT SAME concept/object as the source entity
- It could replace the source entity in a translation without changing the meaning

REJECT candidates that are:
- Hypernyms (more general): "animal" is NOT a translation of "dog"
- Hyponyms (more specific): "puppy" is NOT a translation of "dog"
- Related but different: "cat", "wolf", "pet" are NOT translations of "dog"
- Associated concepts: "bone", "bark" are NOT translations of "dog"
- Part-whole relations: "leg" is NOT a translation of "body"

CROSS-LINGUAL EXAMPLES:
- "dog" (EN) → TRUE: "狗", "犬" | FALSE: "猫"(cat), "动物"(animal), "宠物"(pet)
- "water" (EN) → TRUE: "水" | FALSE: "河"(river), "雨"(rain), "海"(sea), "冰"(ice)
- "龙" (ZH) → TRUE: "dragon" | FALSE: "snake", "lizard", "monster", "creature"
- "月" (ZH) → TRUE: "moon" | FALSE: "sun", "night", "sky", "star"
- "heart" (EN) → TRUE: "心" | FALSE: "爱"(love), "情"(emotion), "胸"(chest)

OUTPUT FORMAT:
Return a JSON array of the 1-indexed numbers of TRUE TRANSLATIONS only.
- If candidates 1 and 3 are true translations: [1, 3]
- If only candidate 2 is a true translation: [2]
- If NONE are true translations: []

Your response (JSON array only):"""

        messages = [{"role": "user", "content": prompt}]
        all_messages.append((entity, messages, candidates))

    # Process in batches
    for i in tqdm(range(0, len(all_messages), batch_size), desc=f"Semantic filtering {source_lang} -> {target_lang}"):
        batch = all_messages[i:i+batch_size]
        batch_for_api = [(entity, messages) for entity, messages, _ in batch]

        results = await chat_model.batch_generate_with_indices(batch_for_api)

        for (entity, _, candidates), (_, response, error) in zip(batch, results):
            if error or response is None:
                print(f"Error in semantic filtering for '{entity}': {error}")
                filtered_results[entity] = candidates  # Fail open
                continue

            try:
                response = response.strip()
                if response.startswith("```"):
                    response = re.sub(r'^```\w*\n?', '', response)
                    response = re.sub(r'\n?```$', '', response)

                indices = json.loads(response)

                if isinstance(indices, list):
                    filtered = []
                    for idx in indices:
                        if isinstance(idx, int) and 1 <= idx <= len(candidates):
                            filtered.append(candidates[idx - 1])
                    filtered_results[entity] = filtered if filtered else []
                else:
                    filtered_results[entity] = []

            except (json.JSONDecodeError, Exception) as e:
                print(f"Error parsing response for '{entity}': {e}")
                filtered_results[entity] = candidates  # Fail open

    return filtered_results


def expand_entity_set_with_similar(
    seed_entities: List[str],
    all_entities: List[str],
    all_embeddings: np.ndarray,
    embedding_manager: EntityEmbeddingManager,
    similarity_threshold: float = 0.7
) -> List[str]:
    """
    Expand entity set by finding similar entities from a dictionary.

    Args:
        seed_entities: Initial entities to expand from
        all_entities: List of all entities in the dictionary
        all_embeddings: Precomputed embeddings for all_entities
        embedding_manager: EntityEmbeddingManager instance
        similarity_threshold: Minimum similarity to include (default 0.7)

    Returns:
        Expanded list of entities (including original seeds)
    """
    if not seed_entities:
        return []

    # Compute embeddings for seed entities
    seed_embeddings = embedding_manager.compute_embeddings(seed_entities)

    # Find similar entities from all_entities
    expanded_set = set(seed_entities)

    for seed_emb in seed_embeddings:
        # Compute similarity with all entities
        similarities = np.dot(all_embeddings, seed_emb)

        # Find entities above threshold
        for idx, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                expanded_set.add(all_entities[idx])

    return list(expanded_set)


def build_entity_cluster(
    primary_entity: str,
    entity_list: List[str],
    entity_embeddings: np.ndarray,
    embedding_manager: EntityEmbeddingManager,
    similarity_threshold: float = 0.7,
    lang: str = "en"
) -> EntityCluster:
    """
    Build an entity cluster by finding similar entities.

    Args:
        primary_entity: The seed entity
        entity_list: List of all entities in the language
        entity_embeddings: Precomputed embeddings for entity_list
        embedding_manager: EntityEmbeddingManager instance
        similarity_threshold: Minimum similarity to include
        lang: Language code

    Returns:
        EntityCluster containing all similar entities
    """
    # Find entities similar to primary_entity
    expanded = expand_entity_set_with_similar(
        seed_entities=[primary_entity],
        all_entities=entity_list,
        all_embeddings=entity_embeddings,
        embedding_manager=embedding_manager,
        similarity_threshold=similarity_threshold
    )

    return EntityCluster(
        primary_entity=primary_entity,
        cluster_entities=expanded,
        lang=lang
    )


# ============================================================================
# Data loading functions
# ============================================================================

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

    prompt_template = """You are a linguistic expert specializing in translating ENTITIES (nouns or noun phrases) that appear in idioms.

Given the {source_name} entity "{entity}", provide ALL possible {target_name} entity translations that could appear in {target_name} idioms.
Focus on high recall - include:
1. Direct entity translations
2. Synonymous entities
3. Related entities that could substitute in different contexts
4. Different forms (e.g., singular/plural, different characters for same concept)

Avoid translations that are too distant from the original entity.

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
    """Find idioms containing the given entity. Uses random sampling if more than max_idioms available."""
    matches = []
    idioms = entity_to_idioms.get(entity, [])

    # Random sample if we have more idioms than max_idioms
    if len(idioms) > max_idioms:
        sampled_idioms = random.sample(idioms, max_idioms)
    else:
        sampled_idioms = idioms

    for item in sampled_idioms:
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
) -> Tuple[List[IdiomMatch], List[str]]:
    """
    Find idioms containing any of the translated entities.
    Also does substring matching for better recall.
    Returns matches and ALL matched translations (not just the first one).
    """
    all_matches = []
    matched_translations = set()  # Track all translations that matched

    # First try exact entity match
    for trans in translations:
        if trans in entity_to_idioms:
            matches = find_idioms_with_entity(trans, entity_to_idioms, lang, max_idioms)
            if matches:
                all_matches.extend(matches)
                matched_translations.add(trans)

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
                    matched_translations.add(trans)
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

    # Random sample if we have more matches than max_idioms
    if len(unique_matches) > max_idioms:
        sampled_matches = random.sample(unique_matches, max_idioms)
    else:
        sampled_matches = unique_matches

    return sampled_matches, list(matched_translations)


def find_idioms_for_entity_cluster(
    cluster: EntityCluster,
    entity_to_idioms: Dict[str, List[Dict[str, Any]]],
    idioms_data: List[Dict[str, Any]],
    max_idioms: int = 20
) -> List[IdiomMatch]:
    """
    Find idioms containing any entity in the cluster.

    Args:
        cluster: EntityCluster containing entities to search for
        entity_to_idioms: Index mapping entity -> list of idioms
        idioms_data: All idioms data (for substring fallback)
        max_idioms: Maximum number of idioms to return

    Returns:
        List of IdiomMatch objects
    """
    all_matches = []
    matched_entities = set()

    # First try exact entity match for all entities in cluster
    for entity in cluster.cluster_entities:
        if entity in entity_to_idioms:
            idioms = entity_to_idioms.get(entity, [])
            for item in idioms:
                output = item.get("output", {})
                match = IdiomMatch(
                    idiom=output.get("idiom", item.get("idiom", "")),
                    entity=entity,
                    literal_meanings=output.get("literal_meanings", []),
                    figurative_meanings=output.get("figurative_meanings", []),
                    lang=cluster.lang
                )
                all_matches.append(match)
                matched_entities.add(entity)

    # If no exact matches, try substring matching
    if not all_matches:
        for entity in cluster.cluster_entities:
            for item in idioms_data:
                output = item.get("output", {})
                idiom_text = output.get("idiom", item.get("idiom", ""))
                if entity in idiom_text:
                    match = IdiomMatch(
                        idiom=idiom_text,
                        entity=entity,
                        literal_meanings=output.get("literal_meanings", []),
                        figurative_meanings=output.get("figurative_meanings", []),
                        lang=cluster.lang
                    )
                    all_matches.append(match)
                    matched_entities.add(entity)

    # Deduplicate by idiom text
    seen = set()
    unique_matches = []
    for m in all_matches:
        if m.idiom not in seen:
            seen.add(m.idiom)
            unique_matches.append(m)

    # Random sample if we have more matches than max_idioms
    if len(unique_matches) > max_idioms:
        sampled_matches = random.sample(unique_matches, max_idioms)
    else:
        sampled_matches = unique_matches

    return sampled_matches


def flatten_meanings(meanings) -> str:
    """Safely flatten and join meanings, handling nested lists and non-strings."""
    if not meanings:
        return ""

    def _flatten(item):
        """Recursively flatten nested structures."""
        if isinstance(item, str):
            return [item]
        elif isinstance(item, list):
            result = []
            for sub_item in item:
                result.extend(_flatten(sub_item))
            return result
        else:
            return [str(item)]

    flat_items = _flatten(meanings)
    return '; '.join(flat_items)


async def analyze_cultural_differences(
    entity_pair: CrossLingualEntityPair,
    chat_model: ChatModel
) -> str:
    """
    Use LLM to analyze cultural/meaning differences between idioms
    in two languages that share the same entity.
    """
    # Format idioms for analysis, including which entity matched
    en_idioms_text = "\n".join([
        f"- {m.idiom} (entity: {m.entity}): {flatten_meanings(m.figurative_meanings)}"
        for m in entity_pair.idioms_en
    ])

    zh_idioms_text = "\n".join([
        f"- {m.idiom} (entity: {m.entity}): {flatten_meanings(m.figurative_meanings)}"
        for m in entity_pair.idioms_zh
    ])

    # Format matched translations info
    matched_trans_info = ""
    if entity_pair.matched_translations:
        # Ensure all items are strings
        trans_strs = [str(t) for t in entity_pair.matched_translations if t is not None]
        if trans_strs:
            matched_trans_info = f"\nMatched translations used: {', '.join(trans_strs)}"

    prompt = f"""You are a cultural linguistics expert analyzing how the same entity/concept conveys different meanings across English and Chinese idioms.

Entity in English: "{entity_pair.entity_en}"
Entity in Chinese: "{entity_pair.entity_zh}"
Translation direction: {entity_pair.translation_direction}{matched_trans_info}

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
    "matched_translations": {json.dumps(entity_pair.matched_translations or [], ensure_ascii=False)},
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


async def analyze_cultural_differences_for_clusters(
    cluster_pair: CrossLingualEntityClusterPair,
    chat_model: ChatModel
) -> str:
    """
    Use LLM to analyze cultural/meaning differences between idioms
    for entity clusters (multiple related entities per language).
    """
    # Format entity clusters
    en_entities_str = ", ".join(cluster_pair.en_cluster.cluster_entities)
    zh_entities_str = ", ".join(cluster_pair.zh_cluster.cluster_entities)

    # Format idioms for analysis, including which entity matched
    en_idioms_text = "\n".join([
        f"- {m.idiom} (entity: {m.entity}): {flatten_meanings(m.figurative_meanings)}"
        for m in cluster_pair.idioms_en
    ])

    zh_idioms_text = "\n".join([
        f"- {m.idiom} (entity: {m.entity}): {flatten_meanings(m.figurative_meanings)}"
        for m in cluster_pair.idioms_zh
    ])

    prompt = f"""You are a cultural linguistics expert analyzing how similar entities/concepts convey different meanings across English and Chinese idioms.

English entity cluster (semantically similar entities): [{en_entities_str}]
Primary English entity: "{cluster_pair.en_cluster.primary_entity}"

Chinese entity cluster (semantically similar entities): [{zh_entities_str}]
Primary Chinese entity: "{cluster_pair.zh_cluster.primary_entity}"

Translation direction: {cluster_pair.translation_direction}

English idioms containing entities from this cluster:
{en_idioms_text}

Chinese idioms containing entities from this cluster:
{zh_idioms_text}

Analyze the cultural and semantic differences:
1. What are the PRIMARY figurative meanings/connotations associated with this entity cluster in English idioms?
2. What are the PRIMARY figurative meanings/connotations associated with this entity cluster in Chinese idioms?
3. What cultural values, beliefs, or perspectives might explain these differences?
4. Are there any SHARED meanings across both languages?
5. What unique cultural aspects does each language capture that the other doesn't?

Provide a structured analysis in JSON format:
{{
    "english_entity_cluster": {json.dumps(cluster_pair.en_cluster.cluster_entities, ensure_ascii=False)},
    "chinese_entity_cluster": {json.dumps(cluster_pair.zh_cluster.cluster_entities, ensure_ascii=False)},
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
        print(f"Error analyzing cluster pair: {e}")
        return json.dumps({"error": str(e)})


async def run_analysis_with_embeddings(
    en_idioms_file: str,
    zh_idioms_file: str,
    output_dir: str,
    top_k: int = 200,
    model: str = "gpt-5.2-chat",
    provider: str = "openai",
    max_idioms_per_entity: int = 20,
    batch_size: int = 10,
    run_stage1: bool = True,
    run_stage2: bool = True,
    embedding_model: str = "/home/jiaruil5/math_rl/mix_teachers/r3lit_rl/models/Qwen/Qwen3-Embedding-0.6B",
    entity_expand_threshold: float = 0.7,
    use_semantic_filter: bool = False
):
    """
    Run cross-lingual entity analysis with embedding-based filtering and expansion.

    Stage 1: Translation and entity cluster building
    Stage 2: Cultural analysis with LLM

    Translation filtering pipeline (for each source entity):

    WITHOUT semantic filter (use_semantic_filter=False):
    1. LLM generates multiple translation candidates
    2. Expand with similar entities from target dictionary (threshold: entity_expand_threshold)
    3. Build entity clusters and find matching idioms

    WITH semantic filter (use_semantic_filter=True) - RECOMMENDED:
    1. LLM generates multiple translation candidates
    2. Expand with similar entities from target dictionary (threshold: entity_expand_threshold)
    3. LLM semantic filter - removes non-synonyms
    4. Build entity clusters and find matching idioms
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Loading idioms data...")
    en_idioms = load_idioms_data(en_idioms_file)
    zh_idioms = load_idioms_data(zh_idioms_file)
    print(f"Loaded {len(en_idioms)} English idioms, {len(zh_idioms)} Chinese idioms")

    # Get entity counters and all entities
    print(f"\nExtracting entities...")
    en_counter = get_entity_counter(en_idioms)
    zh_counter = get_entity_counter(zh_idioms)

    top_en_entities = get_top_entities(en_counter, top_k)
    top_zh_entities = get_top_entities(zh_counter, top_k)

    # Get all unique entities for embedding computation
    all_en_entities = list(en_counter.keys())
    all_zh_entities = list(zh_counter.keys())

    print(f"Top-{top_k} English entities (sample): {top_en_entities[:10]}")
    print(f"Top-{top_k} Chinese entities (sample): {top_zh_entities[:10]}")
    print(f"Total unique entities: EN={len(all_en_entities)}, ZH={len(all_zh_entities)}")

    # Save top entities
    with open(os.path.join(output_dir, "top_entities_en.json"), "w", encoding="utf8") as f:
        json.dump(top_en_entities, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "top_entities_zh.json"), "w", encoding="utf8") as f:
        json.dump(top_zh_entities, f, ensure_ascii=False, indent=2)

    # Build entity-to-idioms indices
    print("\nBuilding entity-to-idioms indices...")
    en_entity_to_idioms = build_entity_to_idioms_index(en_idioms)
    zh_entity_to_idioms = build_entity_to_idioms_index(zh_idioms)

    # Initialize embedding manager
    print(f"\nInitializing embedding model: {embedding_model}")
    embedding_manager = EntityEmbeddingManager(model_name=embedding_model)

    # Compute embeddings for all entities
    print("\nComputing embeddings for all English entities...")
    all_en_embeddings = embedding_manager.compute_embeddings(all_en_entities, show_progress=True)

    print("\nComputing embeddings for all Chinese entities...")
    all_zh_embeddings = embedding_manager.compute_embeddings(all_zh_entities, show_progress=True)

    # Save embeddings index for reuse
    embeddings_cache = {
        "en_entities": all_en_entities,
        "zh_entities": all_zh_entities,
    }
    # Note: embeddings are numpy arrays, save separately if needed

    # Initialize LLM
    print(f"\nInitializing {model} for translation and analysis...")
    chat_model = ChatModel(model=model, provider=provider)

    # ========== STAGE 1: Translation ==========
    en_to_zh_path = os.path.join(output_dir, "translations_en_to_zh.json")
    zh_to_en_path = os.path.join(output_dir, "translations_zh_to_en.json")

    if run_stage1:
        print("\n" + "="*50)
        print("STAGE 1: Translation")
        print("="*50)

        print("\nTranslating English entities to Chinese...")
        en_to_zh_translations = await translate_entities_with_llm(
            top_en_entities, "en", "zh", chat_model, batch_size
        )

        print("\nTranslating Chinese entities to English...")
        zh_to_en_translations = await translate_entities_with_llm(
            top_zh_entities, "zh", "en", chat_model, batch_size
        )

        # Save translations
        with open(en_to_zh_path, "w", encoding="utf8") as f:
            json.dump(en_to_zh_translations, f, ensure_ascii=False, indent=2)
        with open(zh_to_en_path, "w", encoding="utf8") as f:
            json.dump(zh_to_en_translations, f, ensure_ascii=False, indent=2)

        print(f"\nStage 1 complete! Translations saved to {output_dir}")

        if not run_stage2:
            print("\nStopping after Stage 1 (translation only).")
            return None
    else:
        # Load existing translations for Stage 2
        print("\nLoading existing translations for Stage 2...")
        if os.path.exists(en_to_zh_path) and os.path.exists(zh_to_en_path):
            with open(en_to_zh_path, "r", encoding="utf8") as f:
                en_to_zh_translations = json.load(f)
            with open(zh_to_en_path, "r", encoding="utf8") as f:
                zh_to_en_translations = json.load(f)
            print(f"Loaded {len(en_to_zh_translations)} EN->ZH and {len(zh_to_en_translations)} ZH->EN translations")
        else:
            raise FileNotFoundError(f"Translation files not found in {output_dir}. Run Stage 1 first.")

    # ========== SEMANTIC FILTERING (Optional) ==========
    if use_semantic_filter:
        print("\n" + "="*50)
        print("SEMANTIC FILTERING: Filtering translations by semantic equivalence")
        print("="*50)

        # Skip mutual embedding similarity filter - semantic filter handles this more accurately
        # Pipeline: LLM translations -> expand with dictionary -> semantic filter
        en_to_zh_for_semantic = {}
        for entity, raw_trans in en_to_zh_translations.items():
            # Expand with similar entities from Chinese dictionary
            expanded = expand_entity_set_with_similar(
                seed_entities=raw_trans,
                all_entities=all_zh_entities,
                all_embeddings=all_zh_embeddings,
                embedding_manager=embedding_manager,
                similarity_threshold=entity_expand_threshold
            )
            if expanded:
                en_to_zh_for_semantic[entity] = expanded

        zh_to_en_for_semantic = {}
        for entity, raw_trans in zh_to_en_translations.items():
            # Expand with similar entities from English dictionary
            expanded = expand_entity_set_with_similar(
                seed_entities=raw_trans,
                all_entities=all_en_entities,
                all_embeddings=all_en_embeddings,
                embedding_manager=embedding_manager,
                similarity_threshold=entity_expand_threshold
            )
            if expanded:
                zh_to_en_for_semantic[entity] = expanded

        # Apply semantic filtering using LLM
        print(f"\nFiltering EN->ZH translations ({len(en_to_zh_for_semantic)} entities)...")
        en_to_zh_semantic_filtered = await batch_filter_translations_by_semantic_equivalence(
            en_to_zh_for_semantic, "en", "zh", chat_model, batch_size
        )

        print(f"\nFiltering ZH->EN translations ({len(zh_to_en_for_semantic)} entities)...")
        zh_to_en_semantic_filtered = await batch_filter_translations_by_semantic_equivalence(
            zh_to_en_for_semantic, "zh", "en", chat_model, batch_size
        )

        # Save semantic filtered translations
        semantic_filtered_path = os.path.join(output_dir, "translations_semantic_filtered.json")
        with open(semantic_filtered_path, "w", encoding="utf8") as f:
            json.dump({
                "en_to_zh": en_to_zh_semantic_filtered,
                "zh_to_en": zh_to_en_semantic_filtered
            }, f, ensure_ascii=False, indent=2)
        print(f"\nSemantic filtered translations saved to {semantic_filtered_path}")

        # Log filtering statistics
        en_to_zh_before = sum(len(v) for v in en_to_zh_for_semantic.values())
        en_to_zh_after = sum(len(v) for v in en_to_zh_semantic_filtered.values())
        zh_to_en_before = sum(len(v) for v in zh_to_en_for_semantic.values())
        zh_to_en_after = sum(len(v) for v in zh_to_en_semantic_filtered.values())
        print(f"EN->ZH: {en_to_zh_before} -> {en_to_zh_after} translations ({en_to_zh_before - en_to_zh_after} removed)")
        print(f"ZH->EN: {zh_to_en_before} -> {zh_to_en_after} translations ({zh_to_en_before - zh_to_en_after} removed)")

    # Track analyzed entities to avoid duplicates
    analyzed_en_entities: Set[str] = set()
    analyzed_zh_entities: Set[str] = set()

    # Build entity cluster pairs
    print("\nBuilding entity cluster pairs with embedding-based filtering...")
    cluster_pairs: List[CrossLingualEntityClusterPair] = []

    # English -> Chinese direction
    for en_entity in tqdm(top_en_entities, desc="EN->ZH cluster building"):
        # Skip if this entity was already analyzed
        if en_entity in analyzed_en_entities:
            continue

        # Build English entity cluster (expand with similar entities)
        en_cluster = build_entity_cluster(
            primary_entity=en_entity,
            entity_list=all_en_entities,
            entity_embeddings=all_en_embeddings,
            embedding_manager=embedding_manager,
            similarity_threshold=entity_expand_threshold,
            lang="en"
        )

        # Get target language entities (use semantic filtered if available)
        if use_semantic_filter:
            # Use pre-computed semantic filtered translations
            zh_cluster_entities = en_to_zh_semantic_filtered.get(en_entity, [])
            filtered_translations = zh_cluster_entities[:1] if zh_cluster_entities else []
        else:
            # Original flow: LLM translations -> expand with dictionary
            raw_translations = en_to_zh_translations.get(en_entity, [en_entity])
            filtered_translations = raw_translations

            # Expand with similar entities from Chinese dictionary
            zh_cluster_entities = expand_entity_set_with_similar(
                seed_entities=raw_translations,
                all_entities=all_zh_entities,
                all_embeddings=all_zh_embeddings,
                embedding_manager=embedding_manager,
                similarity_threshold=entity_expand_threshold
            )

        # Skip if no entities found in target language
        if not zh_cluster_entities:
            continue

        # Check if any Chinese entity was already analyzed
        already_analyzed_zh = any(e in analyzed_zh_entities for e in zh_cluster_entities)
        if already_analyzed_zh:
            continue

        # Create Chinese cluster
        zh_cluster = EntityCluster(
            primary_entity=filtered_translations[0] if filtered_translations else zh_cluster_entities[0],
            cluster_entities=zh_cluster_entities,
            lang="zh"
        )

        # Find idioms for each cluster
        en_idiom_matches = find_idioms_for_entity_cluster(
            en_cluster, en_entity_to_idioms, en_idioms, max_idioms_per_entity
        )
        zh_idiom_matches = find_idioms_for_entity_cluster(
            zh_cluster, zh_entity_to_idioms, zh_idioms, max_idioms_per_entity
        )

        # Only create pair if both have idioms
        if en_idiom_matches and zh_idiom_matches:
            cluster_pair = CrossLingualEntityClusterPair(
                en_cluster=en_cluster,
                zh_cluster=zh_cluster,
                translation_direction="en_to_zh",
                idioms_en=en_idiom_matches,
                idioms_zh=zh_idiom_matches
            )
            cluster_pairs.append(cluster_pair)

            # Mark entities as analyzed
            analyzed_en_entities.update(en_cluster.cluster_entities)
            analyzed_zh_entities.update(zh_cluster.cluster_entities)

    # Chinese -> English direction
    for zh_entity in tqdm(top_zh_entities, desc="ZH->EN cluster building"):
        # Skip if this entity was already analyzed
        if zh_entity in analyzed_zh_entities:
            continue

        # Build Chinese entity cluster
        zh_cluster = build_entity_cluster(
            primary_entity=zh_entity,
            entity_list=all_zh_entities,
            entity_embeddings=all_zh_embeddings,
            embedding_manager=embedding_manager,
            similarity_threshold=entity_expand_threshold,
            lang="zh"
        )

        # Get target language entities (use semantic filtered if available)
        if use_semantic_filter:
            # Use pre-computed semantic filtered translations
            en_cluster_entities = zh_to_en_semantic_filtered.get(zh_entity, [])
            filtered_translations = en_cluster_entities[:1] if en_cluster_entities else []
        else:
            # Original flow: LLM translations -> expand with dictionary
            raw_translations = zh_to_en_translations.get(zh_entity, [zh_entity])
            filtered_translations = raw_translations

            # Expand with similar entities from English dictionary
            en_cluster_entities = expand_entity_set_with_similar(
                seed_entities=raw_translations,
                all_entities=all_en_entities,
                all_embeddings=all_en_embeddings,
                embedding_manager=embedding_manager,
                similarity_threshold=entity_expand_threshold
            )

        # Skip if no entities found in target language
        if not en_cluster_entities:
            continue

        # Check if any English entity was already analyzed
        already_analyzed_en = any(e in analyzed_en_entities for e in en_cluster_entities)
        if already_analyzed_en:
            continue

        # Create English cluster
        en_cluster = EntityCluster(
            primary_entity=filtered_translations[0] if filtered_translations else en_cluster_entities[0],
            cluster_entities=en_cluster_entities,
            lang="en"
        )

        # Find idioms for each cluster
        en_idiom_matches = find_idioms_for_entity_cluster(
            en_cluster, en_entity_to_idioms, en_idioms, max_idioms_per_entity
        )
        zh_idiom_matches = find_idioms_for_entity_cluster(
            zh_cluster, zh_entity_to_idioms, zh_idioms, max_idioms_per_entity
        )

        # Only create pair if both have idioms
        if en_idiom_matches and zh_idiom_matches:
            cluster_pair = CrossLingualEntityClusterPair(
                en_cluster=en_cluster,
                zh_cluster=zh_cluster,
                translation_direction="zh_to_en",
                idioms_en=en_idiom_matches,
                idioms_zh=zh_idiom_matches
            )
            cluster_pairs.append(cluster_pair)

            # Mark entities as analyzed
            analyzed_en_entities.update(en_cluster.cluster_entities)
            analyzed_zh_entities.update(zh_cluster.cluster_entities)

    print(f"\nFound {len(cluster_pairs)} cross-lingual entity cluster pairs")

    # Save intermediate results
    pairs_data = []
    for pair in cluster_pairs:
        pairs_data.append({
            "en_cluster": {
                "primary_entity": pair.en_cluster.primary_entity,
                "cluster_entities": pair.en_cluster.cluster_entities
            },
            "zh_cluster": {
                "primary_entity": pair.zh_cluster.primary_entity,
                "cluster_entities": pair.zh_cluster.cluster_entities
            },
            "translation_direction": pair.translation_direction,
            "idioms_en": [asdict(m) for m in pair.idioms_en],
            "idioms_zh": [asdict(m) for m in pair.idioms_zh],
            "num_en_idioms": len(pair.idioms_en),
            "num_zh_idioms": len(pair.idioms_zh)
        })

    with open(os.path.join(output_dir, "entity_cluster_pairs.json"), "w", encoding="utf8") as f:
        json.dump(pairs_data, f, ensure_ascii=False, indent=2)

    # ========== STAGE 2: Cultural Analysis ==========
    if run_stage2:
        print("\n" + "="*50)
        print("STAGE 2: Cultural Analysis")
        print("="*50)

        analyzed_pairs = []

        for i, pair in enumerate(tqdm(cluster_pairs, desc="Cultural analysis")):
            try:
                analysis = await analyze_cultural_differences_for_clusters(pair, chat_model)
                pair.cultural_analysis = analysis
                analyzed_pairs.append(pair)

                # Save progress periodically
                if (i + 1) % 20 == 0:
                    save_analyzed_cluster_pairs(analyzed_pairs, output_dir)
            except Exception as e:
                print(f"Error analyzing cluster pair {pair.en_cluster.primary_entity}/{pair.zh_cluster.primary_entity}: {e}")
                pair.cultural_analysis = json.dumps({"error": str(e)})
                analyzed_pairs.append(pair)

        # Save final results
        save_analyzed_cluster_pairs(analyzed_pairs, output_dir)

        print(f"\nStage 2 complete! Analysis saved to {output_dir}")
        return analyzed_pairs
    else:
        print("\nStage 2 skipped. Entity cluster pairs saved without cultural analysis.")
        return cluster_pairs


def save_analyzed_cluster_pairs(pairs: List[CrossLingualEntityClusterPair], output_dir: str):
    """Save analyzed cluster pairs to file."""
    results = []
    for pair in pairs:
        result = {
            "en_cluster": {
                "primary_entity": pair.en_cluster.primary_entity,
                "cluster_entities": pair.en_cluster.cluster_entities
            },
            "zh_cluster": {
                "primary_entity": pair.zh_cluster.primary_entity,
                "cluster_entities": pair.zh_cluster.cluster_entities
            },
            "translation_direction": pair.translation_direction,
            "idioms_en": [asdict(m) for m in pair.idioms_en],
            "idioms_zh": [asdict(m) for m in pair.idioms_zh],
            "cultural_analysis": pair.cultural_analysis
        }
        results.append(result)

    with open(os.path.join(output_dir, "cultural_analysis_clusters.json"), "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also save as JSONL for easier processing
    with open(os.path.join(output_dir, "cultural_analysis_clusters.jsonl"), "w", encoding="utf8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def run_analysis(
    en_idioms_file: str,
    zh_idioms_file: str,
    output_dir: str,
    top_k: int = 200,
    model: str = "gpt-5.2-chat",
    provider: str = "openai",
    max_idioms_per_entity: int = 20,
    batch_size: int = 10,
    run_stage1: bool = True,
    run_stage2: bool = True
):
    """
    Run the full cross-lingual entity analysis (legacy version without embeddings).

    Stage 1: Translation
    Stage 2: Cultural analysis
    """

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

    # ========== STAGE 1: Translation ==========
    en_to_zh_path = os.path.join(output_dir, "translations_en_to_zh.json")
    zh_to_en_path = os.path.join(output_dir, "translations_zh_to_en.json")

    if run_stage1:
        print("\n" + "="*50)
        print("STAGE 1: Translation")
        print("="*50)

        print("\nTranslating English entities to Chinese...")
        en_to_zh_translations = await translate_entities_with_llm(
            top_en_entities, "en", "zh", chat_model, batch_size
        )

        print("\nTranslating Chinese entities to English...")
        zh_to_en_translations = await translate_entities_with_llm(
            top_zh_entities, "zh", "en", chat_model, batch_size
        )

        # Save translations
        with open(en_to_zh_path, "w", encoding="utf8") as f:
            json.dump(en_to_zh_translations, f, ensure_ascii=False, indent=2)
        with open(zh_to_en_path, "w", encoding="utf8") as f:
            json.dump(zh_to_en_translations, f, ensure_ascii=False, indent=2)

        print(f"\nStage 1 complete! Translations saved to {output_dir}")

        if not run_stage2:
            print("\nStopping after Stage 1 (translation only).")
            return None
    else:
        # Load existing translations for Stage 2
        print("\nLoading existing translations for Stage 2...")
        if os.path.exists(en_to_zh_path) and os.path.exists(zh_to_en_path):
            with open(en_to_zh_path, "r", encoding="utf8") as f:
                en_to_zh_translations = json.load(f)
            with open(zh_to_en_path, "r", encoding="utf8") as f:
                zh_to_en_translations = json.load(f)
            print(f"Loaded {len(en_to_zh_translations)} EN->ZH and {len(zh_to_en_translations)} ZH->EN translations")
        else:
            raise FileNotFoundError(f"Translation files not found in {output_dir}. Run Stage 1 first.")

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
        zh_idiom_matches, matched_zh_list = find_idioms_containing_translations(
            translations, zh_entity_to_idioms, zh_idioms, "zh", max_idioms_per_entity
        )

        if en_idiom_matches and zh_idiom_matches:
            pair = CrossLingualEntityPair(
                entity_en=en_entity,
                entity_zh=matched_zh_list[0] if matched_zh_list else (translations[0] if translations else en_entity),
                translation_direction="en_to_zh",
                idioms_en=en_idiom_matches,
                idioms_zh=zh_idiom_matches,
                matched_translations=matched_zh_list  # Store all matched translations
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
        en_idiom_matches, matched_en_list = find_idioms_containing_translations(
            translations, en_entity_to_idioms, en_idioms, "en", max_idioms_per_entity
        )

        if zh_idiom_matches and en_idiom_matches:
            # Check if this pair already exists from EN->ZH
            existing = any(
                p.entity_zh == zh_entity or
                (p.entity_en in matched_en_list and p.entity_zh == zh_entity)
                for p in entity_pairs
            )
            if not existing:
                pair = CrossLingualEntityPair(
                    entity_en=matched_en_list[0] if matched_en_list else (translations[0] if translations else zh_entity),
                    entity_zh=zh_entity,
                    translation_direction="zh_to_en",
                    idioms_en=en_idiom_matches,
                    idioms_zh=zh_idiom_matches,
                    matched_translations=matched_en_list  # Store all matched translations
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
            "matched_translations": pair.matched_translations,
            "idioms_en": [asdict(m) for m in pair.idioms_en],
            "idioms_zh": [asdict(m) for m in pair.idioms_zh],
            "num_en_idioms": len(pair.idioms_en),
            "num_zh_idioms": len(pair.idioms_zh)
        })

    with open(os.path.join(output_dir, "entity_pairs.json"), "w", encoding="utf8") as f:
        json.dump(pairs_data, f, ensure_ascii=False, indent=2)

    # ========== STAGE 2: Cultural Analysis ==========
    if run_stage2:
        print("\n" + "="*50)
        print("STAGE 2: Cultural Analysis")
        print("="*50)

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

        print(f"\nStage 2 complete! Analysis saved to {output_dir}")
        return analyzed_pairs
    else:
        print("\nStage 2 skipped. Entity pairs saved without cultural analysis.")
        return entity_pairs


def save_analyzed_pairs(pairs: List[CrossLingualEntityPair], output_dir: str):
    """Save analyzed pairs to file."""
    results = []
    for pair in pairs:
        result = {
            "entity_en": pair.entity_en,
            "entity_zh": pair.entity_zh,
            "translation_direction": pair.translation_direction,
            "matched_translations": pair.matched_translations,
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
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "all"],
        help="Which stage to run: '1' = translation only, '2' = analysis only (loads existing translations), 'all' = both stages (default)"
    )
    parser.add_argument(
        "--use_embeddings",
        action="store_true",
        help="Use embedding-based filtering and expansion for entity clusters"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="/home/jiaruil5/math_rl/mix_teachers/r3lit_rl/models/Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model for similarity computation"
    )
    parser.add_argument(
        "--entity_expand_threshold",
        type=float,
        default=0.7,
        help="Minimum similarity for entity cluster expansion (default: 0.7)"
    )
    parser.add_argument(
        "--use_semantic_filter",
        action="store_true",
        help="Use LLM-based semantic equivalence filtering to remove non-synonym translations (requires --use_embeddings)"
    )

    args = parser.parse_args()

    # Determine which stages to run
    run_stage1 = args.stage in ["1", "all"]
    run_stage2 = args.stage in ["2", "all"]

    # Run async analysis
    if args.use_embeddings:
        asyncio.run(run_analysis_with_embeddings(
            en_idioms_file=args.en_idioms,
            zh_idioms_file=args.zh_idioms,
            output_dir=args.output_dir,
            top_k=args.top_k,
            model=args.model,
            provider=args.provider,
            max_idioms_per_entity=args.max_idioms,
            batch_size=args.batch_size,
            run_stage1=run_stage1,
            run_stage2=run_stage2,
            embedding_model=args.embedding_model,
            entity_expand_threshold=args.entity_expand_threshold,
            use_semantic_filter=args.use_semantic_filter
        ))
    else:
        asyncio.run(run_analysis(
            en_idioms_file=args.en_idioms,
            zh_idioms_file=args.zh_idioms,
            output_dir=args.output_dir,
            top_k=args.top_k,
            model=args.model,
            provider=args.provider,
            max_idioms_per_entity=args.max_idioms,
            batch_size=args.batch_size,
            run_stage1=run_stage1,
            run_stage2=run_stage2
        ))


if __name__ == "__main__":
    main()
