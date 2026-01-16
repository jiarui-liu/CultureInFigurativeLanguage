import re
import json
import argparse
from typing import Dict, List, Any

HEADING_RE = re.compile(r'^(=+)\s*(.*?)\s*\1$')

def parse_wikitext_hierarchy(wikitext: str) -> Dict[str, Any]:
    root = {
        "_content": [],
        "_children": {}
    }

    stack = [(0, root)]  # (level, node)

    for line in wikitext.splitlines():
        m = HEADING_RE.match(line)

        if m:
            level = len(m.group(1))
            title = line.strip()

            # pop until parent level < current level
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent = stack[-1][1]

            node = {
                "_content": [],
                "_children": {}
            }

            # Handle duplicate section names by merging content
            if title in parent["_children"]:
                # Append a separator and merge content with existing section
                existing_node = parent["_children"][title]
                existing_node["_content"].append("")  # Add blank line separator
                # Continue using the existing node, content will be appended to it
                stack.append((level, existing_node))
            else:
                parent["_children"][title] = node
                stack.append((level, node))
        else:
            stack[-1][1]["_content"].append(line)

    return collapse_tree(root)

def collapse_tree(node: Dict[str, Any]) -> Dict[str, Any]:
    result = {}

    if node["_content"]:
        content = "\n".join(node["_content"]).strip()
        if content:
            result["_content"] = content

    for k, v in node["_children"].items():
        collapsed = collapse_tree(v)
        # If we already have this key, merge the content
        if k in result:
            # Merge _content if both have it
            if "_content" in collapsed and "_content" in result[k]:
                # Combine content with newline separator
                result[k]["_content"] = result[k]["_content"] + "\n\n" + collapsed["_content"]
            elif "_content" in collapsed:
                result[k]["_content"] = collapsed["_content"]
            # Merge other keys (child sections)
            for sub_k, sub_v in collapsed.items():
                if sub_k != "_content":
                    if sub_k in result[k]:
                        # Recursively merge nested sections with same name
                        if isinstance(result[k][sub_k], dict) and isinstance(sub_v, dict):
                            # Merge dictionaries
                            merged = {**result[k][sub_k]}
                            for merge_k, merge_v in sub_v.items():
                                if merge_k in merged:
                                    if isinstance(merged[merge_k], list) and isinstance(merge_v, list):
                                        merged[merge_k] = merged[merge_k] + merge_v
                                    elif isinstance(merged[merge_k], dict) and isinstance(merge_v, dict):
                                        merged[merge_k] = {**merged[merge_k], **merge_v}
                                    else:
                                        merged[merge_k] = merge_v
                                else:
                                    merged[merge_k] = merge_v
                            result[k][sub_k] = merged
                        else:
                            result[k][sub_k] = sub_v
                    else:
                        result[k][sub_k] = sub_v
        else:
            result[k] = collapsed

    return result

def process_jsonl_line(line: str) -> Dict[str, Any]:
    """Process a single JSONL line and return the parsed result."""
    obj = json.loads(line.strip())
    idiom = obj.get("idiom")
    wikitext = obj.get("wikitext", "")
    parsed = parse_wikitext_hierarchy(wikitext)
    return {idiom: parsed}


# Regex patterns for parsing _content
SENSE_LINE_RE = re.compile(r'^#\s+(.*)$', re.MULTILINE)
LABEL_RE = re.compile(r'\{\{(?:lb|label)\|[^}]*\}\}')
TEMPLATE_RE = re.compile(r'\{\{[^}]*\}\}')

# Regex for cleaning wiki links from explanation text
# [[link#section|display]] -> display
# [[link|display]] -> display  
# [[simple]] -> simple
WIKILINK_PIPED_RE = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')

# Regex for extracting {{l|en|...}} patterns (used in "See also" sections)
# Captures target (group 1) and optional display (group 2)
L_EN_TEMPLATE_RE = re.compile(r'\{\{l\|en\|([^}|]+)(?:\|([^}]+))?\}\}')

# Regex for extracting {{syn|en|...}} patterns (synonyms)
# Matches {{syn|en|word1|word2|...}} and extracts all words after en|
SYN_TEMPLATE_RE = re.compile(r'\{\{syn\|en\|([^}]+)\}\}')

# Regex for extracting {{syn of|en|...}} patterns
SYN_OF_TEMPLATE_RE = re.compile(r'\{\{syn of\|en\|([^}|]+)(?:\|[^}]*)?\}\}')

# Regex for extracting {{synonym of|en|...}} patterns
SYNONYM_OF_TEMPLATE_RE = re.compile(r'\{\{synonym of\|en\|([^}|]+)(?:\|[^}]*)?\}\}')

# Regex for extracting {{altform|en|...}} patterns
ALTFORM_TEMPLATE_RE = re.compile(r'\{\{altform\|en\|([^}|]+)(?:\|[^}]*)?\}\}')

# Regex for extracting {{alternative form of|en|...}} patterns
ALTERNATIVE_FORM_OF_TEMPLATE_RE = re.compile(r'\{\{alternative form of\|en\|([^}|]+)(?:\|[^}]*)?\}\}')

# Regex for extracting examples from {{ux|en|...}} templates
UX_TEMPLATE_RE = re.compile(r'\{\{ux\|en\|([^}]+)\}\}')

# Regex for extracting explanation from {{syn of|en|term||explanation}} format
# Captures both term and explanation
SYN_OF_WITH_EXPLANATION_RE = re.compile(r'\{\{syn of\|en\|([^|]+)\|\|([^}]+)\}\}')

# Regex for extracting explanation from {{synonym of|en|term||explanation}} format
SYNONYM_OF_WITH_EXPLANATION_RE = re.compile(r'\{\{synonym of\|en\|([^|]+)\|\|([^}]+)\}\}')

# Regex for extracting explanation from {{altform|en|term||explanation}} format
ALTFORM_WITH_EXPLANATION_RE = re.compile(r'\{\{altform\|en\|([^|]+)\|\|([^}]+)\}\}')


def clean_term_for_pattern(term: str) -> str:
    """
    Clean a term by removing metadata like <id:...> that shouldn't be in patterns.
    E.g., "out<id:having depleted stocks>" -> "out"
    """
    # Remove <id:...> patterns
    term = re.sub(r'<id:[^>]+>', '', term)
    # Remove any other angle bracket metadata like <...>
    term = re.sub(r'<[^>]+>', '', term)
    return term.strip()


def extract_l_en_items(text: str) -> List[str]:
    """
    Extract items from {{l|en|...}} templates.
    For pattern extraction, we want the target (first part), not the display.
    E.g., "{{l|en|at last}}" -> ["at last"]
    E.g., "{{l|en|all over the map|All over the map}}" -> ["all over the map"]
    """
    items = []
    for match in L_EN_TEMPLATE_RE.finditer(text):
        # Extract the target (first part), which is what we want for patterns
        target = match.group(1)
        if target:
            items.append(target)
    return items


def extract_synonyms(text: str) -> List[str]:
    """
    Extract synonyms from {{syn|en|...}}, {{syn of|en|...}}, {{synonym of|en|...}}, 
    and {{altform|en|...}} templates.
    E.g., "{{syn|en|word1|word2}}" -> ["word1", "word2"]
    E.g., "{{syn of|en|term}}" -> ["term"]
    E.g., "{{synonym of|en|term}}" -> ["term"]
    E.g., "{{altform|en|term}}" -> ["term"]
    """
    synonyms = []
    
    # Extract from {{syn|en|word1|word2|...}}
    for match in SYN_TEMPLATE_RE.finditer(text):
        # Split by | to get individual words
        words = match.group(1).split('|')
        for word in words:
            word = word.strip()
            # Skip empty strings and template parameters (starting with special chars)
            if word and not word.startswith(('t=', 'id=', 'alt=', 'tr=', 'pos=')):
                synonyms.append(word)
    
    # Extract from {{syn of|en|term}}
    synonyms.extend(SYN_OF_TEMPLATE_RE.findall(text))
    
    # Extract from {{synonym of|en|term}}
    synonyms.extend(SYNONYM_OF_TEMPLATE_RE.findall(text))
    
    # Extract from {{altform|en|term}}
    synonyms.extend(ALTFORM_TEMPLATE_RE.findall(text))
    
    # Extract from {{alternative form of|en|term}}
    synonyms.extend(ALTERNATIVE_FORM_OF_TEMPLATE_RE.findall(text))
    
    return synonyms


def clean_explanation_text(text: str) -> str:
    """
    Clean wiki markup from explanation text.
    - [[rundown#Noun|rundown]] -> rundown
    - [[basic#Noun|basic]] -> basic
    - [[introduction]] -> introduction
    - {{l|en|target|display}} -> display (if display present), otherwise target
    - {{l|en|all over the map|All over the map}} -> All over the map
    - {{m|en|word}} -> word
    - {{non-gloss|xxx}} -> xxx
    - {{gloss|xxx}} -> xxx
    - {{n-g|xxx}} -> xxx
    - {{ng|xxx}} -> xxx
    - {{ngd|xxx}} -> xxx
    - {{w|xxxx}} -> xxxx (Wikipedia links)
    - {{alternative spelling of|en|term}} -> alternative spelling of term
    - Remove remaining {{...}} templates
    
    Note: {{l|en|...}} templates are NOT extracted as patterns - they're only
    used for explanations and links to other entries.
    """
    # Handle piped wiki links: [[target|display]] or [[target#section|display]] -> display
    # Also handles simple links: [[word]] -> word
    text = WIKILINK_PIPED_RE.sub(r'\1', text)
    
    # Handle {{l|en|target|display}} templates - use display text if present, otherwise target
    # e.g., {{l|en|together|Together}} -> Together (display)
    # e.g., {{l|en|all over the map|All over the map}} -> All over the map (display)
    # e.g., {{l|en|word}} -> word (target, no display)
    def replace_l_en_template(match):
        target = match.group(1)
        display = match.group(2) if match.lastindex >= 2 and match.group(2) else None
        return display if display else target
    text = L_EN_TEMPLATE_RE.sub(replace_l_en_template, text)
    
    # Handle {{m|en|...}} templates (mentions) - extract the last part if present, otherwise first part
    # e.g., {{m|en|cost|Cost}} -> Cost, {{m|en|cost}} -> cost
    # Match {{m|en|word}} or {{m|en|word|Display}}
    def replace_m_en_template(match):
        full_match = match.group(0)
        # Extract everything between {{m|en| and }}
        content = full_match[7:-2]  # Remove {{m|en| and }}
        parts = content.split('|')
        if len(parts) > 1:
            # Has display form: return the last part
            return parts[-1]
        else:
            # No display form: return the word
            return parts[0]
    text = re.sub(r'\{\{m\|en\|[^}]+\}\}', replace_m_en_template, text)
    
    # Handle {{non-gloss|...}} templates - extract the content after non-gloss|
    text = re.sub(r'\{\{non-gloss\|([^}|]+)(?:\|[^}]*)?\}\}', r'\1', text)
    
    # Handle {{gloss|...}} templates - extract the content after gloss|
    text = re.sub(r'\{\{gloss\|([^}|]+)(?:\|[^}]*)?\}\}', r'\1', text)
    
    # Handle {{n-g|...}} templates (abbreviation for non-gloss)
    text = re.sub(r'\{\{n-g\|([^}|]+)(?:\|[^}]*)?\}\}', r'\1', text)
    
    # Handle {{ng|...}} templates (another abbreviation for non-gloss)
    # Need to handle content that may contain } characters, so match up to closing }}
    text = re.sub(r'\{\{ng\|([^}]+)\}\}', r'\1', text)
    
    # Handle {{ngd|...}} templates (another abbreviation for non-gloss)
    text = re.sub(r'\{\{ngd\|([^}|]+)(?:\|[^}]*)?\}\}', r'\1', text)
    
    # Handle simple templates like {{,}}, {{!}}, etc. - extract the content (or replace with appropriate text)
    # {{,}} is a comma template, replace with comma
    text = re.sub(r'\{\{,\}\}', ',', text)
    # Handle other simple templates {{xxx}} where xxx doesn't contain |
    text = re.sub(r'\{\{([^|{}]+)\}\}', r'\1', text)
    
    # Handle {{w|xxxx}} templates - extract the content after w|
    # e.g., {{w|Wikipedia}} -> Wikipedia
    # e.g., {{w|HM Treasury}} -> HM Treasury
    # Match everything after w| up to the closing }}
    text = re.sub(r'\{\{w\|([^}]+)\}\}', r'\1', text)
    
    # Handle {{taxlink|species|label}} templates - extract the species name (first part)
    # e.g., {{taxlink|Dicentra cucullaria|species}} -> Dicentra cucullaria
    text = re.sub(r'\{\{taxlink\|([^}|]+)(?:\|[^}]*)?\}\}', r'\1', text)
    
    # Handle {{xxx|en|yyy}} templates -> "xxx yyy"
    # e.g., {{alternative spelling of|en|the cat's pyjamas}} -> "alternative spelling of the cat's pyjamas"
    def replace_template_with_text(match):
        template_name = match.group(1).replace('_', ' ')
        term = match.group(2)
        return f"{template_name} {term}"
    text = re.sub(r'\{\{([^|{}]+)\|en\|([^}|]+)(?:\|[^}]*)?\}\}', replace_template_with_text, text)
    
    # Remove any remaining templates {{...}}
    text = TEMPLATE_RE.sub('', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_examples(text: str) -> List[str]:
    """
    Extract example sentences from {{ux|en|...}} templates.
    Returns a list of example strings.
    """
    examples = []
    for match in UX_TEMPLATE_RE.finditer(text):
        example = match.group(1).strip()
        if example:
            # Clean wiki markup from example
            example = clean_explanation_text(example)
            examples.append(example)
    return examples


def parse_content_to_entries(content: str, section_name: str = None) -> List[Dict[str, Any]]:
    """
    Parse _content into list of entries with metadata, explanation, other_content, section_name, examples.
    
    - metadata: template markup at start (e.g., {{en-noun}}) and labels (e.g., {{lb|en|informal}})
    - explanation: the core definition text (up to first period or semicolon)
    - other_content: everything after the explanation (examples, quotes, etc.)
    - section_name: the section header (e.g., "Verb", "Noun")
    - examples: list of example sentences from {{ux|en|...}} templates
    """
    if not content or not content.strip():
        return []
    
    # Remove HTML comments before processing
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    lines = content.split('\n')
    entries = []
    
    # Find header metadata (lines before first # definition line)
    header_lines = []
    definition_start_idx = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Only treat single # as definition start, not ## (subsenses)
        if stripped.startswith('#') and not stripped.startswith('##') and not stripped.startswith('#:') and not stripped.startswith('#*'):
            definition_start_idx = i
            break
        header_lines.append(line)
    else:
        # No definition lines found, return content as-is in metadata
        return [{"metadata": content.strip(), "explanation": "", "other_content": ""}]
    
    header_metadata = '\n'.join(header_lines).strip()
    
    # Process each sense (# lines that are definitions, not #: examples or #* quotes)
    current_sense_lines = []
    sense_groups = []
    
    for i in range(definition_start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()
        
        # Check if this is a new sense definition (# followed by space, not ##, #:, or #*)
        # ## lines are subsenses and should be part of the parent sense
        if stripped.startswith('#') and not stripped.startswith('##') and not stripped.startswith('#:') and not stripped.startswith('#*'):
            if current_sense_lines:
                sense_groups.append(current_sense_lines)
            current_sense_lines = [line]
        else:
            current_sense_lines.append(line)
    
    if current_sense_lines:
        sense_groups.append(current_sense_lines)
    
    # Parse each sense group
    for sense_lines in sense_groups:
        sense_text = '\n'.join(sense_lines)
        
        # Check if this sense has subsenses (## lines)
        has_subsenses = any(line.strip().startswith('##') and not line.strip().startswith('##:') and not line.strip().startswith('##*') for line in sense_lines)
        
        if has_subsenses:
            # Split into main sense and subsenses
            main_sense_lines = []
            subsense_groups = []
            current_subsense = []
            
            for line in sense_lines:
                stripped = line.strip()
                if stripped.startswith('##') and not stripped.startswith('##:') and not stripped.startswith('##*'):
                    if current_subsense:
                        subsense_groups.append(current_subsense)
                    current_subsense = [line]
                else:
                    if current_subsense:
                        current_subsense.append(line)
                    else:
                        main_sense_lines.append(line)
            
            if current_subsense:
                subsense_groups.append(current_subsense)
            
            # Parse main sense
            if main_sense_lines:
                main_sense_text = '\n'.join(main_sense_lines)
                main_entry = parse_single_sense(main_sense_text, header_metadata, section_name)
                if main_entry["explanation"]:
                    entries.append(main_entry)
            
            # Parse each subsense as a separate entry
            for subsense_lines in subsense_groups:
                subsense_text = '\n'.join(subsense_lines)
                subsense_entry = parse_single_sense(subsense_text, header_metadata, section_name)
                if subsense_entry["explanation"]:
                    entries.append(subsense_entry)
        else:
            # No subsenses, parse normally
            entry = parse_single_sense(sense_text, header_metadata, section_name)
            if entry["explanation"]:  # Only add if there's an explanation
                entries.append(entry)
    
    # If no valid entries but we have content, create one entry
    if not entries and content.strip():
        entries.append({"metadata": content.strip(), "explanation": "", "other_content": ""})
    
    return entries


def parse_single_sense(sense_text: str, header_metadata: str, section_name: str = None) -> Dict[str, Any]:
    """Parse a single sense definition into metadata, explanation, other_content, section_name, examples."""
    lines = sense_text.split('\n')
    
    if not lines:
        return {"metadata": header_metadata, "explanation": "", "other_content": ""}
    
    first_line = lines[0].strip()
    other_lines = lines[1:] if len(lines) > 1 else []
    
    # Remove leading # from definition line (but preserve ## for subsenses)
    if first_line.startswith('##'):
        first_line = first_line[2:].strip()
    elif first_line.startswith('#'):
        first_line = first_line[1:].strip()
    
    # Extract labels (metadata within the definition line)
    inline_labels = []
    remaining_text = first_line
    
    # Check for templates with explanations: {{syn of|en|term||explanation}}, 
    # {{synonym of|en|term||explanation}}, {{altform|en|term||explanation}}
    explanation_from_template = None
    template_type = None
    
    # Check {{syn of|en|term||explanation}}
    syn_of_explanation_match = SYN_OF_WITH_EXPLANATION_RE.search(first_line)
    if syn_of_explanation_match:
        term = syn_of_explanation_match.group(1).strip()
        explanation_from_template = syn_of_explanation_match.group(2).strip()
        # Clean term for pattern extraction (remove <id:...> etc.)
        cleaned_term = clean_term_for_pattern(term)
        simplified_template = f"{{{{syn of|en|{cleaned_term}}}}}"
        remaining_text = SYN_OF_WITH_EXPLANATION_RE.sub(simplified_template, remaining_text).strip()
        template_type = "syn_of"
    
    # Check {{synonym of|en|term||explanation}}
    if not explanation_from_template:
        synonym_of_explanation_match = SYNONYM_OF_WITH_EXPLANATION_RE.search(first_line)
        if synonym_of_explanation_match:
            term = synonym_of_explanation_match.group(1).strip()
            explanation_from_template = synonym_of_explanation_match.group(2).strip()
            # Clean term for pattern extraction (remove <id:...> etc.)
            cleaned_term = clean_term_for_pattern(term)
            simplified_template = f"{{{{synonym of|en|{cleaned_term}}}}}"
            remaining_text = SYNONYM_OF_WITH_EXPLANATION_RE.sub(simplified_template, remaining_text).strip()
            template_type = "synonym_of"
    
    # Check {{altform|en|term||explanation}}
    if not explanation_from_template:
        altform_explanation_match = ALTFORM_WITH_EXPLANATION_RE.search(first_line)
        if altform_explanation_match:
            term = altform_explanation_match.group(1).strip()
            explanation_from_template = altform_explanation_match.group(2).strip()
            # Clean term for pattern extraction (remove <id:...> etc.)
            cleaned_term = clean_term_for_pattern(term)
            simplified_template = f"{{{{altform|en|{cleaned_term}}}}}"
            remaining_text = ALTFORM_WITH_EXPLANATION_RE.sub(simplified_template, remaining_text).strip()
            template_type = "altform"
    
    # Find and extract {{lb|...}} or {{label|...}} patterns
    for match in LABEL_RE.finditer(remaining_text):
        inline_labels.append(match.group())
    remaining_text = LABEL_RE.sub('', remaining_text).strip()
    
    # Also extract other templates at the start (like {{&lit|...}})
    # But skip templates that contain definition content (non-gloss, gloss, l|en, w|, etc.)
    # Note: {{alternative form of|en|term}} should be extracted as metadata (not definition),
    # but the term will be extracted as a pattern from metadata
    definition_templates = ('{{non-gloss', '{{gloss', '{{n-g', '{{ng', '{{ngd', '{{l|en', '{{m|en', '{{w|')
    while remaining_text.startswith('{{'):
        # Don't extract templates that contain the actual definition
        if any(remaining_text.startswith(t) for t in definition_templates):
            break
        match = re.match(r'\{\{[^}]*\}\}', remaining_text)
        if match:
            inline_labels.append(match.group())
            remaining_text = remaining_text[match.end():].strip()
        else:
            break
    
    # Build metadata
    metadata_parts = []
    if header_metadata:
        metadata_parts.append(header_metadata)
    if inline_labels:
        metadata_parts.append(' '.join(inline_labels))
    metadata = '\n'.join(metadata_parts).strip()
    
    # Extract explanation (up to first period followed by space/newline)
    explanation = ""
    other_content_start = ""
    
    # If we found an explanation in a template, use it
    if explanation_from_template:
        explanation = explanation_from_template
        # Clean wiki markup from explanation
        explanation = clean_explanation_text(explanation)
    else:
        # First, extract content from definition templates like {{non-gloss|...}}, {{gloss|...}}, etc.
        # This ensures we get the full content even if it contains periods or nested templates
        text_to_parse = remaining_text
        explanation = None
        
        # Check for {{non-gloss|...}} template
        # Need to handle nested templates, so find the matching closing }}
        non_gloss_start = text_to_parse.find('{{non-gloss|')
        if non_gloss_start != -1:
            # Find the content start (after {{non-gloss|)
            content_start = non_gloss_start + len('{{non-gloss|')
            # Find the matching closing }} by counting braces
            # We need to handle nested templates correctly
            brace_count = 2  # Start with {{
            i = content_start
            while i < len(text_to_parse) and brace_count > 0:
                # Check if we have at least 2 characters remaining
                if i + 1 < len(text_to_parse):
                    if text_to_parse[i:i+2] == '{{':
                        brace_count += 2
                        i += 2
                    elif text_to_parse[i:i+2] == '}}':
                        brace_count -= 2
                        if brace_count == 0:
                            explanation = text_to_parse[content_start:i].strip()
                            # Remove the template from remaining text
                            text_to_parse = text_to_parse[:non_gloss_start] + text_to_parse[i+2:].strip()
                            other_content_start = text_to_parse
                            break
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
        
        if explanation is None:
            # Check for {{gloss|...}} template
            gloss_match = re.search(r'\{\{gloss\|([^}]+)\}\}', text_to_parse)
            if gloss_match:
                explanation = gloss_match.group(1).strip()
                # Remove the template from remaining text
                text_to_parse = re.sub(r'\{\{gloss\|[^}]+\}\}', '', text_to_parse).strip()
                other_content_start = text_to_parse
            else:
                # Check for {{n-g|...}}, {{ng|...}}, {{ngd|...}} templates
                n_g_match = re.search(r'\{\{n-g\|([^}]+)\}\}', text_to_parse)
                if n_g_match:
                    explanation = n_g_match.group(1).strip()
                    text_to_parse = re.sub(r'\{\{n-g\|[^}]+\}\}', '', text_to_parse).strip()
                    other_content_start = text_to_parse
                else:
                    ng_match = re.search(r'\{\{ng\|([^}]+)\}\}', text_to_parse)
                    if ng_match:
                        explanation = ng_match.group(1).strip()
                        text_to_parse = re.sub(r'\{\{ng\|[^}]+\}\}', '', text_to_parse).strip()
                        other_content_start = text_to_parse
                    else:
                        ngd_match = re.search(r'\{\{ngd\|([^}]+)\}\}', text_to_parse)
                        if ngd_match:
                            explanation = ngd_match.group(1).strip()
                            text_to_parse = re.sub(r'\{\{ngd\|[^}]+\}\}', '', text_to_parse).strip()
                            other_content_start = text_to_parse
                        else:
                            # No definition template found, use period-based extraction
                            # Find the end of explanation (first . followed by space, or end of text)
                            period_match = re.search(r'[.](?:\s|$)', text_to_parse)
                            if period_match:
                                explanation = text_to_parse[:period_match.start()].strip()
                                other_content_start = text_to_parse[period_match.end():].strip()
                            else:
                                explanation = text_to_parse.strip()
                                other_content_start = ""
        
        # Clean wiki markup from explanation
        explanation = clean_explanation_text(explanation)
    
    # Combine other_content
    other_parts = []
    if other_content_start:
        other_parts.append(other_content_start)
    if other_lines:
        other_parts.append('\n'.join(other_lines))
    other_content = '\n'.join(other_parts).strip()
    
    # Extract examples from other_content
    examples = extract_examples(other_content)
    
    return {
        "metadata": metadata,
        "explanation": explanation,
        "other_content": other_content,
        "section_name": section_name,
        "examples": examples
    }


def process_section(section: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """
    Recursively process a section, converting _content to structured entries.
    Returns flattened structure with lists of entries.
    """
    result = {}
    
    # Process _content if present
    if "_content" in section:
        content = section["_content"]
        entries = parse_content_to_entries(content, section_name)
        
        # For "See also" sections, extract {{l|en|...}} items into "reformatted" field
        if section_name == "See also" and entries:
            for entry in entries:
                # Extract from metadata (where the raw content is stored)
                metadata = entry.get("metadata", "")
                l_en_items = extract_l_en_items(metadata)
                if l_en_items:
                    entry["reformatted"] = l_en_items
        
        if entries:
            result["_entries"] = entries
    
    # Process child sections
    for key, value in section.items():
        if key == "_content":
            continue
        
        clean_key = key.strip("=").strip()
        
        if isinstance(value, dict):
            processed = process_section(value, clean_key)
            if processed:
                # If we already have this key, merge the entries
                if clean_key in result:
                    # Merge _entries if both have them
                    if "_entries" in processed and "_entries" in result[clean_key]:
                        # Combine entries from both sections
                        if isinstance(result[clean_key], dict) and isinstance(processed, dict):
                            result[clean_key]["_entries"].extend(processed["_entries"])
                            # Merge other keys
                            for k, v in processed.items():
                                if k != "_entries":
                                    if k in result[clean_key]:
                                        # If key exists, try to merge (for nested structures)
                                        if isinstance(result[clean_key][k], list) and isinstance(v, list):
                                            result[clean_key][k].extend(v)
                                        elif isinstance(result[clean_key][k], dict) and isinstance(v, dict):
                                            result[clean_key][k] = {**result[clean_key][k], **v}
                                        else:
                                            result[clean_key][k] = v
                                    else:
                                        result[clean_key][k] = v
                        elif isinstance(result[clean_key], list):
                            # If result is a list, convert to dict and merge
                            result[clean_key] = {"_entries": result[clean_key] + processed.get("_entries", [])}
                    elif "_entries" in processed:
                        # processed has entries, result doesn't - merge
                        if isinstance(result[clean_key], list):
                            result[clean_key] = {"_entries": result[clean_key] + processed["_entries"]}
                        elif isinstance(result[clean_key], dict):
                            result[clean_key]["_entries"] = processed["_entries"]
                            result[clean_key].update({k: v for k, v in processed.items() if k != "_entries"})
                    elif "_entries" in result[clean_key]:
                        # result has entries, processed doesn't - keep result's entries
                        if isinstance(processed, dict):
                            result[clean_key].update({k: v for k, v in processed.items() if k != "_entries"})
                else:
                    result[clean_key] = processed
        else:
            result[clean_key] = value
    
    return result


def flatten_to_list_format(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the structure so each section maps to a list of entries.
    Combines _entries with subsection entries.
    """
    result = {}
    
    for key, value in section.items():
        if key == "_entries":
            continue
        
        if isinstance(value, dict):
            # Check if this section has _entries
            if "_entries" in value:
                # This is a terminal section with entries
                entries = value["_entries"]
                # Also include any subsections
                sub_result = flatten_to_list_format({k: v for k, v in value.items() if k != "_entries"})
                if sub_result:
                    result[key] = {"entries": entries, **sub_result}
                else:
                    result[key] = entries
            else:
                # Recurse into subsections
                sub_result = flatten_to_list_format(value)
                if sub_result:
                    result[key] = sub_result
        else:
            result[key] = value
    
    # Handle top-level _entries
    if "_entries" in section:
        result["_entries"] = section["_entries"]
    
    return result


# Step 3: Extract definitions and patterns from flattened entries

def extract_definitions(entries: Any) -> List[Dict[str, Any]]:
    """
    Recursively extract all non-empty 'explanation' values from the entries structure.
    Returns a list of definition dicts with section_name, explanation, and examples.
    """
    definitions = []
    
    if isinstance(entries, list):
        # List of entry dicts
        for entry in entries:
            if isinstance(entry, dict):
                explanation = entry.get("explanation", "")
                if explanation and explanation.strip():
                    section_name = entry.get("section_name")
                    examples = entry.get("examples", [])
                    
                    # Format: {section_name: {"explanation": "...", "usage": null, "example": [...]}}
                    # Ensure examples is always a list
                    example_list = examples if isinstance(examples, list) else ([examples] if examples else [])
                    
                    if section_name:
                        # Lowercase the section name for consistency
                        section_key = section_name.lower()
                        def_dict = {
                            section_key: {
                                "explanation": explanation.strip(),
                                "usage": None,
                                "example": example_list
                            }
                        }
                        definitions.append(def_dict)
                    else:
                        # Fallback: if no section_name, use "unknown" as key to match other format
                        def_dict = {
                            "unknown": {
                                "explanation": explanation.strip(),
                                "usage": None,
                                "example": example_list
                            }
                        }
                        definitions.append(def_dict)
    elif isinstance(entries, dict):
        # Could be a nested structure with "entries" key or other subsections
        for key, value in entries.items():
            if key == "entries" and isinstance(value, list):
                # This is a list of entries
                definitions.extend(extract_definitions(value))
            elif isinstance(value, (list, dict)):
                # Recurse into subsections
                definitions.extend(extract_definitions(value))
    
    return definitions


def extract_patterns_from_entry(entry: Dict[str, str]) -> List[str]:
    """
    Extract patterns from a single entry dict by looking for synonyms
    in metadata and other_content fields.
    """
    patterns = []
    
    # Check for reformatted field (from "See also" sections)
    if "reformatted" in entry:
        patterns.extend(entry["reformatted"])
    
    # Extract synonyms from metadata
    metadata = entry.get("metadata", "")
    if metadata:
        patterns.extend(extract_synonyms(metadata))
    
    # Extract synonyms from other_content
    other_content = entry.get("other_content", "")
    if other_content:
        patterns.extend(extract_synonyms(other_content))
    
    return patterns


def extract_patterns(entries: Any) -> List[str]:
    """
    Extract patterns from 'See also' section, 'Synonyms' section, and
    {{syn|en|...}}, {{syn of|en|...}} templates throughout the entries.
    Returns a list of unique pattern strings.
    """
    patterns = []
    
    if isinstance(entries, list):
        # List of entry dicts
        for entry in entries:
            if isinstance(entry, dict):
                patterns.extend(extract_patterns_from_entry(entry))
    elif isinstance(entries, dict):
        # Check for "See also" section
        see_also = entries.get("See also")
        if see_also:
            if isinstance(see_also, list):
                # Direct list of entries
                for entry in see_also:
                    if isinstance(entry, dict):
                        patterns.extend(extract_patterns_from_entry(entry))
            elif isinstance(see_also, dict):
                # Nested structure with "entries" key
                if "entries" in see_also and isinstance(see_also["entries"], list):
                    for entry in see_also["entries"]:
                        if isinstance(entry, dict):
                            patterns.extend(extract_patterns_from_entry(entry))
        
        # Check for "Synonyms" section
        synonyms_section = entries.get("Synonyms")
        if synonyms_section:
            if isinstance(synonyms_section, list):
                for entry in synonyms_section:
                    if isinstance(entry, dict):
                        patterns.extend(extract_patterns_from_entry(entry))
            elif isinstance(synonyms_section, dict):
                if "entries" in synonyms_section and isinstance(synonyms_section["entries"], list):
                    for entry in synonyms_section["entries"]:
                        if isinstance(entry, dict):
                            patterns.extend(extract_patterns_from_entry(entry))
        
        # Recurse into other nested sections
        for key, value in entries.items():
            if key not in ("See also", "Synonyms") and isinstance(value, (dict, list)):
                patterns.extend(extract_patterns(value))
    
    return patterns


def finalize_output(idiom: str, flattened: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Convert flattened entries to final output format with
    definitions list and patterns list.
    """
    definitions = extract_definitions(flattened)
    patterns_raw = extract_patterns(flattened)
    
    # Deduplicate patterns while preserving order
    seen = set()
    patterns = []
    for p in patterns_raw:
        if p not in seen:
            seen.add(p)
            patterns.append(p)
    
    return {
        "idiom": idiom,
        "definition": definitions,
        "patterns": patterns
    }


def process_combined(input_path: str, output_path: str) -> None:
    """
    Combined processing: parse wikitext hierarchy, extract structured entries, and finalize output.
    Output format: {"idiom": "name", "definition": ["def1", "def2", ...], "patterns": ["pattern1", ...]}
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            # Step 1: Parse wikitext hierarchy
            parsed = process_jsonl_line(line)

            if not parsed:
                continue
                
            idiom = list(parsed.keys())[0]
            entry_content = parsed[idiom]
            
            if not isinstance(entry_content, dict):
                continue

            # Step 2: Get the English section
            english_section = entry_content.get("==English==")
            if not english_section:
                continue

            # Process the section recursively
            processed = process_section(english_section, "English")
            
            # Flatten to list format
            flattened = flatten_to_list_format(processed)

            # Step 3: Finalize output with definitions and patterns
            out = finalize_output(idiom, flattened)
            
            # Step 4: Filter out entries with no definitions
            # if not out["definition"]:
            #     continue

            outfile.write(json.dumps(out, ensure_ascii=False) + "\n")




def main():
    parser = argparse.ArgumentParser(
        description='Reformat Wiktionary English idioms JSONL file.'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input JSONL file containing idioms and wikitext.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to the output JSONL file for reformatted data.'
    )
    
    args = parser.parse_args()
    
    process_combined(args.input, args.output)
    print(f"Processing complete. Output written to: {args.output}")

if __name__ == "__main__":
    main()
