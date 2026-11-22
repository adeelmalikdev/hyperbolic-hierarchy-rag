"""
Dataset construction script for Nextcloud and Django documentation.
Parses versioned documentation into hierarchical passages,
identifies cross-version procedural conflicts (stale passages),
and generates support-style benchmark queries.
"""

import os
import re
import json
import random
import difflib
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONES_DIR = os.path.join(BASE_DIR, "data", "raw", "clones")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Products configuration
CORPORA = {
    "Nextcloud": {
        "versions": ["28", "29", "30"],
        "target_version": "30",
        "dirs": {
            "28": os.path.join(CLONES_DIR, "nextcloud-stable28"),
            "29": os.path.join(CLONES_DIR, "nextcloud-stable29"),
            "30": os.path.join(CLONES_DIR, "nextcloud-stable30"),
        },
        "subdirs": ["admin_manual", "user_manual"],
        "exts": [".rst", ".md"],
    },
    "Django": {
        "versions": ["4.2", "5.0", "5.1"],
        "target_version": "5.1",
        "dirs": {
            "4.2": os.path.join(CLONES_DIR, "django-4.2", "docs"),
            "5.0": os.path.join(CLONES_DIR, "django-5.0", "docs"),
            "5.1": os.path.join(CLONES_DIR, "django-5.1", "docs"),
        },
        "subdirs": [""],  # Use all docs/
        "exts": [".txt", ".rst", ".md"],
    }
}

RST_UNDERLINE_CHARS = set("=-~^\"'#*_+`<>")

def clean_heading(heading):
    """Clean heading of RST/Markdown formatting like backticks and tags."""
    if not heading:
        return ""
    h = heading.strip()
    h = re.sub(r"^[#\*\-=\s]+", "", h)
    h = re.sub(r"[#\*\-=\s]+$", "", h)
    h = h.strip("`'\"*")
    h = re.sub(r":[a-zA-Z0-9_\-]+:`([^`]+)`", r"\1", h)
    return h.strip()

def clean_rst_text(text):
    """Clean RST markup while keeping substantive content, code, commands, and numbers."""
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Drop sphinx directives headers like '.. note::', '.. code-block:: ...'
        if stripped.startswith("..") and not stripped.startswith(".. _"):
            # keep directive argument text if present
            m = re.match(r"^\.\.\s+[a-zA-Z0-9_\-]+::\s*(.*)$", stripped)
            if m and m.group(1):
                cleaned_lines.append(m.group(1))
            continue
        if stripped.startswith(".. _"):
            continue
        cleaned_lines.append(line)
    
    content = "\n".join(cleaned_lines)
    # Remove inline roles like :ref:`text <target>` -> text
    content = re.sub(r":[a-zA-Z0-9_\-]+:`([^`<]+)(?:<[^>]+>)?`", r"\1", content)
    # Remove general backticks
    content = re.sub(r"``([^`]+)``", r"\1", content)
    content = re.sub(r"`([^`]+)`", r"\1", content)
    # Remove bold / italic
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
    content = re.sub(r"\*([^*]+)\*", r"\1", content)
    # Remove sphinx link syntax
    content = re.sub(r"`([^`<]+)<[^>]+>`_", r"\1", content)
    content = re.sub(r"([a-zA-Z0-9_\-]+)_", r"\1", content)
    # Replace multiple whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()

def parse_rst_sections(file_path):
    """
    Parses an RST/TXT/MD file into a list of (heading, text) sections.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return []

    sections = []
    current_heading = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ").title()
    current_lines = []
    
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()
        
        # Check for overline + underline heading
        if i + 2 < n:
            l1 = lines[i].strip()
            l2 = lines[i+1].strip()
            l3 = lines[i+2].strip()
            if len(l1) >= 3 and len(l3) >= 3 and l1[0] in RST_UNDERLINE_CHARS and l3[0] in RST_UNDERLINE_CHARS:
                if len(set(l1)) == 1 and len(set(l3)) == 1 and l1[0] == l3[0] and l2:
                    if current_lines:
                        sections.append((clean_heading(current_heading), "\n".join(current_lines)))
                        current_lines = []
                    current_heading = clean_heading(l2)
                    i += 3
                    continue

        # Check for underline heading
        if i + 1 < n:
            l1 = lines[i].rstrip()
            l2 = lines[i+1].rstrip()
            if l1 and len(l2) >= 3 and len(set(l2)) == 1 and l2[0] in RST_UNDERLINE_CHARS:
                if abs(len(l1) - len(l2)) <= 6:
                    if current_lines:
                        sections.append((clean_heading(current_heading), "\n".join(current_lines)))
                        current_lines = []
                    current_heading = clean_heading(l1)
                    i += 2
                    continue

        # Check markdown heading (# Heading)
        if stripped.startswith("#"):
            m = re.match(r"^#{1,4}\s+(.+)$", stripped)
            if m:
                if current_lines:
                    sections.append((clean_heading(current_heading), "\n".join(current_lines)))
                    current_lines = []
                current_heading = clean_heading(m.group(1))
                i += 1
                continue

        current_lines.append(line)
        i += 1

    if current_lines:
        sections.append((clean_heading(current_heading), "\n".join(current_lines)))

    return sections

def chunk_section(heading, raw_text, min_words=80, max_words=300):
    """
    Splits text under a heading into passages of min_words to max_words words.
    """
    clean_text = clean_rst_text(raw_text)
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    passages = []
    current_paras = []
    current_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_count + para_words <= max_words:
            current_paras.append(para)
            current_count += para_words
        else:
            if current_count >= min_words:
                passages.append(" ".join(current_paras))
                current_paras = [para]
                current_count = para_words
            else:
                current_paras.append(para)
                combined = " ".join(current_paras)
                words = combined.split()
                if len(words) <= max_words:
                    passages.append(combined)
                    current_paras = []
                    current_count = 0
                else:
                    while len(words) > max_words:
                        slice_chunk = words[:max_words]
                        passages.append(" ".join(slice_chunk))
                        words = words[max_words:]
                    if words:
                        current_paras = [" ".join(words)]
                        current_count = len(words)
                    else:
                        current_paras = []
                        current_count = 0

    if current_paras:
        combined = " ".join(current_paras)
        c_count = len(combined.split())
        if c_count >= min_words:
            passages.append(combined)
        elif passages:
            prev_words = len(passages[-1].split())
            if prev_words + c_count <= max_words + 50:
                passages[-1] = passages[-1] + " " + combined
            elif c_count >= 50:
                passages.append(combined)
        elif c_count >= 50:
            passages.append(combined)

    return passages

def extract_product_passages(product_name, config):
    """
    Extracts all passages for all versions of a product.
    Returns: dict version -> dict of rel_path -> list of passages dicts
    """
    results = defaultdict(lambda: defaultdict(list))
    
    for version in config["versions"]:
        base_path = config["dirs"][version]
        for subdir in config["subdirs"]:
            search_path = os.path.join(base_path, subdir) if subdir else base_path
            if not os.path.exists(search_path):
                continue
            
            for root, _, files in os.walk(search_path):
                for f in sorted(files):
                    if not any(f.endswith(ext) for ext in config["exts"]):
                        continue
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, base_path).replace("\\", "/")
                    
                    parts = rel_path.split("/")
                    if len(parts) > 1:
                        category = parts[0]
                        if subdir and len(parts) > 2:
                            category = f"{parts[0]}/{parts[1]}"
                    else:
                        category = "general"

                    article = parts[-1]
                    sections = parse_rst_sections(full_path)
                    
                    p_idx = 0
                    for heading, sec_text in sections:
                        chunks = chunk_section(heading, sec_text, min_words=80, max_words=300)
                        for chunk in chunks:
                            pid = f"{product_name.lower()}_{version}_{rel_path}_{p_idx}".replace("/", "_").replace(".", "_")
                            passage_obj = {
                                "passage_id": pid,
                                "product": product_name,
                                "version": version,
                                "category": category,
                                "article": article,
                                "rel_path": rel_path,
                                "heading": heading,
                                "chunk_idx": p_idx,
                                "text": chunk,
                                "word_count": len(chunk.split()),
                            }
                            results[version][rel_path].append(passage_obj)
                            p_idx += 1

    return results

def detect_procedural_change(text_new, text_old):
    """
    Determine if text_old is a stale/conflicting version of text_new.
    Returns (is_conflict, reason).
    """
    if text_new == text_old:
        return False, "identical"

    tokens_new = set(re.findall(r"\b[a-zA-Z0-9_\-\.:/]+\b", text_new.lower()))
    tokens_old = set(re.findall(r"\b[a-zA-Z0-9_\-\.:/]+\b", text_old.lower()))

    if not tokens_new or not tokens_old:
        return False, "empty"

    jaccard = len(tokens_new & tokens_old) / len(tokens_new | tokens_old)
    if jaccard < 0.25:
        return False, "too_dissimilar"

    nums_new = set(re.findall(r"\b\d+(?:\.\d+)*\b", text_new))
    nums_old = set(re.findall(r"\b\d+(?:\.\d+)*\b", text_old))
    num_diff = (nums_new != nums_old)

    code_new = set(re.findall(r"\b[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+|\.[a-zA-Z0-9_]+|--[a-zA-Z0-9_\-]+)\b", text_new))
    code_old = set(re.findall(r"\b[a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+|\.[a-zA-Z0-9_]+|--[a-zA-Z0-9_\-]+)\b", text_old))
    code_diff = (code_new != code_old)

    ratio = difflib.SequenceMatcher(None, text_new, text_old).quick_ratio()

    if (num_diff or code_diff or ratio < 0.90) and jaccard >= 0.30:
        return True, f"jaccard={jaccard:.2f},ratio={ratio:.2f},num_diff={num_diff},code_diff={code_diff}"

    return False, "minor_wording_only"

def generate_queries_for_product(product_name, new_passages, older_passages_by_ver, target_version, target_count=200):
    """
    Generates support-style queries from changed passages.
    Paraphrases with multiple distinct templates.
    Splits exact 50% version_stated=True, 50% version_stated=False.
    """
    candidates = []

    for rel_path, p_list in new_passages.items():
        for p in p_list:
            gold_id = p["passage_id"]
            heading = clean_heading(p["heading"])
            article = p["article"].replace(".rst", "").replace(".txt", "").replace(".md", "").replace("_", " ").title()
            category = p["category"].split("/")[-1].replace("_", " ").title()
            
            stale_ids = []
            for old_ver, old_files in older_passages_by_ver.items():
                if rel_path in old_files:
                    for old_p in old_files[rel_path]:
                        old_h = clean_heading(old_p["heading"])
                        if old_h.lower() == heading.lower() or abs(old_p["chunk_idx"] - p["chunk_idx"]) <= 1:
                            is_conflict, reason = detect_procedural_change(p["text"], old_p["text"])
                            if is_conflict:
                                stale_ids.append(old_p["passage_id"])

            if stale_ids:
                candidates.append({
                    "gold_passage": p,
                    "stale_passage_ids": sorted(list(set(stale_ids))),
                    "heading": heading,
                    "article": article,
                    "category": category,
                })

    random.shuffle(candidates)
    selected = candidates[:target_count]

    queries = []
    templates_unstated = [
        "How do I configure {topic} in {product}?",
        "What is the recommended setup for {topic}?",
        "Where can I find instructions to manage {topic}?",
        "How should I handle {topic} in {product}?",
        "What are the correct settings and requirements for {topic}?",
        "Can you explain the procedure to configure {topic}?",
    ]

    templates_stated = [
        "In {product} {version}, how do I configure {topic}?",
        "What is the recommended setup for {topic} in {product} {version}?",
        "Where can I find instructions to manage {topic} in {product} {version}?",
        "How should I handle {topic} in {product} {version}?",
        "What are the correct settings and requirements for {topic} in {product} {version}?",
        "In version {version} of {product}, what is the procedure for {topic}?",
    ]

    half = len(selected) // 2
    for i, item in enumerate(selected):
        p = item["gold_passage"]
        heading = item["heading"]
        article = item["article"]
        if len(heading) > 4 and heading.lower() not in ["overview", "introduction", "general"]:
            topic = heading
        else:
            topic = f"{article} {heading}".strip()

        version_stated = (i < half)
        tmpl_list = templates_stated if version_stated else templates_unstated
        template = tmpl_list[i % len(tmpl_list)]
        
        query_text = template.format(
            topic=topic,
            product=product_name,
            version=target_version
        )

        qid = f"q_{product_name.lower()}_{i:04d}"
        queries.append({
            "query_id": qid,
            "product": product_name,
            "target_version": target_version,
            "query_text": query_text,
            "gold_passage_id": p["passage_id"],
            "stale_passage_ids": item["stale_passage_ids"],
            "version_stated": version_stated,
            "heading": heading,
            "article": p["article"],
            "category": p["category"]
        })

    return queries

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    all_passages = []
    all_queries = []
    stats = {}

    for prod_name, config in CORPORA.items():
        print(f"Extracting passages for {prod_name}...")
        passages_by_ver = extract_product_passages(prod_name, config)
        
        target_ver = config["target_version"]
        older_vers = [v for v in config["versions"] if v != target_ver]
        
        prod_stats = {"passages_per_version": {}, "total_passages": 0}
        for v in config["versions"]:
            v_passages = [p for rel in passages_by_ver[v].values() for p in rel]
            prod_stats["passages_per_version"][v] = len(v_passages)
            prod_stats["total_passages"] += len(v_passages)
            all_passages.extend(v_passages)
            
        print(f"  {prod_name} passages: {prod_stats['passages_per_version']}")
        
        older_dict = {v: passages_by_ver[v] for v in older_vers}
        queries = generate_queries_for_product(
            prod_name,
            passages_by_ver[target_ver],
            older_dict,
            target_ver,
            target_count=200
        )
        print(f"  Generated {len(queries)} queries for {prod_name} (stated={sum(1 for q in queries if q['version_stated'])}, unstated={sum(1 for q in queries if not q['version_stated'])})")
        
        prod_stats["queries_count"] = len(queries)
        prod_stats["stated_queries"] = sum(1 for q in queries if q["version_stated"])
        prod_stats["unstated_queries"] = sum(1 for q in queries if not q["version_stated"])
        
        total_stale_links = sum(len(q["stale_passage_ids"]) for q in queries)
        prod_stats["total_stale_links"] = total_stale_links
        stats[prod_name] = prod_stats
        
        all_queries.extend(queries)

    passages_path = os.path.join(PROCESSED_DIR, "passages.jsonl")
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(all_passages)} passages to {passages_path}")

    queries_path = os.path.join(PROCESSED_DIR, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q) + "\n")
    print(f"Saved {len(all_queries)} queries to {queries_path}")

    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
