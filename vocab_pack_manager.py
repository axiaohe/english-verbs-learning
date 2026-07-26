"""
Vocabulary pack manager — file I/O, CSV parsing, validation, and AI pack generation.

Every vocabulary pack is a .json file in the packs/ directory. This module handles
reading, writing, validating and converting pack files, as well as CSV import/export
and AI-powered pack generation via the Gemini API.
"""

import csv
import io
import json
import os
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKS_DIR = os.path.join(os.path.dirname(__file__), "packs")
USER_IMPORT_SUBDIR = "user_imported"
GENERATED_PREFIX = "generated_"

VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}
REQUIRED_PACK_KEYS = {"pack_name", "display_name", "description", "category", "version", "verbs"}


def ensure_packs_dir() -> str:
    """Creates packs/ and packs/user_imported/ if they don't exist. Returns PACKS_DIR."""
    os.makedirs(os.path.join(PACKS_DIR, USER_IMPORT_SUBDIR), exist_ok=True)
    return PACKS_DIR


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pack_structure(data: dict) -> tuple[bool, list[str]]:
    """Validates a pack dict. Returns (is_valid, list_of_error_messages)."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return False, ["词包数据必须是 JSON 对象 (dictionary)。"]

    # Required top-level keys
    missing = REQUIRED_PACK_KEYS - data.keys()
    if missing:
        errors.append(f"缺少必填字段: {', '.join(sorted(missing))}。")

    verbs = data.get("verbs")
    if not isinstance(verbs, list) or len(verbs) == 0:
        errors.append("'verbs' 字段必须是一个非空的数组。")
        return False, errors

    for i, v in enumerate(verbs):
        if not isinstance(v, dict):
            errors.append(f"verbs[{i}]: 必须是对象，当前为 {type(v).__name__}。")
            continue
        if not v.get("verb") or not isinstance(v["verb"], str):
            errors.append(f"verbs[{i}]: 缺少 'verb' 字符串字段。")
        elif not re.fullmatch(r"[a-z]+(?:\s[a-z]+)*", v["verb"].strip().lower()):
            errors.append(f"verbs[{i}]: 'verb' ('{v['verb']}') 只能包含小写英文字母和空格（短语动词如 'give up' 是允许的）。")

        diff = v.get("difficulty", "")
        if diff not in VALID_DIFFICULTIES:
            errors.append(
                f"verbs[{i}]: 'difficulty' ('{diff}') 无效，必须是 {sorted(VALID_DIFFICULTIES)} 之一。"
            )

        if not v.get("definition") or not isinstance(v["definition"], str):
            errors.append(f"verbs[{i}]: 缺少 'definition' 字符串字段。")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Pack File I/O
# ---------------------------------------------------------------------------

def list_pack_files() -> list[str]:
    """Scans packs/ for *.json files and returns their filenames (sorted)."""
    ensure_packs_dir()
    files: list[str] = []
    for entry in os.listdir(PACKS_DIR):
        full = os.path.join(PACKS_DIR, entry)
        if os.path.isfile(full) and entry.endswith(".json"):
            files.append(entry)
    files.sort()
    return files


def load_pack_file(filename: str) -> dict | None:
    """Loads a pack JSON file from packs/. Returns None if not found or malformed."""
    filepath = os.path.join(PACKS_DIR, filename)
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_all_packs() -> list[dict]:
    """Loads every .json pack file in packs/ and returns valid pack dicts."""
    packs: list[dict] = []
    for filename in list_pack_files():
        data = load_pack_file(filename)
        if data:
            # Attach the filename for reference
            data["_filename"] = filename
            packs.append(data)
    return packs


def save_pack_to_file(data: dict, filename: str) -> str:
    """Saves a pack dict to packs/<filename>.json. Returns the full path."""
    ensure_packs_dir()
    filepath = os.path.join(PACKS_DIR, filename)
    # Strip internal keys before saving
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return filepath


def delete_pack_file(filename: str) -> bool:
    """Deletes a pack file from disk. Returns True if the file was removed."""
    filepath = os.path.join(PACKS_DIR, filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
        return True
    return False


# ---------------------------------------------------------------------------
# CSV Import / Export
# ---------------------------------------------------------------------------

def _sniff_encoding(raw: bytes) -> str:
    """Tries UTF-8, UTF-8-BOM, and GBK; returns the first that decodes without error."""
    candidates = ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"]
    for enc in candidates:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"  # last resort


def parse_csv_verb_file(uploaded_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Parses CSV verb data from raw bytes.

    Expected columns (case-insensitive): verb, difficulty, definition.

    Returns (verbs_list, error_messages).
    """
    encoding = _sniff_encoding(uploaded_bytes)
    text = uploaded_bytes.decode(encoding)

    # Strip BOM if present (handled by utf-8-sig, but belt-and-suspenders)
    if text and text[0] == "﻿":
        text = text[1:]

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return [], ["CSV 文件为空或格式不正确。"]

    # Normalize column names
    col_map: dict[str, str] = {}
    for col in reader.fieldnames:
        key = col.strip().lower()
        if key in ("verb", "difficulty", "definition"):
            col_map[col] = key

    if "verb" not in col_map.values():
        return [], [f"CSV 缺少 'verb' 列。找到的列: {reader.fieldnames}"]
    if "definition" not in col_map.values():
        return [], [f"CSV 缺少 'definition' 列。找到的列: {reader.fieldnames}"]

    verbs: list[dict] = []
    errors: list[str] = []

    for i, row in enumerate(reader, start=2):  # start=2 because row 1 is header
        verb_name = (row.get(col_map.get("verb", ""), "") or "").strip().lower()
        if not verb_name:
            errors.append(f"第 {i} 行: 动词名称为空，已跳过。")
            continue
        if not re.fullmatch(r"[a-z]+(?:\s[a-z]+)*", verb_name):
            errors.append(f"第 {i} 行: 动词 '{verb_name}' 只能包含字母和空格，已跳过。")
            continue

        difficulty = (row.get(col_map.get("difficulty", ""), "") or "").strip().upper()
        if difficulty not in VALID_DIFFICULTIES:
            difficulty = "B1"  # sensible default for unmarked words

        definition = (row.get(col_map.get("definition", ""), "") or "").strip()
        if not definition:
            errors.append(f"第 {i} 行: '{verb_name}' 缺少释义，已跳过。")
            continue

        verbs.append({"verb": verb_name, "difficulty": difficulty, "definition": definition})

    return verbs, errors


def parse_xlsx_verb_file_from_path(filepath: str) -> tuple[list[dict], list[str]]:
    """
    Convenience wrapper — reads an .xlsx file from disk and parses it.
    Returns (verbs_list, error_messages).
    """
    with open(filepath, "rb") as f:
        return parse_xlsx_verb_file(f.read())


def parse_xlsx_verb_file(uploaded_bytes: bytes) -> tuple[list[dict], list[str]]:
    """
    Parses vocabulary data from .xlsx bytes via pandas/openpyxl.

    Auto-detects column names (case-insensitive):
      - 'Words' / 'Word' / 'verb' → verb
      - 'Notes' / 'Note' / 'definition' / 'meaning' → definition
      - 'difficulty' / 'level' / 'CEFR' → difficulty

    If difficulty is not found, defaults to 'B1'.
    If definition is not found, defaults to '' (definition-less verbs still work
    because the AI generates questions based on the verb itself).

    Returns (verbs_list, error_messages).
    """
    try:
        import pandas as pd
    except ImportError:
        return [], ["需要安装 pandas 和 openpyxl 来处理 .xlsx 文件。请运行: pip install pandas openpyxl"]

    try:
        df = pd.read_excel(io.BytesIO(uploaded_bytes), dtype=str)
    except Exception as e:
        return [], [f"无法解析 .xlsx 文件: {e}"]

    if df.empty:
        return [], [".xlsx 文件为空或没有数据行。"]

    # Normalize column names: lowercase + strip
    col_map: dict[str, str] = {}
    for col in df.columns:
        if col is None:
            continue
        key = str(col).strip().lower()
        # Map known column patterns
        if key in ("words", "word", "verb", "vocabulary", "english", "term"):
            col_map[col] = "verb"
        elif key in ("notes", "note", "definition", "meaning", "chinese", "translation", "def"):
            col_map[col] = "definition"
        elif key in ("difficulty", "level", "cefr", "grade", "rank"):
            col_map[col] = "difficulty"

    verbs: list[dict] = []
    errors: list[str] = []

    for i, (_, row) in enumerate(df.iterrows(), start=2):
        verb_name = None
        definition = ""
        difficulty = "B1"

        for orig_col, mapped in col_map.items():
            val = row.get(orig_col, "")
            if pd.isna(val):
                val = ""
            val = str(val).strip()
            if mapped == "verb" and val:
                verb_name = val.lower()
            elif mapped == "definition" and val:
                definition = val
            elif mapped == "difficulty" and val:
                val_upper = val.upper()
                difficulty = val_upper if val_upper in VALID_DIFFICULTIES else "B1"

        if not verb_name:
            errors.append(f"第 {i} 行: 动词名称为空，已跳过。")
            continue

        # For xlsx imports we are more lenient than CSV — the user's vocabulary
        # list may contain phrasal verbs, collocations, and alternative forms
        # separated by "/".  We split on "/" to produce one verb per alternative.

        # First, strip trailing parenthetical context e.g. "toss (the pan)" → "toss"
        verb_name = re.sub(r"\s*\([^)]*\)\s*", " ", verb_name).strip()
        # Also strip trailing dots e.g. "prevent sth. from doing sth" → "prevent sth from doing sth"
        verb_name = verb_name.replace(".", "")

        # Split on "/" to handle alternatives like "forge / temper"
        alternatives = re.split(r"\s*/\s*", verb_name)

        for alt in alternatives:
            alt = alt.strip()
            if not alt:
                continue

            # Skip fragments that are clearly not standalone verbs (function
            # words produced by splitting grammar-pattern strings like
            # "strive for / to do / towards").
            if alt.lower() in {
                "to", "to do", "towards", "toward", "from", "for", "with", "without",
                "against", "into", "onto", "upon", "about", "over", "under", "away",
            }:
                continue

            # Allow lowercase letters, single spaces, and common grammar-placeholder
            # tokens ("sth", "sb") that appear in learner vocabulary lists.
            if not re.fullmatch(r"[a-z]+(?:\s(?:[a-z]+|sth|sb))*", alt):
                errors.append(
                    f"第 {i} 行: 动词 '{alt}' 包含不支持的字符，已跳过。"
                )
                continue

            verbs.append({
                "verb": alt,
                "difficulty": difficulty,
                "definition": definition if definition else f"（待补充释义）",
            })

    return verbs, errors


def verbs_to_csv_string(verbs: list[dict], include_progress: bool = False) -> str:
    """Converts a list of verb dicts to a CSV string for download."""
    output = io.StringIO()
    fieldnames = ["verb", "difficulty", "definition"]
    if include_progress:
        fieldnames.extend(["attempts", "correct_attempts", "mastery_score", "starred"])

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(verbs)
    return output.getvalue()


def verbs_to_pack_json(
    verbs: list[dict],
    pack_name: str = "",
    display_name: str = "",
    description: str = "",
    category: str = "custom",
) -> str:
    """Converts a verb list + metadata to a pack-format JSON string."""
    pack = {
        "pack_name": pack_name or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "display_name": display_name or "导出的词汇包",
        "description": description or "从个人单词库导出的词汇集合。",
        "category": category,
        "language": "en-zh",
        "version": "1.0",
        "author": "user",
        "verbs": verbs,
    }
    return json.dumps(pack, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# AI Pack Generation (Gemini Structured Output)
# ---------------------------------------------------------------------------

class AIVerbItem(BaseModel):
    verb: str = Field(description="Base form of the English verb (lowercase)")
    difficulty: str = Field(description="CEFR level: A1, A2, B1, B2, C1, or C2")
    definition: str = Field(
        description="Clear Chinese definition with usage context, e.g. '谈判，协商(合同条款)'"
    )


class AIGeneratedPack(BaseModel):
    display_name: str = Field(description="Concise, attractive Chinese name for this pack")
    description: str = Field(
        description="1-2 sentences in Chinese explaining what this pack covers and who it's for"
    )
    verbs: list[AIVerbItem] = Field(description="The list of verbs in this pack")


def build_ai_generation_prompt(topic: str, count: int, difficulty_prefs: list[str] | None = None) -> str:
    """Builds the prompt for Gemini to generate a vocabulary pack."""
    diff_desc = ""
    if difficulty_prefs:
        diff_desc = (
            f"Preferred difficulty levels: {', '.join(difficulty_prefs)}. "
            f"Try to keep most verbs within these levels, but you may include a few outside if they are essential."
        )
    else:
        diff_desc = "Distribute verbs across A1-C2 levels with emphasis on B1-B2 (intermediate)."

    return f"""
You are an expert ESL curriculum designer helping a Chinese-speaking English learner build a vocabulary pack.

Topic: "{topic}"
Number of verbs requested: {count}
{diff_desc}

Requirements:
1. Select English verbs that are genuinely useful and high-frequency for this topic.
2. For each verb, provide:
   - verb: base form in lowercase (e.g., "negotiate", not "Negotiate" or "negotiating")
   - difficulty: appropriate CEFR level (A1, A2, B1, B2, C1, C2)
   - definition: clear Chinese definition with common usage notes (e.g., "谈判，协商(合同条款)")
3. Include a mix of difficulties appropriate to the topic — don't make everything C1.
4. Prioritize practical, everyday utility. Avoid obscure or rarely-used verbs.
5. Do NOT include verbs that are already extremely basic (like "be", "have", "do", "go", "come", "eat", "see") unless they have a specialized meaning in this topic context.
6. The display_name should be a concise, attractive Chinese name for this pack (max 15 characters).
7. The description should be 1-2 sentences in Chinese explaining what the pack covers and who would benefit from it.
"""


def generate_pack_with_ai(
    client,  # GeminiClient instance
    topic: str,
    count: int = 80,
    difficulty_prefs: list[str] | None = None,
) -> dict | None:
    """
    Calls Gemini to generate a vocabulary pack for the given topic.

    Returns a pack dict on success, or None on failure.
    """
    if not client.is_configured():
        return None

    prompt = build_ai_generation_prompt(topic, count, difficulty_prefs)

    try:
        from google.genai import types

        response = client.client.models.generate_content(
            model=client.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AIGeneratedPack,
                temperature=0.9,
            ),
        )

        if response.parsed:
            parsed = response.parsed
        elif response.text:
            parsed = AIGeneratedPack(**json.loads(response.text))
        else:
            return None

        # Build pack dict
        pack_name = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        verbs = [v.model_dump() for v in parsed.verbs]

        # Deduplicate within the generated list
        seen: set[str] = set()
        deduped: list[dict] = []
        for v in verbs:
            key = v["verb"].strip().lower()
            if key not in seen:
                seen.add(key)
                v["verb"] = key
                deduped.append(v)

        # Count difficulty distribution
        dist: dict[str, int] = {}
        for v in deduped:
            d = v.get("difficulty", "B1")
            dist[d] = dist.get(d, 0) + 1

        return {
            "pack_name": pack_name,
            "display_name": parsed.display_name,
            "description": parsed.description,
            "category": topic,
            "language": "en-zh",
            "version": "1.0",
            "author": "ai_generated",
            "difficulty_distribution": dist,
            "verbs": deduped,
        }

    except Exception as e:
        print(f"AI pack generation error: {e}")
        return None
