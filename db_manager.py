import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from verbs_data import VERBS_SEED

import vocab_pack_manager

DB_PATH = os.path.join(os.path.dirname(__file__), "vocab_tracker.db")

# Columns shared by every "verb + progress" query. COALESCE guards against verbs
# that somehow lost their user_progress row, so callers always get integers.
VERB_COLUMNS = """
    v.verb, v.difficulty, v.definition, v.source_pack,
    COALESCE(p.attempts, 0) AS attempts,
    COALESCE(p.correct_attempts, 0) AS correct_attempts,
    COALESCE(p.mastery_score, 0) AS mastery_score,
    p.last_tested, p.next_test,
    COALESCE(p.starred, 0) AS starred
"""


@contextmanager
def get_connection():
    """Yields a SQLite connection with row factory enabled, always closed on exit."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _escape_like(text):
    """Escapes LIKE wildcards so a literal '%' or '_' in a search box matches itself."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# =========================================================================
# Database Initialization & Seeding
# =========================================================================

def _ensure_source_pack_column(cursor):
    """Adds source_pack column to verbs table if it doesn't exist (idempotent)."""
    try:
        cursor.execute("ALTER TABLE verbs ADD COLUMN source_pack TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # Column already exists


def _seed_from_pack_files():
    """
    Scans packs/ for .json pack files and imports them into the database.
    Returns the total number of verbs seeded across all packs.
    """
    total_seeded = 0

    try:
        pack_filenames = vocab_pack_manager.list_pack_files()
    except Exception:
        return 0  # packs dir doesn't exist or is inaccessible

    if not pack_filenames:
        return 0

    with get_connection() as conn:
        cursor = conn.cursor()

        for filename in pack_filenames:
            data = vocab_pack_manager.load_pack_file(filename)
            if data is None:
                continue

            pack_name = data.get("pack_name", "")
            if not pack_name:
                continue

            # Register the pack in vocabulary_packs (INSERT OR IGNORE)
            cursor.execute(
                """
                INSERT OR IGNORE INTO vocabulary_packs
                    (pack_name, display_name, description, category, version, is_builtin, is_ai_generated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_name,
                    data.get("display_name", pack_name),
                    data.get("description", ""),
                    data.get("category", ""),
                    data.get("version", "1.0"),
                    1 if data.get("author") == "curated" else 0,
                    1 if data.get("author") == "ai_generated" else 0,
                ),
            )

            # Import verbs
            verbs = data.get("verbs") or []
            if verbs:
                rows = [
                    (v["verb"].strip().lower(), v.get("difficulty", "B1"), v.get("definition", ""), pack_name)
                    for v in verbs
                ]
                cursor.executemany(
                    "INSERT OR IGNORE INTO verbs (verb, difficulty, definition, source_pack) VALUES (?, ?, ?, ?)",
                    rows,
                )
                seeded_count = cursor.rowcount
                total_seeded += seeded_count

                # Update verb count in the pack metadata
                cursor.execute(
                    "UPDATE vocabulary_packs SET verb_count = (SELECT COUNT(*) FROM verbs WHERE source_pack = ?) WHERE pack_name = ?",
                    (pack_name, pack_name),
                )

        conn.commit()

    return total_seeded


def _seed_from_fallback():
    """Seeds VERBS_SEED (legacy fallback) into the database when no pack files exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO verbs (verb, difficulty, definition) VALUES (?, ?, ?)",
            [(item["verb"], item["difficulty"], item["definition"]) for item in VERBS_SEED],
        )
        seeded = cursor.rowcount
        conn.commit()
    return seeded


def _sync_user_vocabulary_xlsx():
    """
    Auto-syncs Vocabulary.xlsx from packs/user_imported/ on every startup.
    If the file exists, its first-column words are imported as a pack named
    'user_vocabulary' so the user never needs to manually re-upload after editing
    the spreadsheet.
    """
    xlsx_path = os.path.join(
        os.path.dirname(__file__), "packs", "user_imported", "Vocabulary.xlsx"
    )
    if not os.path.isfile(xlsx_path):
        return  # Nothing to sync

    try:
        verbs, errors = vocab_pack_manager.parse_xlsx_verb_file_from_path(xlsx_path)
    except Exception:
        return  # Silently skip if xlsx parsing fails (missing dependency, etc.)

    if not verbs:
        return

    # The user's spreadsheet notes are informal; treat this as a bare word list
    # and let the AI handle meanings.  Override any auto-detected definitions.
    for v in verbs:
        v["definition"] = ""

    pack_name = "user_vocabulary"
    display_name = "手动导入词汇表"
    description = "从 packs/user_imported/Vocabulary.xlsx 自动同步的个人词汇表。编辑该 Excel 文件后重启应用即可自动更新。"
    category = "手动导入"

    with get_connection() as conn:
        cursor = conn.cursor()

        # Register / update the pack
        cursor.execute(
            """
            INSERT INTO vocabulary_packs (pack_name, display_name, description, category, version, is_builtin, is_ai_generated)
            VALUES (?, ?, ?, ?, '1.0', 0, 0)
            ON CONFLICT(pack_name) DO UPDATE SET
                display_name = excluded.display_name,
                description = excluded.description,
                category = excluded.category
            """,
            (pack_name, display_name, description, category),
        )

        # Import verbs (INSERT OR IGNORE keeps existing ones intact)
        rows = [
            (v["verb"], v.get("difficulty", "B1"), v.get("definition", ""), pack_name)
            for v in verbs
        ]
        cursor.executemany(
            "INSERT OR IGNORE INTO verbs (verb, difficulty, definition, source_pack) VALUES (?, ?, ?, ?)",
            rows,
        )

        # Refresh verb count
        cursor.execute(
            "UPDATE vocabulary_packs SET verb_count = (SELECT COUNT(*) FROM verbs WHERE source_pack = ?) WHERE pack_name = ?",
            (pack_name, pack_name),
        )

        # Create user_progress placeholders for new verbs
        cursor.execute("INSERT OR IGNORE INTO user_progress (verb) SELECT verb FROM verbs")

        conn.commit()

    return len(verbs)


def init_db():
    """Initializes the database, runs migrations, and seeds vocabulary."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Create verbs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS verbs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verb TEXT UNIQUE,
            difficulty TEXT,
            definition TEXT
        )
        """)

        # 2. Create user progress table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            verb TEXT PRIMARY KEY,
            attempts INTEGER DEFAULT 0,
            correct_attempts INTEGER DEFAULT 0,
            mastery_score INTEGER DEFAULT 0, -- 0 to 100
            last_tested TIMESTAMP,
            next_test TIMESTAMP,
            starred INTEGER DEFAULT 0, -- 0 or 1
            FOREIGN KEY(verb) REFERENCES verbs(verb)
        )
        """)

        # 3. Create test history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verb TEXT,
            scenario TEXT,
            chinese_sentence TEXT,
            expected_answer TEXT,
            user_answer TEXT,
            is_correct INTEGER,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 4. Create vocabulary packs metadata table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vocabulary_packs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pack_name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            version TEXT DEFAULT '1.0',
            is_enabled INTEGER DEFAULT 1,
            is_builtin INTEGER DEFAULT 0,
            is_ai_generated INTEGER DEFAULT 0,
            verb_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()

        # 5. Migration: add source_pack column if missing
        _ensure_source_pack_column(cursor)
        conn.commit()

    # 6. Seed from pack files (primary path) or fallback
    seeded = _seed_from_pack_files()

    if seeded == 0:
        # No pack files found — fall back to VERBS_SEED
        seeded = _seed_from_fallback()

    # 7. Backfill user_progress placeholders for any verb that lacks one
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO user_progress (verb) SELECT verb FROM verbs")
        conn.commit()

    # 8. Auto-sync user's personal Vocabulary.xlsx (if it exists)
    synced = _sync_user_vocabulary_xlsx()
    if synced:
        print(f"Synced {synced} verbs from packs/user_imported/Vocabulary.xlsx")

    if seeded > 0:
        print(f"Seeded {seeded} new verbs into the database.")


# =========================================================================
# Vocabulary Pack Management
# =========================================================================

def register_pack(pack_name, display_name, description="", category="", version="1.0",
                  is_builtin=False, is_ai_generated=False):
    """Inserts or updates a pack record in vocabulary_packs."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO vocabulary_packs (pack_name, display_name, description, category, version, is_builtin, is_ai_generated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pack_name) DO UPDATE SET
                display_name = excluded.display_name,
                description = excluded.description,
                category = excluded.category,
                version = excluded.version
            """,
            (pack_name, display_name, description, category, version,
             1 if is_builtin else 0, 1 if is_ai_generated else 0),
        )
        conn.commit()


def get_all_packs():
    """Returns all vocabulary pack records with their current verb counts."""
    with get_connection() as conn:
        # Refresh verb counts from the verbs table before returning
        conn.execute("""
            UPDATE vocabulary_packs
            SET verb_count = (SELECT COUNT(*) FROM verbs WHERE verbs.source_pack = vocabulary_packs.pack_name)
        """)
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM vocabulary_packs ORDER BY is_builtin DESC, display_name ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_enabled_pack_names():
    """Returns a list of pack_names for all enabled packs."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1"
        ).fetchall()
    return [row["pack_name"] for row in rows]


def get_enabled_pack_count():
    """Returns the count of enabled packs."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM vocabulary_packs WHERE is_enabled = 1"
        ).fetchone()
    return row["cnt"] if row else 0


def get_total_pack_count():
    """Returns the total number of vocabulary packs."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM vocabulary_packs").fetchone()
    return row["cnt"] if row else 0


def enable_pack(pack_name, enabled=True):
    """Enables or disables a vocabulary pack. Disabled packs are excluded from practice."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE vocabulary_packs SET is_enabled = ? WHERE pack_name = ?",
            (1 if enabled else 0, pack_name),
        )
        conn.commit()


def import_pack_verbs(pack_name, verbs):
    """
    Bulk-imports verbs for a given pack. Uses INSERT OR IGNORE so duplicates
    (same verb name already in DB) are silently skipped.

    Returns (imported_count, skipped_count).
    """
    if not verbs:
        return 0, 0

    with get_connection() as conn:
        cursor = conn.cursor()
        before_count = cursor.execute(
            "SELECT COUNT(*) FROM verbs WHERE source_pack = ?", (pack_name,)
        ).fetchone()[0]

        rows = [
            (v["verb"].strip().lower(), v.get("difficulty", "B1"), v.get("definition", ""), pack_name)
            for v in verbs
        ]
        cursor.executemany(
            "INSERT OR IGNORE INTO verbs (verb, difficulty, definition, source_pack) VALUES (?, ?, ?, ?)",
            rows,
        )

        after_count = cursor.execute(
            "SELECT COUNT(*) FROM verbs WHERE source_pack = ?", (pack_name,)
        ).fetchone()[0]

        imported = after_count - before_count
        skipped = len(verbs) - imported

        # Update verb_count in pack metadata
        cursor.execute(
            "UPDATE vocabulary_packs SET verb_count = ? WHERE pack_name = ?",
            (after_count, pack_name),
        )

        # Create user_progress placeholders
        cursor.execute("INSERT OR IGNORE INTO user_progress (verb) SELECT verb FROM verbs")

        conn.commit()

    return imported, skipped


def delete_pack(pack_name):
    """
    Deletes a pack and all its verbs (and associated progress/test history).
    Only non-builtin packs can be deleted.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        # Only allow deleting non-builtin packs
        row = cursor.execute(
            "SELECT is_builtin FROM vocabulary_packs WHERE pack_name = ?", (pack_name,)
        ).fetchone()
        if row and row["is_builtin"]:
            return False  # Cannot delete builtin packs

        # Delete from test_history for this pack's verbs
        cursor.execute(
            "DELETE FROM test_history WHERE verb IN (SELECT verb FROM verbs WHERE source_pack = ?)",
            (pack_name,),
        )
        # Delete from user_progress
        cursor.execute(
            "DELETE FROM user_progress WHERE verb IN (SELECT verb FROM verbs WHERE source_pack = ?)",
            (pack_name,),
        )
        # Delete from verbs
        cursor.execute("DELETE FROM verbs WHERE source_pack = ?", (pack_name,))
        # Delete pack metadata
        cursor.execute("DELETE FROM vocabulary_packs WHERE pack_name = ?", (pack_name,))
        conn.commit()
    return True


def get_verb_count_by_pack(pack_name):
    """Returns the number of verbs belonging to a specific pack."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM verbs WHERE source_pack = ?", (pack_name,)
        ).fetchone()
    return row["cnt"] if row else 0


def export_pack_to_dict(pack_name):
    """Exports a pack's metadata and verbs as a pack-format dict, or None if not found."""
    with get_connection() as conn:
        pack = conn.execute(
            "SELECT * FROM vocabulary_packs WHERE pack_name = ?", (pack_name,)
        ).fetchone()
        if not pack:
            return None

        verbs = conn.execute(
            "SELECT verb, difficulty, definition FROM verbs WHERE source_pack = ? ORDER BY verb",
            (pack_name,),
        ).fetchall()

    return {
        "pack_name": pack["pack_name"],
        "display_name": pack["display_name"],
        "description": pack["description"],
        "category": pack["category"],
        "language": "en-zh",
        "version": pack["version"],
        "author": "ai_generated" if pack["is_ai_generated"] else ("curated" if pack["is_builtin"] else "user"),
        "verbs": [{"verb": v["verb"], "difficulty": v["difficulty"], "definition": v["definition"]} for v in verbs],
    }


def export_all_verbs_to_dict(pack_filter=None):
    """
    Exports all verbs (optionally filtered by pack) as a list of dicts.
    Each dict includes verb, difficulty, definition, and progress fields.
    """
    query = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE 1=1
    """
    params = []

    if pack_filter:
        query += " AND v.source_pack = ?"
        params.append(pack_filter)
    else:
        # Include all verbs from enabled packs + NULL-source legacy verbs
        query += (
            " AND (v.source_pack IS NULL "
            " OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1))"
        )

    query += " ORDER BY v.verb ASC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


# =========================================================================
# Verb & Progress Queries
# =========================================================================

def get_all_verbs(difficulty_filter=None, starred_only=False, search_query=None, pack_filter=None):
    """
    Fetches all verbs with their progress details.

    Parameters:
        difficulty_filter: CEFR level (e.g. 'B1') or None for all.
        starred_only: If True, only starred verbs.
        search_query: Searches verb name and definition.
        pack_filter: If a pack_name string, only verbs from that pack.
                     If None, verbs from all enabled packs + legacy (NULL source_pack).
    """
    query = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE 1=1
    """
    params = []

    if pack_filter is not None:
        query += " AND v.source_pack = ?"
        params.append(pack_filter)
    else:
        # Default: verbs from enabled packs + legacy verbs with NULL source_pack
        query += (
            " AND (v.source_pack IS NULL "
            " OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1))"
        )

    if difficulty_filter:
        query += " AND v.difficulty = ?"
        params.append(difficulty_filter)

    if starred_only:
        query += " AND p.starred = 1"

    if search_query:
        pattern = f"%{_escape_like(search_query)}%"
        query += " AND (v.verb LIKE ? ESCAPE '\\' OR v.definition LIKE ? ESCAPE '\\')"
        params.extend([pattern, pattern])

    query += " ORDER BY v.verb ASC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_verb(verb):
    """Fetches one verb by exact name, with its progress details. None if not found."""
    query = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE v.verb = ?
    """
    with get_connection() as conn:
        row = conn.execute(query, (verb,)).fetchone()
    return dict(row) if row else None


def add_custom_verb(verb, difficulty, definition):
    """Allows user to add a new verb to their learning list."""
    verb = verb.strip().lower()
    if not verb:
        return False, "单词名称不能为空。"

    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO verbs (verb, difficulty, definition) VALUES (?, ?, ?)",
                (verb, difficulty, definition),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO user_progress (verb) VALUES (?)",
                (verb,),
            )
            conn.commit()
            return True, f"已将 '{verb}' 添加到你的单词库。"
        except sqlite3.IntegrityError:
            return False, f"单词 '{verb}' 已经存在于单词库中。"


def toggle_star(verb):
    """Toggles the starred status of a verb and returns the new status (0 or 1)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT starred FROM user_progress WHERE verb = ?", (verb,))
        row = cursor.fetchone()

        if row is None:
            cursor.execute("INSERT INTO user_progress (verb, starred) VALUES (?, 1)", (verb,))
            new_status = 1
        else:
            new_status = 0 if row["starred"] == 1 else 1
            cursor.execute("UPDATE user_progress SET starred = ? WHERE verb = ?", (new_status, verb))

        conn.commit()
    return new_status


def get_verb_for_test(difficulty_filter=None, starred_only=False):
    """
    Selects a verb to test the user based on Spaced Repetition (SM-2 principles):
    Priority:
    1. Starred verbs due for testing (next_test <= now) or starred and never tested.
    2. Other verbs due for testing (next_test <= now).
    3. Un-tested verbs (attempts = 0).
    4. Verbs with the lowest mastery score.

    Only verbs from enabled packs (or legacy verbs with NULL source_pack) are considered.
    """
    now_str = datetime.now().isoformat()

    # Restrict to enabled packs + legacy verbs
    pack_filter = (
        " AND (v.source_pack IS NULL "
        " OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1))"
    )

    base_join = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE 1=1{pack_filter}
    """
    params = []

    if difficulty_filter:
        base_join += " AND v.difficulty = ?"
        params.append(difficulty_filter)

    if starred_only:
        base_join += " AND p.starred = 1"

    # Each candidate query is tried in order; the first one with a hit wins.
    candidates = [
        # 1. Starred verbs that are due (or never tested)
        (base_join + " AND p.starred = 1 AND (p.next_test IS NULL OR p.next_test <= ?)"
                     " ORDER BY p.mastery_score ASC, RANDOM() LIMIT 1", params + [now_str]),
        # 2. Previously practiced verbs that are due
        (base_join + " AND (p.next_test IS NULL OR p.next_test <= ?) AND p.attempts > 0"
                     " ORDER BY p.mastery_score ASC, RANDOM() LIMIT 1", params + [now_str]),
        # 3. Never-practiced verbs
        (base_join + " AND p.attempts = 0 ORDER BY RANDOM() LIMIT 1", params),
        # 4. Anything left, weakest first
        (base_join + " ORDER BY p.mastery_score ASC, RANDOM() LIMIT 1", params),
    ]

    with get_connection() as conn:
        for query, query_params in candidates:
            row = conn.execute(query, query_params).fetchone()
            if row:
                return dict(row)
    return None


def update_progress(verb, is_correct):
    """
    Updates attempts and computes the new mastery score + next test date.
    Mastery score rules:
    - If correct: mastery_score = min(100, mastery_score + 20)
    - If incorrect: mastery_score = max(0, mastery_score - 15)

    Spaced Repetition interval based on mastery score:
    - 0-20 score: 1 hour delay
    - 21-40 score: 1 day delay
    - 41-60 score: 3 days delay
    - 61-80 score: 7 days delay
    - 81-100 score: 14 days delay
    """
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT attempts, correct_attempts, mastery_score FROM user_progress WHERE verb = ?",
            (verb,),
        )
        row = cursor.fetchone()

        if row is None:
            attempts, correct_attempts, mastery_score = 0, 0, 0
        else:
            attempts = row["attempts"]
            correct_attempts = row["correct_attempts"]
            mastery_score = row["mastery_score"]

        new_attempts = attempts + 1
        new_correct_attempts = correct_attempts + (1 if is_correct else 0)

        if is_correct:
            new_mastery = min(100, mastery_score + 20)
        else:
            new_mastery = max(0, mastery_score - 15)

        # Calculate next test time based on mastery
        now = datetime.now()
        if new_mastery <= 20:
            delay = timedelta(hours=1)
        elif new_mastery <= 40:
            delay = timedelta(days=1)
        elif new_mastery <= 60:
            delay = timedelta(days=3)
        elif new_mastery <= 80:
            delay = timedelta(days=7)
        else:
            delay = timedelta(days=14)

        next_test = (now + delay).isoformat()
        last_tested = now.isoformat()

        cursor.execute("""
            INSERT INTO user_progress (verb, attempts, correct_attempts, mastery_score, last_tested, next_test)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(verb) DO UPDATE SET
                attempts = excluded.attempts,
                correct_attempts = excluded.correct_attempts,
                mastery_score = excluded.mastery_score,
                last_tested = excluded.last_tested,
                next_test = excluded.next_test
        """, (verb, new_attempts, new_correct_attempts, new_mastery, last_tested, next_test))

        conn.commit()

    return {
        "attempts": new_attempts,
        "correct_attempts": new_correct_attempts,
        "old_mastery": mastery_score,
        "new_mastery": new_mastery,
        "next_test": next_test,
    }


def save_test_history(verb, scenario, chinese_sentence, expected_answer, user_answer, is_correct, feedback):
    """Saves details of a test round to the history table."""
    created_at = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO test_history
                (verb, scenario, chinese_sentence, expected_answer, user_answer, is_correct, feedback, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (verb, scenario, chinese_sentence, expected_answer, user_answer,
              1 if is_correct else 0, feedback, created_at))
        conn.commit()


def get_test_history(limit=50):
    """Fetches past test histories, newest first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, verb, scenario, chinese_sentence, expected_answer, user_answer,
                   is_correct, feedback, created_at
            FROM test_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_vocab_stats():
    """Calculates statistics for dashboard display, now pack-aware."""
    stats = {}

    with get_connection() as conn:
        cursor = conn.cursor()

        # Totals: only count verbs from enabled packs + legacy
        cursor.execute("""
            SELECT COUNT(*) FROM verbs v
            WHERE v.source_pack IS NULL
               OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1)
        """)
        stats["total_verbs"] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM user_progress p
            INNER JOIN verbs v ON v.verb = p.verb
            WHERE p.attempts > 0
              AND (v.source_pack IS NULL
                   OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1))
        """)
        stats["practiced_verbs"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM user_progress WHERE starred = 1")
        stats["starred_verbs"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*), SUM(is_correct) FROM test_history")
        history_row = cursor.fetchone()
        stats["total_tests_taken"] = history_row[0] or 0
        stats["total_correct_tests"] = history_row[1] or 0
        stats["accuracy_rate"] = (
            round(stats["total_correct_tests"] / stats["total_tests_taken"] * 100)
            if stats["total_tests_taken"] > 0 else 0
        )

        # Mastery bands
        cursor.execute("""
            SELECT
                SUM(CASE WHEN p.mastery_score >= 80 AND p.attempts > 0 THEN 1 ELSE 0 END) as master,
                SUM(CASE WHEN p.mastery_score >= 40 AND p.mastery_score < 80 THEN 1 ELSE 0 END) as intermediate,
                SUM(CASE WHEN p.mastery_score > 0 AND p.mastery_score < 40 THEN 1 ELSE 0 END) as beginner,
                SUM(CASE WHEN p.attempts = 0 OR p.attempts IS NULL THEN 1 ELSE 0 END) as unpracticed
            FROM verbs v
            LEFT JOIN user_progress p ON v.verb = p.verb
            WHERE v.source_pack IS NULL
               OR v.source_pack IN (SELECT pack_name FROM vocabulary_packs WHERE is_enabled = 1)
        """)
        bands_row = cursor.fetchone()
        stats["master_count"] = bands_row["master"] or 0
        stats["intermediate_count"] = bands_row["intermediate"] or 0
        stats["beginner_count"] = bands_row["beginner"] or 0
        stats["unpracticed_count"] = bands_row["unpracticed"] or 0

        # Pack-level stats
        cursor.execute("""
            SELECT source_pack, COUNT(*) AS verb_count
            FROM verbs
            WHERE source_pack IS NOT NULL
            GROUP BY source_pack
        """)
        stats["pack_verb_counts"] = {row["source_pack"]: row["verb_count"] for row in cursor.fetchall()}

    return stats


def clear_all_history():
    """Resets practice progress and clears test history. Starred verbs are preserved."""
    with get_connection() as conn:
        conn.execute("DELETE FROM test_history")
        conn.execute("""
            UPDATE user_progress
            SET attempts = 0, correct_attempts = 0, mastery_score = 0,
                last_tested = NULL, next_test = NULL
        """)
        conn.commit()
