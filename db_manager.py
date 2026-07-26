import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from verbs_data import VERBS_SEED

DB_PATH = os.path.join(os.path.dirname(__file__), "vocab_tracker.db")

# Columns shared by every "verb + progress" query. COALESCE guards against verbs
# that somehow lost their user_progress row, so callers always get integers.
VERB_COLUMNS = """
    v.verb, v.difficulty, v.definition,
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


def init_db():
    """Initializes the database and seeds it with default English verbs."""
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

        conn.commit()

        # Seed any verb that isn't in the table yet. Running this on every start (rather
        # than only when the table is empty) means new seed words reach existing users.
        cursor.executemany(
            "INSERT OR IGNORE INTO verbs (verb, difficulty, definition) VALUES (?, ?, ?)",
            [(item["verb"], item["difficulty"], item["definition"]) for item in VERBS_SEED]
        )
        seeded = cursor.rowcount

        # Backfill progress placeholders for any verb that lacks one
        cursor.execute("""
            INSERT OR IGNORE INTO user_progress (verb)
            SELECT verb FROM verbs
        """)

        conn.commit()

    if seeded > 0:
        print(f"Seeded {seeded} new verbs into the database.")


def get_all_verbs(difficulty_filter=None, starred_only=False, search_query=None):
    """Fetches all verbs with their progress details."""
    query = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE 1=1
    """
    params = []

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
                (verb, difficulty, definition)
            )
            cursor.execute(
                "INSERT OR IGNORE INTO user_progress (verb) VALUES (?)",
                (verb,)
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
            # Create progress record if not exists
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
    """
    now_str = datetime.now().isoformat()

    base_join = f"""
    SELECT {VERB_COLUMNS}
    FROM verbs v
    LEFT JOIN user_progress p ON v.verb = p.verb
    WHERE 1=1
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
            (verb,)
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
    # Written explicitly in local time; SQLite's CURRENT_TIMESTAMP default is UTC, which
    # would disagree with the local timestamps used for spaced-repetition scheduling.
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
        # Ordered by id (monotonic) rather than created_at, so rows written before the
        # switch to local timestamps still interleave in true insertion order.
        rows = conn.execute("""
            SELECT id, verb, scenario, chinese_sentence, expected_answer, user_answer,
                   is_correct, feedback, created_at
            FROM test_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_vocab_stats():
    """Calculates statistics for dashboard display."""
    stats = {}

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM verbs")
        stats["total_verbs"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM user_progress WHERE attempts > 0")
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

        cursor.execute("""
            SELECT
                SUM(CASE WHEN mastery_score >= 80 THEN 1 ELSE 0 END) as master,
                SUM(CASE WHEN mastery_score >= 40 AND mastery_score < 80 THEN 1 ELSE 0 END) as intermediate,
                SUM(CASE WHEN mastery_score > 0 AND mastery_score < 40 THEN 1 ELSE 0 END) as beginner,
                SUM(CASE WHEN attempts = 0 THEN 1 ELSE 0 END) as unpracticed
            FROM user_progress
        """)
        bands_row = cursor.fetchone()
        stats["master_count"] = bands_row["master"] or 0
        stats["intermediate_count"] = bands_row["intermediate"] or 0
        stats["beginner_count"] = bands_row["beginner"] or 0
        stats["unpracticed_count"] = bands_row["unpracticed"] or 0

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
