import sqlite3
import json
import time
from typing import Optional

vt_db_path = "data/vt_cache.db"
cache_ttl = 7 * 24 * 3600  # 7 days

def init_vt_cache():
    conn = sqlite3.connect(vt_db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vt_cache (
            url TEXT PRIMARY KEY,
            response TEXT,
            timestamp INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_cached_vt(url: str) -> Optional[dict]:
    conn = sqlite3.connect(vt_db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT response, timestamp FROM vt_cache WHERE url = ?",
        (url,)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    response, ts = row
    if time.time() - ts > cache_ttl:
        return None

    return json.loads(response)


def save_vt_cache(url: str, response: dict):
    conn = sqlite3.connect(vt_db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO vt_cache VALUES (?, ?, ?)",
        (url, json.dumps(response), int(time.time()))
    )
    conn.commit()
    conn.close()