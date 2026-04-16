import os
import sqlite3
from pathlib import Path


local_db_path = Path("data/vt_cache.db")
colab_db_path = Path("data/cache/vt_cache.db")
merged_db_path = Path("data/vt_cache_merged.db")

expected_table = "vt_cache"
expected_columns = ["url", "response", "timestamp"]


def validate_db_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cache DB not found: {path}")


def fetch_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return [row[1] for row in rows]


def validate_db_schema(path: Path):
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (expected_table,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"{path} does not contain required table: {expected_table}")

        columns = fetch_table_columns(conn, expected_table)
        if columns != expected_columns:
            raise ValueError(
                f"{path} has unexpected schema.\n"
                f"Expected columns: {expected_columns}\n"
                f"Found columns: {columns}"
            )
    finally:
        conn.close()


def load_cache_rows(path: Path) -> dict[str, tuple[str, int]]:
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT url, response, timestamp FROM {expected_table}")
        rows = cur.fetchall()
        return {url: (response, int(timestamp)) for url, response, timestamp in rows}
    finally:
        conn.close()


def choose_row(
    local_row: tuple[str, int] | None,
    colab_row: tuple[str, int] | None,
) -> tuple[str, int]:
    if local_row is None:
        return colab_row
    if colab_row is None:
        return local_row

    local_response, local_ts = local_row
    colab_response, colab_ts = colab_row

    if local_ts > colab_ts:
        return local_row
    if colab_ts > local_ts:
        return colab_row

    return local_row


def write_merged_db(rows: dict[str, tuple[str, int]], output_path: Path):
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE vt_cache (
                url TEXT PRIMARY KEY,
                response TEXT,
                timestamp INTEGER
            )
            """
        )
        cur.executemany(
            "INSERT INTO vt_cache (url, response, timestamp) VALUES (?, ?, ?)",
            [(url, response, timestamp) for url, (response, timestamp) in rows.items()],
        )
        conn.commit()
    finally:
        conn.close()


def verify_merged_db(
    local_rows: dict[str, tuple[str, int]],
    colab_rows: dict[str, tuple[str, int]],
    merged_path: Path,
):
    merged_rows = load_cache_rows(merged_path)

    local_count = len(local_rows)
    colab_count = len(colab_rows)
    overlap_count = len(set(local_rows) & set(colab_rows))
    expected_merged_count = len(set(local_rows) | set(colab_rows))
    merged_count = len(merged_rows)

    print(f"Local cache rows: {local_count}")
    print(f"Colab cache rows: {colab_count}")
    print(f"Overlapping URLs: {overlap_count}")
    print(f"Expected merged rows: {expected_merged_count}")
    print(f"Actual merged rows: {merged_count}")

    if merged_count != expected_merged_count:
        raise ValueError(
            "Merged DB verification failed: row count does not match expected union of URLs."
        )

    print(f"Merged cache written to: {merged_path}")
    print("Verification passed.")


def main():
    print("Validating cache DB files...")
    validate_db_file(local_db_path)
    validate_db_file(colab_db_path)
    validate_db_schema(local_db_path)
    validate_db_schema(colab_db_path)

    print("Loading cache rows...")
    local_rows = load_cache_rows(local_db_path)
    colab_rows = load_cache_rows(colab_db_path)

    print("Merging cache rows...")
    merged_rows = {}
    all_urls = set(local_rows) | set(colab_rows)

    for url in all_urls:
        merged_rows[url] = choose_row(local_rows.get(url), colab_rows.get(url))

    print("Writing merged cache DB...")
    write_merged_db(merged_rows, merged_db_path)

    print("Verifying merged cache DB...")
    verify_merged_db(local_rows, colab_rows, merged_db_path)


if __name__ == "__main__":
    main()
