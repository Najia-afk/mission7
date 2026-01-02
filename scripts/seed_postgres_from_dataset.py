#!/usr/bin/env python3
"""Seed PostgreSQL with the Home Credit dataset.

Why this exists
---------------
In production the API queries PostgreSQL tables like `application_train` and
`application_test`. On a fresh Postgres volume those tables are missing.

This script is:
- Idempotent: it skips if the target tables already contain rows.
- Optional: it seeds only if a local SQLite dataset file exists.

It is intended to run at container startup (not at image build time).
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import create_engine, text


SQLITE_DEFAULT_PATH = "/app/dataset/home_credit.db"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_postgres_engine():
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError("DB_URI is required to seed Postgres")
    return create_engine(db_uri)


def _table_row_count(pg_engine, table_name: str) -> Optional[int]:
    with pg_engine.connect() as conn:
        exists = conn.execute(
            text("SELECT to_regclass(:full_name) IS NOT NULL"),
            {"full_name": f"public.{table_name}"},
        ).scalar()
        if not exists:
            return None
        return int(conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar() or 0)


def _ensure_indexes(pg_engine) -> None:
    # Queries in the app use quoted "SK_ID_CURR".
    statements = [
        'CREATE INDEX IF NOT EXISTS idx_application_train_sk_id_curr ON application_train ("SK_ID_CURR")',
        'CREATE INDEX IF NOT EXISTS idx_application_test_sk_id_curr ON application_test ("SK_ID_CURR")',
    ]
    with pg_engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()


def _copy_chunks_to_postgres(
    chunks: Iterable[pd.DataFrame],
    pg_engine,
    table_name: str,
) -> int:
    total = 0
    first = True
    for chunk in chunks:
        if chunk is None or chunk.empty:
            continue
        # Keep original column names (including uppercase) so existing SQL uses "SK_ID_CURR".
        if_exists = "replace" if first else "append"
        chunk.to_sql(table_name, pg_engine, if_exists=if_exists, index=False, method="multi")
        total += len(chunk)
        first = False
    return total


def _seed_from_sqlite(pg_engine, sqlite_path: str) -> bool:
    if not os.path.exists(sqlite_path):
        return False

    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")

    # Use chunked reads to avoid loading entire tables into memory.
    train_chunks = pd.read_sql_query("SELECT * FROM application_train", sqlite_engine, chunksize=50_000)
    test_chunks = pd.read_sql_query("SELECT * FROM application_test", sqlite_engine, chunksize=50_000)

    train_rows = _copy_chunks_to_postgres(train_chunks, pg_engine, "application_train")
    test_rows = _copy_chunks_to_postgres(test_chunks, pg_engine, "application_test")

    print(f"✅ Seeded from SQLite: application_train={train_rows:,} rows, application_test={test_rows:,} rows")
    return True


def main() -> int:
    if not _env_bool("USE_POSTGRES", default=False):
        print("ℹ️ USE_POSTGRES is false; skipping dataset seed")
        return 0

    pg_engine = _get_postgres_engine()

    train_count = _table_row_count(pg_engine, "application_train")
    test_count = _table_row_count(pg_engine, "application_test")

    if (train_count or 0) > 0 and (test_count or 0) > 0:
        print(
            "ℹ️ Postgres dataset already present; "
            f"application_train={train_count:,} rows, application_test={test_count:,} rows"
        )
        return 0

    sqlite_path = os.getenv("DB_PATH", SQLITE_DEFAULT_PATH)

    print("📥 Seeding Postgres dataset tables...")
    print(f"   DB_PATH={sqlite_path}")

    seeded = _seed_from_sqlite(pg_engine, sqlite_path)
    if not seeded:
        print("ℹ️ No local SQLite dataset found; skipping Postgres seed")
        return 0

    _ensure_indexes(pg_engine)
    print("✅ Postgres dataset seed complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
