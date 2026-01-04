#!/usr/bin/env python3
"""Seed PostgreSQL with the Home Credit dataset.

Why this exists
---------------
In production the API queries PostgreSQL tables like `application_train` and
`application_test`. On a fresh Postgres volume those tables are missing.

This script is:
- Idempotent: it skips if the target tables already contain rows.
- Flexible: reads from compressed .gz CSVs (preferred) or SQLite fallback.
- OPTIMIZED: Uses PostgreSQL COPY (10-100x faster than pandas to_sql).

It is intended to run at container startup (not at image build time).
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import sys
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text


DATASET_DIR = "/app/dataset"
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


def _get_postgres_connection():
    """Get a raw psycopg2 connection for COPY operations."""
    import psycopg2
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError("DB_URI is required")
    # Parse connection string: postgresql://user:pass@host:port/db
    from urllib.parse import urlparse
    parsed = urlparse(db_uri)
    return psycopg2.connect(
        dbname=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432,
    )


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
    """Create indexes and update query planner statistics."""
    statements = [
        'CREATE INDEX IF NOT EXISTS idx_application_train_sk_id_curr ON application_train ("SK_ID_CURR")',
        'CREATE INDEX IF NOT EXISTS idx_application_test_sk_id_curr ON application_test ("SK_ID_CURR")',
    ]
    with pg_engine.connect() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
    
    # Run ANALYZE to update query planner statistics (critical for performance)
    print("   Running ANALYZE for query optimization...")
    with pg_engine.connect() as conn:
        conn.execute(text("ANALYZE application_train"))
        conn.execute(text("ANALYZE application_test"))
        conn.commit()


def _get_csv_columns(csv_path: str, is_gzipped: bool = False) -> list:
    """Extract column names from CSV header using proper CSV parsing."""
    if is_gzipped:
        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            return next(reader)
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return next(reader)


def _infer_column_types(csv_path: str, is_gzipped: bool = False) -> dict:
    """
    Infer SQL column types from CSV data using proper CSV parsing.
    Reads sample rows to determine if column should be INTEGER, FLOAT, or TEXT.
    """
    if is_gzipped:
        f = gzip.open(csv_path, 'rt', encoding='utf-8')
    else:
        f = open(csv_path, 'r', encoding='utf-8')
    
    try:
        reader = csv.reader(f)
        columns = next(reader)  # Header row
        col_types = {}
        
        # Initialize all as potential integers
        is_int = {col: True for col in columns}
        is_float = {col: True for col in columns}
        
        # Sample first 100 data rows
        for row_num, row in enumerate(reader):
            if row_num >= 100:
                break
            
            for col_idx, col_name in enumerate(columns):
                if col_idx >= len(row):
                    continue
                
                val = row[col_idx].strip()
                if not val:  # Empty values can be any type
                    continue
                
                # Try integer
                if is_int[col_name]:
                    try:
                        int(val)
                    except ValueError:
                        is_int[col_name] = False
                
                # Try float (only if not integer)
                if is_float[col_name] and not is_int[col_name]:
                    try:
                        float(val)
                    except ValueError:
                        is_float[col_name] = False
        
        # Assign types
        for col_name in columns:
            if is_int[col_name]:
                col_types[col_name] = 'INTEGER'
            elif is_float[col_name]:
                col_types[col_name] = 'FLOAT'
            else:
                col_types[col_name] = 'TEXT'
        
        return col_types
    finally:
        f.close()


def _create_table_from_csv(pg_conn, csv_path: str, table_name: str, is_gzipped: bool = False):
    """Create table with proper column types inferred from CSV data."""
    columns = _get_csv_columns(csv_path, is_gzipped)
    col_types = _infer_column_types(csv_path, is_gzipped)
    
    # Create table with inferred types
    cols_def = ', '.join(f'"{col}" {col_types.get(col, "TEXT")}' for col in columns)
    create_sql = f'CREATE TABLE IF NOT EXISTS {table_name} ({cols_def})'
    
    cursor = pg_conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
    cursor.execute(create_sql)
    pg_conn.commit()
    cursor.close()


def _copy_csv_to_postgres_via_copy(
    csv_path: str,
    pg_conn,
    table_name: str,
    is_gzipped: bool = False,
) -> int:
    """Use PostgreSQL COPY command for fast bulk loading."""
    
    # Read CSV into memory
    if is_gzipped:
        with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
            csv_data = f.read()
    else:
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_data = f.read()
    
    # Parse first line to get column names
    lines = csv_data.split('\n')
    header = lines[0]
    columns = header.split(',')
    
    # Build COPY command
    cols_str = ', '.join(f'"{col}"' for col in columns)
    copy_cmd = f'COPY {table_name} ({cols_str}) FROM STDIN WITH (FORMAT csv, HEADER true)'
    
    # Execute COPY
    cursor = pg_conn.cursor()
    cursor.copy_expert(copy_cmd, io.StringIO(csv_data))
    pg_conn.commit()
    cursor.close()
    
    # Get row count
    cursor = pg_conn.cursor()
    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    count = cursor.fetchone()[0]
    cursor.close()
    
    return count


def _seed_from_compressed_csv(pg_engine, pg_conn, dataset_dir: str) -> bool:
    """Seed from compressed .gz CSV files using COPY."""
    train_gz = os.path.join(dataset_dir, "application_train_sample.csv.gz")
    test_gz = os.path.join(dataset_dir, "application_test.csv.gz")
    
    if not os.path.exists(train_gz) or not os.path.exists(test_gz):
        return False
    
    print(f"   Found compressed CSVs in {dataset_dir}")
    
    # Create empty tables with proper schema
    print(f"   Creating application_train table...")
    _create_table_from_csv(pg_conn, train_gz, "application_train", is_gzipped=True)
    
    print(f"   Creating application_test table...")
    _create_table_from_csv(pg_conn, test_gz, "application_test", is_gzipped=True)
    
    # Load train using COPY
    print(f"   Loading application_train...")
    train_rows = _copy_csv_to_postgres_via_copy(train_gz, pg_conn, "application_train", is_gzipped=True)
    
    # Load test using COPY
    print(f"   Loading application_test...")
    test_rows = _copy_csv_to_postgres_via_copy(test_gz, pg_conn, "application_test", is_gzipped=True)
    
    print(f"✅ Seeded from .gz: application_train={train_rows:,} rows, application_test={test_rows:,} rows")
    return True


def _seed_from_sqlite(pg_engine, sqlite_path: str) -> bool:
    if not os.path.exists(sqlite_path):
        return False

    sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
    train_chunks = pd.read_sql_query("SELECT * FROM application_train", sqlite_engine, chunksize=50_000)
    test_chunks = pd.read_sql_query("SELECT * FROM application_test", sqlite_engine, chunksize=50_000)

    total_train = 0
    total_test = 0
    
    first = True
    for chunk in train_chunks:
        if chunk is None or chunk.empty:
            continue
        if_exists = "replace" if first else "append"
        chunk.to_sql("application_train", pg_engine, if_exists=if_exists, index=False, method="multi")
        total_train += len(chunk)
        first = False

    first = True
    for chunk in test_chunks:
        if chunk is None or chunk.empty:
            continue
        if_exists = "replace" if first else "append"
        chunk.to_sql("application_test", pg_engine, if_exists=if_exists, index=False, method="multi")
        total_test += len(chunk)
        first = False

    print(f"✅ Seeded from SQLite: application_train={total_train:,} rows, application_test={total_test:,} rows")
    return True


def main() -> int:
    if not _env_bool("USE_POSTGRES", default=False):
        print("ℹ️ USE_POSTGRES is false; skipping dataset seed")
        return 0

    pg_engine = _get_postgres_engine()

    train_count = _table_row_count(pg_engine, "application_train")
    test_count = _table_row_count(pg_engine, "application_test")

    if (train_count or 0) > 0 and (test_count or 0) > 0:
        print(f"ℹ️ Postgres dataset already present; application_train={train_count:,} rows, application_test={test_count:,} rows")
        return 0

    dataset_dir = os.getenv("DATASET_DIR", DATASET_DIR)
    sqlite_path = os.getenv("DB_PATH", SQLITE_DEFAULT_PATH)

    print("📥 Seeding Postgres dataset tables (optimized with COPY)...")
    print(f"   DATASET_DIR={dataset_dir}")
    print(f"   DB_PATH={sqlite_path}")

    try:
        pg_conn = _get_postgres_connection()
        seeded = _seed_from_compressed_csv(pg_engine, pg_conn, dataset_dir)
        pg_conn.close()
    except Exception as e:
        print(f"⚠️ COPY method failed: {e}, falling back to pandas...")
        seeded = _seed_from_sqlite(pg_engine, sqlite_path)
    
    if not seeded:
        seeded = _seed_from_sqlite(pg_engine, sqlite_path)
    
    if not seeded:
        print("ℹ️ No dataset source found (.gz or SQLite); skipping Postgres seed")
        return 0

    _ensure_indexes(pg_engine)
    print("✅ Postgres dataset seed complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
