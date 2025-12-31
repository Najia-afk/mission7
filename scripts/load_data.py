#!/usr/bin/env python3
"""
Load Home Credit dataset CSVs into PostgreSQL database.
Creates tables, bulk inserts data, and creates indexes for fast lookups.

Usage:
    python scripts/load_data.py
    
    # Or with custom paths
    python scripts/load_data.py --dataset-path /path/to/csvs --db-uri postgresql://...
"""
import os
import sys
import argparse
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text, inspect

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.models import (
    Base, get_db_engine, create_all_tables
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASET_PATH = "/app/dataset"
DEFAULT_DB_URI = os.getenv(
    "DB_URI", 
    "postgresql://mission7:mission7pass@postgres:5432/credit_scoring"
)

# Files to load and their table names
CSV_TABLE_MAPPING = {
    "application_train.csv": "application_train",
    "application_test.csv": "application_test",
    "bureau.csv": "bureau",
    "bureau_balance.csv": "bureau_balance",
    "POS_CASH_balance.csv": "POS_CASH_balance",
    "credit_card_balance.csv": "credit_card_balance",
    "previous_application.csv": "previous_application",
    "installments_payments.csv": "installments_payments",
}

# Columns that should be indexed (in addition to primary keys)
INDEX_COLUMNS = {
    "application_train": [("SK_ID_CURR", True)],  # (column, unique)
    "application_test": [("SK_ID_CURR", True)],
    "bureau": [("SK_ID_CURR", False)],
    "bureau_balance": [("SK_ID_BUREAU", False)],
    "POS_CASH_balance": [("SK_ID_CURR", False)],
    "credit_card_balance": [("SK_ID_CURR", False)],
    "previous_application": [("SK_ID_CURR", False)],
    "installments_payments": [("SK_ID_CURR", False)],
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csv_to_postgres(
    csv_path: Path,
    table_name: str,
    engine,
    chunksize: int = 50000,
    if_exists: str = "replace"
) -> int:
    """
    Load a CSV file into PostgreSQL using pandas to_sql with chunked inserts.
    
    Args:
        csv_path: Path to the CSV file
        table_name: Target table name in database
        engine: SQLAlchemy engine
        chunksize: Number of rows per batch insert
        if_exists: 'replace' to overwrite, 'append' to add
    
    Returns:
        Number of rows inserted
    """
    print(f"\n📥 Loading {csv_path.name} → {table_name}...")
    
    if not csv_path.exists():
        print(f"  ⚠️ File not found: {csv_path}")
        return 0
    
    start_time = time.time()
    total_rows = 0
    
    # Read CSV in chunks for memory efficiency
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        # First chunk replaces, subsequent chunks append
        mode = if_exists if i == 0 else "append"
        
        chunk.to_sql(
            table_name,
            engine,
            if_exists=mode,
            index=False,
            method="multi"  # Faster batch inserts
        )
        
        total_rows += len(chunk)
        print(f"  ✓ Chunk {i+1}: {len(chunk):,} rows (total: {total_rows:,})")
    
    elapsed = time.time() - start_time
    print(f"  ✅ Completed in {elapsed:.1f}s ({total_rows:,} rows)")
    
    return total_rows


def create_indexes(engine, table_indexes: dict):
    """
    Create indexes on specified columns for fast lookups.
    
    Args:
        engine: SQLAlchemy engine
        table_indexes: Dict of {table_name: [(column, unique), ...]}
    """
    print("\n🔧 Creating indexes...")
    
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        for table_name, indexes in table_indexes.items():
            # Check if table exists
            if table_name not in inspector.get_table_names():
                print(f"  ⚠️ Table {table_name} not found, skipping indexes")
                continue
            
            existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
            
            for column, unique in indexes:
                index_name = f"idx_{table_name}_{column.lower()}"
                
                if index_name in existing_indexes:
                    print(f"  ✓ Index {index_name} already exists")
                    continue
                
                unique_str = "UNIQUE " if unique else ""
                sql = f'CREATE {unique_str}INDEX IF NOT EXISTS "{index_name}" ON "{table_name}" ("{column}")'
                
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    print(f"  ✅ Created {unique_str}index: {index_name}")
                except Exception as e:
                    print(f"  ❌ Error creating {index_name}: {e}")


def verify_data(engine):
    """
    Print row counts for all loaded tables.
    """
    print("\n📊 Verifying loaded data...")
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                count = result.scalar()
                print(f"  {table}: {count:,} rows")
            except Exception as e:
                print(f"  {table}: Error - {e}")


def test_lookup_performance(engine):
    """
    Test lookup speed on indexed column.
    """
    print("\n⚡ Testing lookup performance...")
    
    with engine.connect() as conn:
        # Get a sample client ID
        result = conn.execute(text("SELECT SK_ID_CURR FROM application_train LIMIT 1"))
        client_id = result.scalar()
        
        if client_id:
            start = time.time()
            for _ in range(100):
                conn.execute(text(f"SELECT * FROM application_train WHERE SK_ID_CURR = {client_id}"))
            elapsed = time.time() - start
            print(f"  ✅ 100 lookups in {elapsed*1000:.1f}ms ({elapsed*10:.2f}ms per query)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Load Home Credit dataset into PostgreSQL")
    parser.add_argument(
        "--dataset-path", 
        default=DEFAULT_DATASET_PATH,
        help="Path to directory containing CSV files"
    )
    parser.add_argument(
        "--db-uri",
        default=DEFAULT_DB_URI,
        help="PostgreSQL connection URI"
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip index creation"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=None,
        help="Specific tables to load (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Mission7 Data Loader")
    print("=" * 60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Database: {args.db_uri.split('@')[-1]}")  # Hide credentials
    
    # Create engine
    engine = create_engine(args.db_uri)
    
    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        sys.exit(1)
    
    # Create predictions table (for logging)
    print("\n📝 Creating predictions table...")
    create_all_tables(engine)
    
    # Load CSV files
    dataset_path = Path(args.dataset_path)
    total_rows = 0
    
    for csv_file, table_name in CSV_TABLE_MAPPING.items():
        # Filter by --tables argument if provided
        if args.tables and table_name not in args.tables:
            continue
        
        csv_path = dataset_path / csv_file
        rows = load_csv_to_postgres(csv_path, table_name, engine)
        total_rows += rows
    
    print(f"\n📦 Total rows loaded: {total_rows:,}")
    
    # Create indexes
    if not args.skip_indexes:
        create_indexes(engine, INDEX_COLUMNS)
    
    # Verify and test
    verify_data(engine)
    test_lookup_performance(engine)
    
    print("\n" + "=" * 60)
    print("✅ Data loading complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
