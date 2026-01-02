#!/usr/bin/env python3
"""
Sync Model Artifacts to PostgreSQL Database.

This script reads model artifacts from prod_models/ directory and stores them
in the PostgreSQL model_artifacts table for audit compliance.

Run this script:
- After a git pull that updates prod_models/
- On container startup (entrypoint)
- Manually when promoting a new model

Usage:
    python scripts/sync_artifacts_to_db.py
    
Environment Variables:
    DB_URI: PostgreSQL connection string (defaults to mission7 standard)
"""
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def get_db_uri():
    """Get database URI from environment or use default."""
    return os.getenv(
        "DB_URI",
        "postgresql://mission7:mission7pass@localhost:5432/credit_scoring"
    )


def ensure_model_artifacts_table(engine):
    """Create model_artifacts table if it doesn't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS model_artifacts (
        id SERIAL PRIMARY KEY,
        model_id VARCHAR(64) NOT NULL,
        artifact_type VARCHAR(50) NOT NULL,
        artifact_name VARCHAR(255) NOT NULL,
        artifact_data TEXT,
        artifact_json JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(model_id, artifact_type)
    );
    CREATE INDEX IF NOT EXISTS idx_model_artifacts_model_id ON model_artifacts(model_id);
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    print("✅ model_artifacts table ensured")


def ensure_predictions_table(engine):
    """Create predictions table if it doesn't exist (for audit logging)."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        client_id INTEGER NOT NULL,
        probability FLOAT NOT NULL,
        threshold FLOAT NOT NULL,
        decision VARCHAR(20) NOT NULL,
        model_id VARCHAR(64),
        model_version VARCHAR(64),
        request_source VARCHAR(20) DEFAULT 'api',
        shap_values JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_client_id ON predictions(client_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
    CREATE INDEX IF NOT EXISTS idx_predictions_decision ON predictions(decision);
    CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    print("✅ predictions table ensured")


def load_file_content(filepath):
    """Load file content, return None if file doesn't exist."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return None


def load_json_file(filepath):
    """Load JSON file, return None if file doesn't exist."""
    content = load_file_content(filepath)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Invalid JSON in {filepath}")
    return None


def upsert_artifact(session, model_id, artifact_type, artifact_name, artifact_data=None, artifact_json=None):
    """Insert or update an artifact in the database."""
    # Check if exists
    check_sql = text("""
        SELECT id FROM model_artifacts 
        WHERE model_id = :model_id AND artifact_type = :artifact_type
    """)
    result = session.execute(check_sql, {"model_id": model_id, "artifact_type": artifact_type})
    existing = result.fetchone()
    
    if existing:
        # Update
        update_sql = text("""
            UPDATE model_artifacts 
            SET artifact_name = :artifact_name,
                artifact_data = :artifact_data,
                artifact_json = :artifact_json,
                created_at = :created_at
            WHERE model_id = :model_id AND artifact_type = :artifact_type
        """)
        session.execute(update_sql, {
            "model_id": model_id,
            "artifact_type": artifact_type,
            "artifact_name": artifact_name,
            "artifact_data": artifact_data,
            "artifact_json": json.dumps(artifact_json) if artifact_json else None,
            "created_at": datetime.utcnow()
        })
        print(f"  📝 Updated: {artifact_type}")
    else:
        # Insert
        insert_sql = text("""
            INSERT INTO model_artifacts (model_id, artifact_type, artifact_name, artifact_data, artifact_json, created_at)
            VALUES (:model_id, :artifact_type, :artifact_name, :artifact_data, :artifact_json, :created_at)
        """)
        session.execute(insert_sql, {
            "model_id": model_id,
            "artifact_type": artifact_type,
            "artifact_name": artifact_name,
            "artifact_data": artifact_data,
            "artifact_json": json.dumps(artifact_json) if artifact_json else None,
            "created_at": datetime.utcnow()
        })
        print(f"  ✨ Inserted: {artifact_type}")


def sync_artifacts(prod_models_dir=None):
    """
    Sync all artifacts from prod_models/ to the database.
    
    Args:
        prod_models_dir: Path to prod_models directory. Defaults to project's prod_models/
    """
    if prod_models_dir is None:
        prod_models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prod_models"
        )
    
    print(f"🔄 Syncing artifacts from: {prod_models_dir}")
    
    # Load metadata to get model_id (run_id)
    metadata_path = os.path.join(prod_models_dir, "metadata.json")
    metadata = load_json_file(metadata_path)
    
    if not metadata:
        print("❌ Error: metadata.json not found or invalid")
        return False
    
    model_id = metadata.get("run_id")
    if not model_id:
        print("❌ Error: run_id not found in metadata.json")
        return False
    
    print(f"📦 Model ID: {model_id}")
    print(f"📦 Model Name: {metadata.get('model_name', 'Unknown')}")
    print(f"📦 Model Version: {metadata.get('model_version', 'Unknown')}")
    
    # Connect to database
    db_uri = get_db_uri()
    # Mask password in log
    masked_uri = db_uri.split('@')[-1] if '@' in db_uri else db_uri
    print(f"🔗 Connecting to: {masked_uri}")
    
    engine = create_engine(db_uri)
    ensure_model_artifacts_table(engine)
    ensure_predictions_table(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 1. Sync metadata.json
        upsert_artifact(
            session, 
            model_id, 
            "metadata", 
            "metadata.json",
            artifact_data=json.dumps(metadata, indent=2),
            artifact_json=metadata
        )
        
        # 2. Sync threshold.json
        threshold_path = os.path.join(prod_models_dir, "threshold.json")
        threshold_data = load_json_file(threshold_path)
        if threshold_data:
            upsert_artifact(
                session,
                model_id,
                "threshold",
                "threshold.json",
                artifact_data=json.dumps(threshold_data, indent=2),
                artifact_json=threshold_data
            )
        
        # 3. Sync evidently_data_drift_report.html
        drift_html_path = os.path.join(prod_models_dir, "evidently_data_drift_report.html")
        drift_html = load_file_content(drift_html_path)
        if drift_html:
            upsert_artifact(
                session,
                model_id,
                "drift_report_html",
                "evidently_data_drift_report.html",
                artifact_data=drift_html
            )
        
        # 4. Sync evidently_data_drift_report.json
        drift_json_path = os.path.join(prod_models_dir, "evidently_data_drift_report.json")
        drift_json = load_json_file(drift_json_path)
        if drift_json:
            upsert_artifact(
                session,
                model_id,
                "drift_report_json",
                "evidently_data_drift_report.json",
                artifact_data=json.dumps(drift_json, indent=2),
                artifact_json=drift_json
            )
        
        # 5. Sync feature_names.txt
        features_path = os.path.join(prod_models_dir, "feature_names.txt")
        features_content = load_file_content(features_path)
        if features_content:
            feature_list = [f.strip() for f in features_content.strip().split('\n') if f.strip()]
            upsert_artifact(
                session,
                model_id,
                "feature_names",
                "feature_names.txt",
                artifact_data=features_content,
                artifact_json={"features": feature_list, "count": len(feature_list)}
            )
        
        session.commit()
        print(f"\n✅ Successfully synced artifacts for model {model_id}")
        return True
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error syncing artifacts: {e}")
        return False
    finally:
        session.close()
        engine.dispose()


def get_latest_model_id(engine):
    """Get the most recent model_id from the database."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT DISTINCT model_id FROM model_artifacts 
            ORDER BY model_id DESC LIMIT 1
        """))
        row = result.fetchone()
        return row[0] if row else None


def get_artifact(model_id, artifact_type, db_uri=None):
    """
    Retrieve an artifact from the database.
    
    Args:
        model_id: The model's run_id. Use 'latest' for most recent.
        artifact_type: Type of artifact (metadata, drift_report_html, etc.)
        db_uri: Database connection string
    
    Returns:
        dict with artifact_data and artifact_json, or None if not found
    """
    if db_uri is None:
        db_uri = get_db_uri()
    
    engine = create_engine(db_uri)
    
    try:
        with engine.connect() as conn:
            if model_id == 'latest':
                model_id = get_latest_model_id(engine)
                if not model_id:
                    return None
            
            result = conn.execute(text("""
                SELECT artifact_data, artifact_json, artifact_name, created_at
                FROM model_artifacts
                WHERE model_id = :model_id AND artifact_type = :artifact_type
            """), {"model_id": model_id, "artifact_type": artifact_type})
            
            row = result.fetchone()
            if row:
                return {
                    "artifact_data": row[0],
                    "artifact_json": row[1],
                    "artifact_name": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "model_id": model_id
                }
            return None
    finally:
        engine.dispose()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync model artifacts to PostgreSQL")
    parser.add_argument(
        "--prod-models-dir",
        help="Path to prod_models directory",
        default=None
    )
    parser.add_argument(
        "--db-uri",
        help="PostgreSQL connection URI",
        default=None
    )
    
    args = parser.parse_args()
    
    if args.db_uri:
        os.environ["DB_URI"] = args.db_uri
    
    success = sync_artifacts(args.prod_models_dir)
    sys.exit(0 if success else 1)
