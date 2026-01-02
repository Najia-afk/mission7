#!/bin/bash
# =============================================================================
# Mission7 - Credit Scoring MLOps Platform
# Copyright (c) 2025-2026 All Rights Reserved.
# 
# This software is proprietary and confidential.
# Commercial use requires a paid license agreement.
# =============================================================================
#
# entrypoint.sh - Production container entrypoint
# Syncs artifacts to database before starting the API server

set -e

echo "🚀 Starting Mission7 Credit Scoring API..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if python -c "
import os
from sqlalchemy import create_engine, text
db_uri = os.getenv('DB_URI', 'postgresql://mission7:mission7pass@postgres:5432/credit_scoring')
engine = create_engine(db_uri)
with engine.connect() as conn:
    conn.execute(text('SELECT 1'))
print('OK')
" 2>/dev/null; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Attempt $RETRY_COUNT/$MAX_RETRIES - PostgreSQL not ready, waiting..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ PostgreSQL not available after $MAX_RETRIES attempts"
    echo "   Continuing anyway - artifact sync will be skipped"
fi

# Seed dataset tables (only if missing/empty)
if [ -f "/app/app/scripts/seed_postgres_from_dataset.py" ]; then
    echo "🗄️ Seeding PostgreSQL dataset tables (if needed)..."
    python /app/app/scripts/seed_postgres_from_dataset.py || {
        echo "⚠️ Warning: Dataset seed failed (non-fatal)"
    }
else
    echo "ℹ️ Dataset seeding script not found; skipping"
fi

# Sync model artifacts to database
if [ -f "/app/app/scripts/sync_artifacts_to_db.py" ]; then
    echo "📦 Syncing model artifacts to database..."
    python /app/app/scripts/sync_artifacts_to_db.py --prod-models-dir /app/prod_models || {
        echo "⚠️ Warning: Artifact sync failed (non-fatal)"
    }
else
    echo "⚠️ Warning: sync_artifacts_to_db.py not found"
fi

# Register model in MLflow for experiment tracking
if [ -f "/app/app/scripts/register_model_mlflow.py" ]; then
    echo "📊 Registering model in MLflow..."
    python /app/app/scripts/register_model_mlflow.py --prod-models-dir /app/prod_models || {
        echo "⚠️ Warning: MLflow registration failed (non-fatal)"
    }
else
    echo "ℹ️ MLflow registration script not found; skipping"
fi

echo "🌐 Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app.wsgi:app
