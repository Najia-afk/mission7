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

echo "ðŸš€ Starting Mission7 Credit Scoring API..."

# Wait for PostgreSQL to be ready
echo "â³ Waiting for PostgreSQL..."
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
        echo "âœ… PostgreSQL is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Attempt $RETRY_COUNT/$MAX_RETRIES - PostgreSQL not ready, waiting..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "âŒ PostgreSQL not available after $MAX_RETRIES attempts"
    echo "   Continuing anyway - artifact sync will be skipped"
fi

# Seed dataset tables (only if missing/empty)
if [ -f "/app/app/scripts/seed_postgres_from_dataset.py" ]; then
    echo "ðŸ—„ï¸ Seeding PostgreSQL dataset tables (if needed)..."
    python /app/app/scripts/seed_postgres_from_dataset.py || {
        echo "âš ï¸ Warning: Dataset seed failed (non-fatal)"
    }
else
    echo "â„¹ï¸ Dataset seeding script not found; skipping"
fi

# Sync model artifacts to database
if [ -f "/app/app/scripts/sync_artifacts_to_db.py" ]; then
    echo "ðŸ“¦ Syncing model artifacts to database..."
    python /app/app/scripts/sync_artifacts_to_db.py --prod-models-dir /app/prod_models || {
        echo "âš ï¸ Warning: Artifact sync failed (non-fatal)"
    }
else
    echo "âš ï¸ Warning: sync_artifacts_to_db.py not found"
fi

# Register model in MLflow for experiment tracking
if [ -f "/app/app/scripts/register_model_mlflow.py" ]; then
    echo "ðŸ“Š Registering model in MLflow..."
    python /app/app/scripts/register_model_mlflow.py --prod-models-dir /app/prod_models || {
        echo "âš ï¸ Warning: MLflow registration failed (non-fatal)"
    }
else
    echo "â„¹ï¸ MLflow registration script not found; skipping"
fi

echo "ðŸŒ Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app.wsgi:app
