# app/utils/database.py
"""
Database utilities for Mission7 Credit Scoring API.
Supports PostgreSQL (production) and SQLite (development).
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from typing import Optional
import logging

from app.config.settings import get_config
from app.utils.logging_config import setup_logging

logger = setup_logging('database')

# Global engine and session factory
_pg_engine = None
_pg_session_factory = None


def get_postgres_engine():
    """Get or create PostgreSQL engine with connection pooling."""
    global _pg_engine
    
    if _pg_engine is None:
        config = get_config()
        _pg_engine = create_engine(
            config.DB_URI,
            pool_size=config.DB_POOL_SIZE,
            max_overflow=config.DB_MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=config.DB_POOL_RECYCLE,
        )
        logger.info(f"PostgreSQL engine created: {config.DB_URI.split('@')[-1]}")
    
    return _pg_engine


def get_postgres_session():
    """Get a thread-safe PostgreSQL session with connection pooling."""
    global _pg_session_factory
    
    engine = get_postgres_engine()
    
    if _pg_session_factory is None:
        _pg_session_factory = scoped_session(sessionmaker(bind=engine))
    
    return _pg_session_factory()


def close_postgres_session(session):
    """Close a PostgreSQL session and remove from scoped registry."""
    global _pg_session_factory
    
    if session:
        session.close()
    
    if _pg_session_factory:
        _pg_session_factory.remove()


def get_client_from_postgres(client_id: int) -> pd.DataFrame:
    """
    Fetch client data from PostgreSQL using indexed SK_ID_CURR lookup.
    
    Args:
        client_id: The SK_ID_CURR to look up
        
    Returns:
        DataFrame with client data or empty DataFrame if not found
    """
    session = get_postgres_session()
    try:
        # Try application_train first
        query = text('SELECT * FROM application_train WHERE "SK_ID_CURR" = :client_id')
        result = session.execute(query, {"client_id": client_id})
        rows = result.fetchall()
        
        if not rows:
            # Try application_test
            query = text('SELECT * FROM application_test WHERE "SK_ID_CURR" = :client_id')
            result = session.execute(query, {"client_id": client_id})
            rows = result.fetchall()
        
        if rows:
            columns = result.keys()
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching client {client_id}: {e}")
        return pd.DataFrame()
    finally:
        session.close()


def log_prediction_to_postgres(
    client_id: int,
    probability: float,
    threshold: float,
    decision: str,
    model_version: Optional[str] = None,
    shap_values: Optional[dict] = None
) -> bool:
    """
    Log prediction to PostgreSQL for drift monitoring and audit.
    
    Args:
        client_id: The SK_ID_CURR
        probability: Model probability output
        threshold: Business threshold used
        decision: ACCEPTED or REJECTED
        model_version: Optional MLflow run_id
        shap_values: Optional dict of top SHAP values for debugging
        
    Returns:
        True if logged successfully, False otherwise
    """
    config = get_config()
    if not config.USE_POSTGRES:
        return False
    
    session = get_postgres_session()
    try:
        # Use JSON column if shap_values provided
        if shap_values:
            import json
            query = text("""
                INSERT INTO predictions (client_id, probability, threshold, decision, model_version, request_source, shap_values, created_at)
                VALUES (:client_id, :probability, :threshold, :decision, :model_version, 'api', CAST(:shap_values AS JSONB), NOW())
            """)
            session.execute(query, {
                "client_id": client_id,
                "probability": probability,
                "threshold": threshold,
                "decision": decision,
                "model_version": model_version,
                "shap_values": json.dumps(shap_values)
            })
        else:
            query = text("""
                INSERT INTO predictions (client_id, probability, threshold, decision, model_version, request_source, created_at)
                VALUES (:client_id, :probability, :threshold, :decision, :model_version, 'api', NOW())
            """)
            session.execute(query, {
                "client_id": client_id,
                "probability": probability,
                "threshold": threshold,
                "decision": decision,
                "model_version": model_version
            })
        session.commit()
        logger.debug(f"Prediction logged for client {client_id}: {decision}")
        return True
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")
        session.rollback()
        return False
    finally:
        session.close()


def get_prediction_history(limit: int = 100) -> list:
    """
    Get recent prediction history for audit trail.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of prediction records
    """
    config = get_config()
    if not config.USE_POSTGRES:
        return []
    
    session = get_postgres_session()
    try:
        query = text("""
            SELECT client_id, probability, threshold, decision, 
                   model_version, request_source, created_at
            FROM predictions 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        result = session.execute(query, {"limit": limit})
        predictions = [dict(row._mapping) for row in result.fetchall()]
        
        # Convert datetime to ISO format
        for pred in predictions:
            if pred.get('created_at'):
                pred['created_at'] = pred['created_at'].isoformat()
        
        return predictions
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        return []
    finally:
        session.close()
