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
        logger.info(f"Fetching client {client_id} from Postgres...")
        # Try application_train first
        query = text('SELECT * FROM application_train WHERE "SK_ID_CURR" = :client_id')
        result = session.execute(query, {"client_id": client_id})
        rows = result.fetchall()
        logger.info(f"Found {len(rows)} rows in application_train")
        
        if not rows:
            # Try application_test
            query = text('SELECT * FROM application_test WHERE "SK_ID_CURR" = :client_id')
            result = session.execute(query, {"client_id": client_id})
            rows = result.fetchall()
            logger.info(f"Found {len(rows)} rows in application_test")
        
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
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    shap_values: Optional[dict] = None,
    input_features: Optional[dict] = None
) -> bool:
    """
    Log prediction to PostgreSQL for drift monitoring and audit.
    
    Args:
        client_id: The SK_ID_CURR
        probability: Model probability output
        threshold: Business threshold used
        decision: ACCEPTED or REJECTED
        model_id: Model ID (run_id from prod_models/metadata.json)
        model_version: Optional model version string
        shap_values: Optional dict of top SHAP values for debugging
        input_features: Optional dict of input features
        
    Returns:
        True if logged successfully, False otherwise
    """
    config = get_config()
    if not config.USE_POSTGRES:
        return False
    
    session = get_postgres_session()
    try:
        import json
        
        # Prepare values
        values = {
            "client_id": client_id,
            "probability": probability,
            "threshold": threshold,
            "decision": decision,
            "model_id": model_id,
            "model_version": model_version,
            "shap_values": json.dumps(shap_values) if shap_values else None,
            "input_features": json.dumps(input_features) if input_features else None
        }
        
        query = text("""
            INSERT INTO predictions (
                client_id, probability, threshold, decision, 
                model_id, model_version, request_source, 
                shap_values, input_features, created_at
            )
            VALUES (
                :client_id, :probability, :threshold, :decision, 
                :model_id, :model_version, 'api', 
                CAST(:shap_values AS JSONB), CAST(:input_features AS JSONB), NOW()
            )
        """)
        
        session.execute(query, values)
        session.commit()
        logger.debug(f"Prediction logged for client {client_id}: {decision} (model: {model_id})")
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


def search_predictions(
    client_id: int = None,
    decision: str = None,
    min_score: float = None,
    max_score: float = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 100,
    offset: int = 0
) -> dict:
    """
    Search predictions with filters.
    
    Args:
        client_id: Filter by client ID
        decision: Filter by decision (ACCEPTED/REJECTED)
        min_score: Minimum probability score
        max_score: Maximum probability score
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        limit: Max results per page
        offset: Pagination offset
        
    Returns:
        Dict with predictions and pagination info
    """
    config = get_config()
    if not config.USE_POSTGRES:
        return {"predictions": [], "total": 0, "limit": limit, "offset": offset}
    
    session = get_postgres_session()
    try:
        # Build dynamic WHERE clause
        conditions = []
        params = {"limit": limit, "offset": offset}
        
        if client_id is not None:
            conditions.append("client_id = :client_id")
            params["client_id"] = client_id
        
        if decision:
            conditions.append("decision = :decision")
            params["decision"] = decision.upper()
        
        if min_score is not None:
            conditions.append("probability >= :min_score")
            params["min_score"] = min_score
        
        if max_score is not None:
            conditions.append("probability <= :max_score")
            params["max_score"] = max_score
        
        if start_date:
            conditions.append("created_at >= :start_date")
            params["start_date"] = start_date
        
        if end_date:
            conditions.append("created_at <= :end_date")
            params["end_date"] = end_date
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Count total
        count_query = text(f"SELECT COUNT(*) FROM predictions WHERE {where_clause}")
        total = session.execute(count_query, params).scalar()
        
        # Get data with model_id
        data_query = text(f"""
            SELECT id, client_id, probability, threshold, decision, 
                   model_id, model_version, request_source, created_at
            FROM predictions 
            WHERE {where_clause}
            ORDER BY created_at DESC 
            LIMIT :limit OFFSET :offset
        """)
        result = session.execute(data_query, params)
        predictions = [dict(row._mapping) for row in result.fetchall()]
        
        # Convert datetime to ISO format
        for pred in predictions:
            if pred.get('created_at'):
                pred['created_at'] = pred['created_at'].isoformat()
        
        return {
            "predictions": predictions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    except Exception as e:
        logger.error(f"Error searching predictions: {e}")
        return {"predictions": [], "total": 0, "limit": limit, "offset": offset, "error": str(e)}
    finally:
        session.close()


def get_prediction_stats() -> dict:
    """Get prediction statistics for dashboard."""
    config = get_config()
    if not config.USE_POSTGRES:
        return {}
    
    session = get_postgres_session()
    try:
        query = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN decision = 'ACCEPTED' THEN 1 ELSE 0 END) as accepted,
                SUM(CASE WHEN decision = 'REJECTED' THEN 1 ELSE 0 END) as rejected,
                AVG(probability) as avg_score,
                MIN(probability) as min_score,
                MAX(probability) as max_score,
                MIN(created_at) as first_prediction,
                MAX(created_at) as last_prediction
            FROM predictions
        """)
        result = session.execute(query).fetchone()
        
        if result:
            return {
                "total": result.total or 0,
                "accepted": result.accepted or 0,
                "rejected": result.rejected or 0,
                "approval_rate": (result.accepted / result.total * 100) if result.total else 0,
                "avg_score": round(result.avg_score, 4) if result.avg_score else 0,
                "min_score": round(result.min_score, 4) if result.min_score else 0,
                "max_score": round(result.max_score, 4) if result.max_score else 0,
                "first_prediction": result.first_prediction.isoformat() if result.first_prediction else None,
                "last_prediction": result.last_prediction.isoformat() if result.last_prediction else None
            }
        return {}
    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        return {}
    finally:
        session.close()


def get_prediction_by_id(prediction_id: int) -> dict:
    """Get a single prediction with all details including SHAP values."""
    config = get_config()
    if not config.USE_POSTGRES:
        return None
    
    session = get_postgres_session()
    try:
        query = text("""
            SELECT 
                id, client_id, probability, decision, threshold, 
                input_features, shap_values, created_at,
                model_id, model_version
            FROM predictions
            WHERE id = :prediction_id
        """)
        result = session.execute(query, {"prediction_id": prediction_id}).fetchone()
        
        if result:
            return {
                "id": result.id,
                "client_id": result.client_id,
                "probability": float(result.probability) if result.probability else None,
                "decision": result.decision,
                "threshold": float(result.threshold) if result.threshold else None,
                "input_features": result.input_features,
                "shap_values": result.shap_values,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                "model_id": result.model_id,
                "model_version": result.model_version
            }
        return None
    except Exception as e:
        logger.error(f"Error getting prediction by id: {e}")
        return None
    finally:
        session.close()
