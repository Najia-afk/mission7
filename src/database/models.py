"""
SQLAlchemy models for Mission7 Credit Scoring database.
Includes indexed models for fast API lookups on SK_ID_CURR.

Uses SQLAlchemy 2.0+ modern patterns for Python 3.12+
"""
import os
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    BigInteger, Float, String, Integer, DateTime,
    Index, create_engine, text
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
    sessionmaker, scoped_session
)

# =============================================================================
# BASE CLASS (SQLAlchemy 2.0 style)
# =============================================================================

class Base(DeclarativeBase):
    """Base class for all models using SQLAlchemy 2.0 declarative style."""
    pass


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_engine(db_uri: str | None = None):
    """
    Create database engine with connection pooling.
    
    Args:
        db_uri: PostgreSQL connection string. If None, reads from DB_URI env var.
    
    Returns:
        SQLAlchemy engine with optimized pool settings.
    """
    if db_uri is None:
        db_uri = os.getenv("DB_URI", "postgresql://mission7:mission7pass@localhost:5432/credit_scoring")
    
    engine = create_engine(
        db_uri,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Test connections before use
        pool_recycle=3600,   # Recycle connections after 1 hour
    )
    return engine


def get_db_session(engine=None):
    """
    Get a thread-safe database session.
    
    Args:
        engine: SQLAlchemy engine. If None, creates new engine from env.
    
    Returns:
        Scoped session for database operations.
    """
    if engine is None:
        engine = get_db_engine()
    
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    return Session()


# =============================================================================
# APPLICATION DATA MODELS (Client Data) - SQLAlchemy 2.0 Style
# =============================================================================

class ApplicationTrain(Base):
    """
    Training data table - clients with known TARGET (loan outcome).
    Primary lookup: SK_ID_CURR (unique index for fast API queries).
    """
    __tablename__ = 'application_train'
    
    # Primary key with index
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    TARGET: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Core features
    NAME_CONTRACT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    CODE_GENDER: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    FLAG_OWN_CAR: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    FLAG_OWN_REALTY: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    CNT_CHILDREN: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_INCOME_TOTAL: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_ANNUITY: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_GOODS_PRICE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # External scores (critical for model)
    EXT_SOURCE_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    EXT_SOURCE_2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    EXT_SOURCE_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Employment & Age
    DAYS_BIRTH: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_EMPLOYED: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_REGISTRATION: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    DAYS_ID_PUBLISH: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Education & Family
    NAME_EDUCATION_TYPE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    NAME_FAMILY_STATUS: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    NAME_INCOME_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    OCCUPATION_TYPE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Housing
    NAME_HOUSING_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    REGION_POPULATION_RELATIVE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_app_train_sk_id_curr', 'SK_ID_CURR', unique=True),
    )


class ApplicationTest(Base):
    """
    Test data table - clients without TARGET (for prediction).
    Same structure as ApplicationTrain minus TARGET column.
    """
    __tablename__ = 'application_test'
    
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Same columns as ApplicationTrain except TARGET
    NAME_CONTRACT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    CODE_GENDER: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    FLAG_OWN_CAR: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    FLAG_OWN_REALTY: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    CNT_CHILDREN: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_INCOME_TOTAL: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_ANNUITY: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_GOODS_PRICE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    EXT_SOURCE_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    EXT_SOURCE_2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    EXT_SOURCE_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    DAYS_BIRTH: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_EMPLOYED: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_REGISTRATION: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    DAYS_ID_PUBLISH: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    NAME_EDUCATION_TYPE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    NAME_FAMILY_STATUS: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    NAME_INCOME_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    OCCUPATION_TYPE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    NAME_HOUSING_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    REGION_POPULATION_RELATIVE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_app_test_sk_id_curr', 'SK_ID_CURR', unique=True),
    )


# =============================================================================
# SECONDARY TABLES (For Multi-Table Feature Engineering)
# =============================================================================

class Bureau(Base):
    """
    Client's previous credits from other institutions reported to Credit Bureau.
    Index on SK_ID_CURR for aggregation joins.
    """
    __tablename__ = 'bureau'
    
    SK_ID_BUREAU: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, index=True)
    CREDIT_ACTIVE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    CREDIT_CURRENCY: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    DAYS_CREDIT: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    CREDIT_DAY_OVERDUE: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_CREDIT_ENDDATE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    DAYS_ENDDATE_FACT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT_MAX_OVERDUE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_CREDIT_PROLONG: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_CREDIT_SUM: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT_SUM_DEBT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT_SUM_LIMIT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT_SUM_OVERDUE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CREDIT_TYPE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    DAYS_CREDIT_UPDATE: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_ANNUITY: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_bureau_sk_id_curr', 'SK_ID_CURR'),
    )


class BureauBalance(Base):
    """
    Monthly balances of previous credits in Credit Bureau.
    Index on SK_ID_BUREAU for parent-child joins.
    """
    __tablename__ = 'bureau_balance'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SK_ID_BUREAU: Mapped[int] = mapped_column(BigInteger, index=True)
    MONTHS_BALANCE: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    STATUS: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    
    __table_args__ = (
        Index('idx_bureau_balance_sk_bureau', 'SK_ID_BUREAU'),
    )


class POSCashBalance(Base):
    """
    Monthly balance snapshots of previous POS and cash loans.
    """
    __tablename__ = 'POS_CASH_balance'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SK_ID_PREV: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, index=True)
    MONTHS_BALANCE: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    CNT_INSTALMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_INSTALMENT_FUTURE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    NAME_CONTRACT_STATUS: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    SK_DPD: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    SK_DPD_DEF: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    __table_args__ = (
        Index('idx_pos_cash_sk_id_curr', 'SK_ID_CURR'),
    )


class CreditCardBalance(Base):
    """
    Monthly balance snapshots of previous credit cards.
    """
    __tablename__ = 'credit_card_balance'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SK_ID_PREV: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, index=True)
    MONTHS_BALANCE: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_BALANCE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT_LIMIT_ACTUAL: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    AMT_DRAWINGS_ATM_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_DRAWINGS_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_DRAWINGS_OTHER_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_DRAWINGS_POS_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_INST_MIN_REGULARITY: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_PAYMENT_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_PAYMENT_TOTAL_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_RECEIVABLE_PRINCIPAL: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_RECIVABLE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_TOTAL_RECEIVABLE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_DRAWINGS_ATM_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_DRAWINGS_CURRENT: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    CNT_DRAWINGS_OTHER_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_DRAWINGS_POS_CURRENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    CNT_INSTALMENT_MATURE_CUM: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    NAME_CONTRACT_STATUS: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    SK_DPD: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    SK_DPD_DEF: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    __table_args__ = (
        Index('idx_credit_card_sk_id_curr', 'SK_ID_CURR'),
    )


class PreviousApplication(Base):
    """
    All previous applications for Home Credit loans.
    """
    __tablename__ = 'previous_application'
    
    SK_ID_PREV: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, index=True)
    NAME_CONTRACT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    AMT_ANNUITY: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_APPLICATION: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_CREDIT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_DOWN_PAYMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_GOODS_PRICE: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    WEEKDAY_APPR_PROCESS_START: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    HOUR_APPR_PROCESS_START: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    FLAG_LAST_APPL_PER_CONTRACT: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    NFLAG_LAST_APPL_IN_DAY: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    NAME_CASH_LOAN_PURPOSE: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    NAME_CONTRACT_STATUS: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    DAYS_DECISION: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    NAME_PAYMENT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    CODE_REJECT_REASON: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    NAME_CLIENT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    NAME_GOODS_CATEGORY: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    NAME_PORTFOLIO: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    NAME_PRODUCT_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    CHANNEL_TYPE: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    SELLERPLACE_AREA: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    NAME_SELLER_INDUSTRY: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    CNT_PAYMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    NAME_YIELD_GROUP: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    PRODUCT_COMBINATION: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    __table_args__ = (
        Index('idx_prev_app_sk_id_curr', 'SK_ID_CURR'),
    )


class InstallmentsPayments(Base):
    """
    Repayment history for previously disbursed credits.
    """
    __tablename__ = 'installments_payments'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SK_ID_PREV: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    SK_ID_CURR: Mapped[int] = mapped_column(BigInteger, index=True)
    NUM_INSTALMENT_VERSION: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    NUM_INSTALMENT_NUMBER: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    DAYS_INSTALMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    DAYS_ENTRY_PAYMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_INSTALMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    AMT_PAYMENT: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_installments_sk_id_curr', 'SK_ID_CURR'),
    )


# =============================================================================
# PREDICTION LOGGING (For Drift Monitoring)
# =============================================================================

class Prediction(Base):
    """
    Log of all predictions made by the API.
    Used for drift monitoring and compliance audit.
    """
    __tablename__ = 'predictions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_id: Mapped[int] = mapped_column(BigInteger, index=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)  # ACCEPTED / REJECTED
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    request_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow, 
        index=True
    )
    
    __table_args__ = (
        Index('idx_predictions_client_id', 'client_id'),
        Index('idx_predictions_created_at', 'created_at'),
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_tables(engine=None):
    """
    Create all tables in the database.
    
    Args:
        engine: SQLAlchemy engine. If None, creates from env.
    """
    if engine is None:
        engine = get_db_engine()
    Base.metadata.create_all(engine)
    print("✅ All tables created successfully")


def drop_all_tables(engine=None):
    """
    Drop all tables in the database. USE WITH CAUTION!
    
    Args:
        engine: SQLAlchemy engine. If None, creates from env.
    """
    if engine is None:
        engine = get_db_engine()
    Base.metadata.drop_all(engine)
    print("⚠️ All tables dropped")


if __name__ == "__main__":
    # Quick test - create tables
    engine = get_db_engine()
    create_all_tables(engine)
