# app/api/services/client_service.py
"""
Client service for credit scoring.
Handles client data lookup and column management.
"""
import pandas as pd
from typing import Optional, List

from app.utils.database import get_client_from_postgres
from app.utils.logging_config import setup_logging

logger = setup_logging('client')


class ClientService:
    """Service for managing client data."""
    
    # Expected columns from application_train
    REQUIRED_COLUMNS = [
        'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
        'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
        'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
        'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
        'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
        'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
        'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
        'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
        'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
        'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
        'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
        'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    
    def __init__(self):
        self._columns_cache = None
    
    def get_client(self, client_id: int) -> Optional[dict]:
        """
        Get client data from PostgreSQL.
        
        Args:
            client_id: The SK_ID_CURR to look up
            
        Returns:
            Client data as dictionary, or None if not found
        """
        df = get_client_from_postgres(client_id)
        if df.empty:
            logger.warning(f"Client {client_id} not found")
            return None
        
        client_data = df.iloc[0].to_dict()
        logger.info(f"Client {client_id} retrieved")
        return client_data
    
    def get_client_dataframe(self, client_id: int) -> pd.DataFrame:
        """
        Get client data as DataFrame for prediction pipeline.
        
        Args:
            client_id: The SK_ID_CURR to look up
            
        Returns:
            DataFrame with client data
        """
        return get_client_from_postgres(client_id)
    
    def get_all_columns(self) -> List[str]:
        """
        Get all expected columns for the model.
        
        Returns:
            List of column names
        """
        return self.REQUIRED_COLUMNS.copy()
    
    def format_client_for_display(self, client_data: dict) -> dict:
        """
        Format client data for API response.
        
        Args:
            client_data: Raw client data dictionary
            
        Returns:
            Formatted dictionary with key fields
        """
        if not client_data:
            return {}
        
        return {
            "client_id": client_data.get("SK_ID_CURR"),
            "contract_type": client_data.get("NAME_CONTRACT_TYPE"),
            "gender": client_data.get("CODE_GENDER"),
            "income": client_data.get("AMT_INCOME_TOTAL"),
            "credit_amount": client_data.get("AMT_CREDIT"),
            "annuity": client_data.get("AMT_ANNUITY"),
            "goods_price": client_data.get("AMT_GOODS_PRICE"),
            "age_days": client_data.get("DAYS_BIRTH"),
            "employment_days": client_data.get("DAYS_EMPLOYED"),
            "education": client_data.get("NAME_EDUCATION_TYPE"),
            "family_status": client_data.get("NAME_FAMILY_STATUS"),
            "housing_type": client_data.get("NAME_HOUSING_TYPE"),
            "occupation": client_data.get("OCCUPATION_TYPE"),
            "own_car": client_data.get("FLAG_OWN_CAR"),
            "own_realty": client_data.get("FLAG_OWN_REALTY"),
            "ext_source_1": client_data.get("EXT_SOURCE_1"),
            "ext_source_2": client_data.get("EXT_SOURCE_2"),
            "ext_source_3": client_data.get("EXT_SOURCE_3"),
        }
    
    def get_client_summary(self, client_id: int) -> dict:
        """
        Get a summary of client information.
        
        Args:
            client_id: The SK_ID_CURR to look up
            
        Returns:
            Summary dictionary with formatted fields
        """
        client_data = self.get_client(client_id)
        if not client_data:
            return {"error": f"Client {client_id} not found"}
        
        return self.format_client_for_display(client_data)
