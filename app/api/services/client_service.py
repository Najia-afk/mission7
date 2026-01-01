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
    
    def get_similar_clients(self, client_id: int, n_neighbors: int = 10) -> dict:
        """
        Find similar clients for comparison analysis.
        
        Uses key numeric features to find nearest neighbors in the dataset.
        
        Args:
            client_id: Reference client SK_ID_CURR
            n_neighbors: Number of similar clients to return
            
        Returns:
            Dict with reference client and similar clients data
        """
        from app.utils.database import get_postgres_engine
        from sqlalchemy import text
        import numpy as np
        
        # Key features for similarity comparison
        comparison_features = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]
        
        # Get reference client
        ref_client = self.get_client(client_id)
        if not ref_client:
            return {"error": f"Client {client_id} not found"}
        
        try:
            engine = get_postgres_engine()
            with engine.connect() as conn:
                # Get sample of clients for comparison (limit for performance)
                query = text(f'''
                    SELECT "SK_ID_CURR", "TARGET", 
                           "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                           "DAYS_BIRTH", "DAYS_EMPLOYED", 
                           "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
                    FROM application_train
                    WHERE "SK_ID_CURR" != :client_id
                    ORDER BY RANDOM()
                    LIMIT 5000
                ''')
                result = conn.execute(query, {"client_id": client_id})
                rows = result.fetchall()
                
                if not rows:
                    return {"error": "No comparison data available"}
                
                # Build comparison matrix
                df_compare = pd.DataFrame(rows, columns=['SK_ID_CURR', 'TARGET'] + comparison_features)
                
                # Normalize features for distance calculation
                ref_values = []
                for feat in comparison_features:
                    ref_values.append(ref_client.get(feat, np.nan))
                ref_values = np.array(ref_values, dtype=float)
                
                # Calculate distances (ignoring NaN)
                distances = []
                for _, row in df_compare.iterrows():
                    compare_values = row[comparison_features].values.astype(float)
                    
                    # Calculate normalized Euclidean distance
                    valid_mask = ~(np.isnan(ref_values) | np.isnan(compare_values))
                    if valid_mask.sum() > 0:
                        diff = (ref_values[valid_mask] - compare_values[valid_mask])
                        # Normalize by feature range to prevent dominance
                        feat_std = df_compare[comparison_features].std()
                        norm_diff = diff / (feat_std[valid_mask].values + 1e-8)
                        dist = np.sqrt(np.sum(norm_diff ** 2))
                    else:
                        dist = np.inf
                    distances.append(dist)
                
                df_compare['distance'] = distances
                similar = df_compare.nsmallest(n_neighbors, 'distance')
                
                # Calculate statistics for comparison
                comparison_stats = {}
                for feat in comparison_features:
                    ref_val = ref_client.get(feat)
                    if ref_val is not None and not pd.isna(ref_val):
                        similar_mean = similar[feat].mean()
                        all_mean = df_compare[feat].mean()
                        comparison_stats[feat] = {
                            'client_value': float(ref_val),
                            'similar_mean': float(similar_mean) if not pd.isna(similar_mean) else None,
                            'population_mean': float(all_mean) if not pd.isna(all_mean) else None
                        }
                
                # Calculate default rate comparison
                ref_target = ref_client.get('TARGET', None)
                similar_default_rate = similar['TARGET'].mean() * 100
                population_default_rate = df_compare['TARGET'].mean() * 100
                
                return {
                    'client_id': client_id,
                    'client_target': int(ref_target) if ref_target is not None else None,
                    'n_similar': n_neighbors,
                    'comparison_stats': comparison_stats,
                    'similar_clients': similar[['SK_ID_CURR', 'TARGET', 'distance']].to_dict('records'),
                    'similar_default_rate': round(similar_default_rate, 2),
                    'population_default_rate': round(population_default_rate, 2)
                }
                
        except Exception as e:
            logger.error(f"Error finding similar clients: {e}")
            return {"error": str(e)}
    
    def get_bivariate_data(self, feature_x: str, feature_y: str, client_id: Optional[int] = None, sample_size: int = 1000) -> dict:
        """
        Get data for bi-variate analysis scatter plots.
        
        Args:
            feature_x: Feature for X axis
            feature_y: Feature for Y axis  
            client_id: Optional client to highlight
            sample_size: Number of samples to return
            
        Returns:
            Dict with scatter plot data points
        """
        from app.utils.database import get_postgres_engine
        from sqlalchemy import text
        
        # Validate features are in allowed list
        allowed_features = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
            'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE'
        ]
        
        if feature_x not in allowed_features or feature_y not in allowed_features:
            return {"error": f"Invalid features. Allowed: {allowed_features}"}
        
        try:
            engine = get_postgres_engine()
            with engine.connect() as conn:
                query = text(f'''
                    SELECT "SK_ID_CURR", "TARGET", "{feature_x}", "{feature_y}"
                    FROM application_train
                    WHERE "{feature_x}" IS NOT NULL AND "{feature_y}" IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT :sample_size
                ''')
                result = conn.execute(query, {"sample_size": sample_size})
                rows = result.fetchall()
                
                if not rows:
                    return {"error": "No data available"}
                
                df = pd.DataFrame(rows, columns=['SK_ID_CURR', 'TARGET', feature_x, feature_y])
                
                # Split by target for coloring
                accepted = df[df['TARGET'] == 0]
                rejected = df[df['TARGET'] == 1]
                
                result = {
                    'feature_x': feature_x,
                    'feature_y': feature_y,
                    'accepted': {
                        'x': accepted[feature_x].tolist(),
                        'y': accepted[feature_y].tolist(),
                        'ids': accepted['SK_ID_CURR'].tolist()
                    },
                    'rejected': {
                        'x': rejected[feature_x].tolist(),
                        'y': rejected[feature_y].tolist(),
                        'ids': rejected['SK_ID_CURR'].tolist()
                    }
                }
                
                # Add highlighted client if specified
                if client_id:
                    client_data = self.get_client(client_id)
                    if client_data:
                        result['highlight'] = {
                            'x': client_data.get(feature_x),
                            'y': client_data.get(feature_y),
                            'client_id': client_id
                        }
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting bivariate data: {e}")
            return {"error": str(e)}
    
    def get_available_features(self) -> List[str]:
        """Return list of features available for analysis."""
        return [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
            'REGION_POPULATION_RELATIVE', 'OWN_CAR_AGE'
        ]
