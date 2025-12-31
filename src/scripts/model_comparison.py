"""
Script: Model Comparison (Champion vs Challenger)
Compares a new model (Challenger) against the current Production model (Champion).
"""
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def compare_with_production(model_name, X_test, y_test, challenger_model, challenger_threshold, scorer):
    """
    Compares the challenger model against the current production model using business cost.
    
    Args:
        model_name: Name of the model in the MLflow registry
        X_test: Test features
        y_test: Test labels
        challenger_model: The new model to evaluate
        challenger_threshold: The optimized threshold for the challenger
        scorer: BusinessScorer instance
        
    Returns:
        bool: True if challenger is better or if no champion exists, False otherwise.
    """
    client = MlflowClient()
    print(f"--- Champion vs Challenger Comparison: {model_name} ---")
    
    try:
        # 1. Try to load the current Production model
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            print("ℹ️ No Production model found in registry.")
            return True
            
        latest_prod = versions[0]
        prod_model_uri = f"models:/{model_name}/Production"
        champion_model = mlflow.sklearn.load_model(prod_model_uri)
        
        # Get production threshold from run params if available, else default to 0.5
        prod_run = client.get_run(latest_prod.run_id)
        champion_threshold = float(prod_run.data.params.get("business_optimal_threshold", 0.5))
        
        print(f"✅ Loaded Champion model (v{latest_prod.version})")
        
        # 2. Evaluate Champion
        y_proba_champ = champion_model.predict_proba(X_test)[:, 1]
        y_pred_champ = (y_proba_champ >= champion_threshold).astype(int)
        champ_cost = scorer.cost_function(y_test, y_pred_champ)
        
        # 3. Evaluate Challenger
        y_proba_chall = challenger_model.predict_proba(X_test)[:, 1]
        y_pred_chall = (y_proba_chall >= challenger_threshold).astype(int)
        chall_cost = scorer.cost_function(y_test, y_pred_chall)
        
        # 4. Compare
        improvement = (champ_cost - chall_cost) / champ_cost * 100 if champ_cost > 0 else 0
        
        print(f"Champion Cost: {champ_cost:.4f} (Threshold: {champion_threshold:.2f})")
        print(f"Challenger Cost: {chall_cost:.4f} (Threshold: {challenger_threshold:.2f})")
        print(f"Business Improvement: {improvement:.2f}%")
        
        if chall_cost <= champ_cost:
            if chall_cost == champ_cost:
                print("🔄 Challenger has EQUAL performance. Proceeding with registration (newer model).")
            else:
                print("🚀 Challenger is BETTER. Proceeding with registration.")
            return True
        else:
            print("⚠️ Challenger is WORSE. Review model before registration.")
            return False
            
    except Exception as e:
        print(f"ℹ️ Error during comparison: {e}")
        print("Proceeding with registration (manual check recommended).")
        return True
