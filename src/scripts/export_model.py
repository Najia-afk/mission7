"""
Script: Export Production Model
Exports the registered model from MLflow to prod_models/ directory for API deployment.
"""
import os
import json
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def export_production_model(model_name: str, output_dir: str = "/app/prod_models"):
    """
    Export the Production-stage model from MLflow registry to disk.
    
    Args:
        model_name: Name of the registered model
        output_dir: Directory to save the model artifacts
        
    Returns:
        dict with export metadata or None on failure
    """
    client = MlflowClient()
    print(f"--- Exporting Production Model: {model_name} ---")
    
    try:
        # 1. Get Production model version
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            print(f"❌ No Production model found for '{model_name}'")
            return None
            
        prod_version = versions[0]
        print(f"✅ Found Production model v{prod_version.version}")
        
        # 2. Load the model
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # 3. Get run metadata
        run = client.get_run(prod_version.run_id)
        run_params = run.data.params
        run_metrics = run.data.metrics
        
        # Get threshold from params or default
        optimal_threshold = float(run_params.get("business_optimal_threshold", 0.45))
        business_cost = run_metrics.get("min_business_cost", None)
        auc = run_metrics.get("final_auc_roc", run_metrics.get("test_auc", run_metrics.get("val_auc", None)))
        
        # Get all final metrics (logged during registration)
        final_metrics = {k.replace("final_", ""): v for k, v in run_metrics.items() if k.startswith("final_")}
        
        # 4. Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 5. Save model pickle
        model_path = os.path.join(output_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Model saved to: {model_path}")
        
        # 6. Save threshold
        threshold_path = os.path.join(output_dir, "threshold.json")
        with open(threshold_path, 'w') as f:
            json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
        print(f"✅ Threshold saved to: {threshold_path}")
        
        # 7. Save feature names if available
        feature_names = []
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps'):
            # Pipeline - try to get from preprocessor
            if 'preprocessor' in model.named_steps:
                try:
                    feature_names = list(model.named_steps['preprocessor'].get_feature_names_out())
                except:
                    pass
        
        feature_path = os.path.join(output_dir, "feature_names.txt")
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_names) if feature_names else "# Feature names not available")
        
        # 8. Save comprehensive metadata
        metadata = {
            "run_id": prod_version.run_id,
            "model_name": model_name,
            "model_version": prod_version.version,
            "algorithm": run_params.get("classifier__class_name", "LightGBM"),
            "optimal_threshold": optimal_threshold,
            "training_date": run.info.start_time,
            "export_date": datetime.now().isoformat(),
            "mlflow_model_uri": model_uri,
            "n_features": len(feature_names) if feature_names else None,
            # Performance metrics
            "metrics": {
                "auc_roc": final_metrics.get("auc_roc", auc),
                "f1_score": final_metrics.get("f1_score"),
                "precision": final_metrics.get("precision"),
                "recall": final_metrics.get("recall"),
                "accuracy": final_metrics.get("accuracy"),
                "business_cost_avg": final_metrics.get("business_cost_avg", business_cost),
                "business_cost_total": final_metrics.get("business_cost_total"),
            },
            # Confusion matrix stats
            "confusion_matrix": {
                "true_positives": final_metrics.get("true_positives"),
                "true_negatives": final_metrics.get("true_negatives"),
                "false_positives": final_metrics.get("false_positives"),
                "false_negatives": final_metrics.get("false_negatives"),
            },
            "test_set_size": final_metrics.get("test_set_size"),
            "positive_rate": final_metrics.get("positive_rate"),
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✅ Metadata saved to: {metadata_path}")
        
        # 9. Summary
        print(f"\n📦 Model Export Complete!")
        print(f"   Version: v{prod_version.version}")
        print(f"   Threshold: {optimal_threshold:.2f}")
        if final_metrics:
            print(f"   AUC-ROC: {final_metrics.get('auc_roc', 'N/A')}")
            print(f"   F1-Score: {final_metrics.get('f1_score', 'N/A')}")
            print(f"   Precision: {final_metrics.get('precision', 'N/A')}")
            print(f"   Recall: {final_metrics.get('recall', 'N/A')}")
            print(f"   Business Cost: {final_metrics.get('business_cost_avg', 'N/A')}")
        print(f"   Output: {output_dir}")
        
        return metadata
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_exported_model(output_dir: str = "/app/prod_models"):
    """
    Verify the exported model can be loaded and make predictions.
    """
    print(f"\n--- Verifying Exported Model ---")
    
    try:
        # Load model
        model_path = os.path.join(output_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load threshold
        threshold_path = os.path.join(output_dir, "threshold.json")
        with open(threshold_path, 'r') as f:
            threshold_data = json.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"✅ Threshold: {threshold_data['optimal_threshold']}")
        print(f"✅ Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
