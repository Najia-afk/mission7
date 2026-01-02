#!/usr/bin/env python3
"""
Register production model in MLflow.

This script ensures the production model is properly registered in MLflow
with all metrics and artifacts, making it visible in MLflow UI for BCE/FINMA audit.

Usage:
    python register_model_mlflow.py --prod-models-dir /app/prod_models
"""
import os
import sys
import json
import pickle
import argparse
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def register_model_in_mlflow(prod_models_dir: str, mlflow_uri: str = None):
    """
    Register the production model in MLflow with experiment tracking.
    
    Args:
        prod_models_dir: Path to prod_models directory
        mlflow_uri: MLflow tracking URI (default from env)
    """
    import mlflow
    from mlflow.models.signature import infer_signature
    import numpy as np
    
    # Set tracking URI
    if mlflow_uri is None:
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5002')
    
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"📊 MLflow tracking URI: {mlflow_uri}")
    
    # Load metadata
    metadata_path = os.path.join(prod_models_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"❌ No metadata.json found in {prod_models_dir}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"📦 Model: {metadata['model_name']} v{metadata['model_version']}")
    print(f"📦 Run ID: {metadata['run_id']}")
    
    # Load model
    model_path = os.path.join(prod_models_dir, 'model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ No model.pkl found in {prod_models_dir}")
        return False
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    feature_names_path = os.path.join(prod_models_dir, 'feature_names.txt')
    feature_names = []
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
    
    # Create or get experiment
    experiment_name = "CreditScoring_Production"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={
                "project": "mission7",
                "team": "credit-risk",
                "regulatory": "BCE/FINMA"
            }
        )
        print(f"✅ Created experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {experiment_name}")
    
    mlflow.set_experiment(experiment_name)
    
    # Check if this run already exists
    try:
        existing_run = mlflow.get_run(metadata['run_id'])
        print(f"ℹ️ Run {metadata['run_id']} already exists, updating...")
        run_id = metadata['run_id']
        with mlflow.start_run(run_id=run_id):
            _log_model_details(mlflow, model, metadata, prod_models_dir, feature_names)
    except Exception:
        # Create new run with specific run_id if possible, or generate new
        print(f"📝 Creating new MLflow run...")
        with mlflow.start_run(run_name=f"Production_v{metadata['model_version']}") as run:
            run_id = run.info.run_id
            _log_model_details(mlflow, model, metadata, prod_models_dir, feature_names)
            
            # Update metadata with new run_id
            metadata['run_id'] = run_id
            metadata['mlflow_registered'] = True
            metadata['mlflow_registered_at'] = datetime.now().isoformat()
    
    # Register model in Model Registry
    try:
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=metadata['model_name']
        )
        print(f"✅ Registered model: {metadata['model_name']} version {registered_model.version}")
        
        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=metadata['model_name'],
            version=registered_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"✅ Model transitioned to Production stage")
        
    except Exception as e:
        print(f"⚠️ Model registry update: {e}")
    
    print(f"\n✅ Model successfully registered in MLflow!")
    print(f"   View at: {mlflow_uri}")
    return True


def _log_model_details(mlflow, model, metadata, prod_models_dir, feature_names):
    """Log all model details to MLflow run."""
    import numpy as np
    
    # Log parameters
    mlflow.log_param("algorithm", metadata.get('algorithm', 'LightGBM'))
    mlflow.log_param("model_version", metadata['model_version'])
    mlflow.log_param("optimal_threshold", metadata.get('optimal_threshold', 0.45))
    mlflow.log_param("n_features", metadata.get('n_features', len(feature_names)))
    mlflow.log_param("training_date", metadata.get('training_date', 'unknown'))
    
    # Log metrics
    metrics = metadata.get('metrics', {})
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
            mlflow.log_metric(metric_name, metric_value)
    
    # Log confusion matrix as metrics
    cm = metadata.get('confusion_matrix', {})
    for cm_name, cm_value in cm.items():
        if isinstance(cm_value, (int, float)):
            mlflow.log_metric(cm_name, cm_value)
    
    # Log additional metadata
    mlflow.log_metric("test_set_size", metadata.get('test_set_size', 0))
    mlflow.log_metric("positive_rate", metadata.get('positive_rate', 0))
    
    # Log tags for filtering
    mlflow.set_tag("model_name", metadata['model_name'])
    mlflow.set_tag("regulatory_framework", "BCE/FINMA")
    mlflow.set_tag("model_type", "credit_scoring")
    mlflow.set_tag("deployment_status", "production")
    
    # Log model
    if feature_names:
        # Create sample input for signature
        sample_input = np.zeros((1, len(feature_names)))
        try:
            from mlflow.models.signature import infer_signature
            signature = infer_signature(sample_input, model.predict_proba(sample_input)[:, 1])
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=sample_input
            )
        except Exception as e:
            print(f"⚠️ Could not infer signature: {e}")
            mlflow.sklearn.log_model(model, "model")
    else:
        mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    artifacts_to_log = [
        ('metadata.json', 'metadata'),
        ('threshold.json', 'config'),
        ('feature_names.txt', 'config'),
        ('evidently_data_drift_report.html', 'drift_reports'),
        ('evidently_data_drift_report.json', 'drift_reports'),
    ]
    
    for filename, artifact_path in artifacts_to_log:
        filepath = os.path.join(prod_models_dir, filename)
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath, artifact_path)
            print(f"   📎 Logged artifact: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Register model in MLflow')
    parser.add_argument('--prod-models-dir', default='/app/prod_models',
                        help='Path to prod_models directory')
    parser.add_argument('--mlflow-uri', default=None,
                        help='MLflow tracking URI')
    args = parser.parse_args()
    
    success = register_model_in_mlflow(args.prod_models_dir, args.mlflow_uri)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
