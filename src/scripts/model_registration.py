"""
Script: Model Registration (Step 10)
Registers the best model in MLflow with business metadata.
"""
import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name, run_name, model_name, optimal_threshold=None, min_cost=None, extra_metrics=None, transition_to_prod=False):
    """
    Finds a run by name, logs business metadata, registers the model, and optionally promotes it.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name of the run to register
        model_name: Name to give the registered model
        optimal_threshold: (Optional) The business-optimized threshold
        min_cost: (Optional) The minimum business cost achieved
        extra_metrics: (Optional) Dict of additional metrics (auc_roc, f1, precision, recall, etc.)
        transition_to_prod: (Optional) If True, transitions the model to 'Production' stage
        
    Returns:
        registered_model object or None
    """
    print(f"Registering model from run '{run_name}' to registry as '{model_name}'...")
    client = MlflowClient()
    
    try:
        # 1. Find the correct Training Run ID
        run_id = None
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment:
            # Specifically look for the run by name
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], 
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                max_results=1
            )
            if not runs.empty:
                run_id = runs.iloc[0].run_id
                print(f"Found training run: {run_id}")
            else:
                print(f"Could not find run with name '{run_name}' in experiment '{experiment_name}'")
                return None
        else:
            print(f"Experiment '{experiment_name}' not found.")
            return None

        # 2. Log Business Metadata if provided
        if optimal_threshold is not None or min_cost is not None or extra_metrics is not None:
            with mlflow.start_run(run_id=run_id, nested=True):
                if optimal_threshold is not None:
                    mlflow.log_param("business_optimal_threshold", optimal_threshold)
                if min_cost is not None:
                    mlflow.log_metric("min_business_cost", min_cost)
                
                # Log all extra metrics (AUC, F1, Precision, Recall, etc.)
                if extra_metrics:
                    for key, value in extra_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"final_{key}", value)
                        else:
                            mlflow.log_param(f"final_{key}", value)
                    print(f"Logged {len(extra_metrics)} performance metrics")
                
                mlflow.set_tag("model_status", "production_ready")
                mlflow.set_tag("optimization_strategy", "cost_minimization")
                print(f"Logged business metadata to run {run_id}")

        # 3. Register the Model
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        print(f"Model registered as '{model_name}' v{registered_model.version}")

        # 4. Transition to Production if requested
        if transition_to_prod:
            print(f"Promoting v{registered_model.version} to 'Production' stage...")
            client.transition_model_version_stage(
                name=model_name,
                version=registered_model.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model v{registered_model.version} is now in Production. Old versions archived.")

        if optimal_threshold is not None:
            print(f"Deployment Note: Use probability threshold {optimal_threshold:.2f} for production inference.")
            
        return registered_model
        
    except Exception as e:
        print(f"Registration Error: {e}")
        return None
