"""
Optuna Hyperparameter Optimization for LightGBM
Clean interface for notebook usage with tqdm progress bar and dashboard
"""

import optuna
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm  # auto-detects notebook vs terminal
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline


def suppress_logging():
    """Suppress all verbose logging from Optuna and MLflow"""
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("mlflow.tracking").setLevel(logging.ERROR)


def run_optuna_optimization(
    preprocessor,
    X_train,
    y_train,
    param_space: dict,
    fixed_params: dict,
    n_trials: int = 50,
    scoring: str = 'roc_auc',
    cv_folds: int = 5,
    n_jobs_cv: int = 2,
    random_state: int = 42
):
    """
    Run Optuna hyperparameter optimization for LightGBM.
    
    Args:
        preprocessor: sklearn preprocessor pipeline
        X_train: training features
        y_train: training labels
        param_space: dict defining search space, e.g.:
            {
                'n_estimators': ('int', 300, 1500),
                'max_depth': ('categorical', [2, 3, 4, 5]),
                'learning_rate': ('float', 0.01, 0.2, True),  # (type, low, high, log)
            }
        fixed_params: dict of fixed LightGBM params (random_state, verbose, etc.)
        n_trials: number of Optuna trials
        scoring: sklearn scoring metric
        cv_folds: number of CV folds
        n_jobs_cv: parallel jobs for CV
        random_state: random seed
    
    Returns:
        study: Optuna study object
        best_pipeline: Fitted pipeline with best parameters
    """
    suppress_logging()
    
    def objective(trial):
        # Build params from param_space
        params = {}
        for param_name, config in param_space.items():
            if config[0] == 'int':
                params[param_name] = trial.suggest_int(param_name, config[1], config[2])
            elif config[0] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, config[1])
            elif config[0] == 'float':
                log = config[3] if len(config) > 3 else False
                params[param_name] = trial.suggest_float(param_name, config[1], config[2], log=log)
        
        # Add fixed params
        params.update(fixed_params)
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**params))
        ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs_cv)
        
        return scores.mean()
    
    # Create study and optimize with tqdm progress bar
    study = optuna.create_study(direction='maximize', study_name='lgbm_optuna')
    
    with tqdm(total=n_trials, desc="🔍 Optuna Optimization") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'best': f'{study.best_value:.4f}'})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    
    # Build final pipeline with best params
    best_params = study.best_params.copy()
    best_params.update(fixed_params)
    
    best_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**best_params))
    ])
    best_pipeline.fit(X_train, y_train)
    
    return study, best_pipeline


def plot_optuna_dashboard(study):
    """
    Display Optuna results dashboard with optimization history and parameter importance.
    """
    trials_df = study.trials_dataframe()
    
    # 1. Optimization History
    fig_history = go.Figure()
    fig_history.add_trace(go.Scatter(
        x=trials_df['number'], y=trials_df['value'],
        mode='markers', name='Trial Score',
        marker=dict(size=8, color=trials_df['value'], colorscale='Viridis', showscale=True)
    ))
    best_values = [trials_df['value'][:i+1].max() for i in range(len(trials_df))]
    fig_history.add_trace(go.Scatter(
        x=trials_df['number'], y=best_values,
        mode='lines', name='Best So Far',
        line=dict(color='red', width=2)
    ))
    fig_history.update_layout(
        title='📈 Optuna Optimization History',
        xaxis_title='Trial #', yaxis_title='CV ROC-AUC',
        template='plotly_white', height=400
    )
    fig_history.show()
    
    # 2. Parameter Importance
    param_cols = [c for c in trials_df.columns if c.startswith('params_')]
    if len(param_cols) > 0:
        importances = {}
        for col in param_cols:
            param_name = col.replace('params_', '')
            try:
                corr = trials_df[[col, 'value']].dropna().corr().iloc[0, 1]
                importances[param_name] = abs(corr) if not pd.isna(corr) else 0
            except:
                importances[param_name] = 0
        
        imp_df = pd.DataFrame({'param': list(importances.keys()), 'importance': list(importances.values())})
        imp_df = imp_df.sort_values('importance', ascending=True)
        
        fig_imp = px.bar(imp_df, x='importance', y='param', orientation='h',
                         title='🎯 Parameter Importance (Correlation with Score)',
                         template='plotly_white', height=400)
        fig_imp.update_traces(marker_color='steelblue')
        fig_imp.show()
    
    # 3. Best Parameters Table
    best_params_df = pd.DataFrame([
        {'Parameter': k, 'Value': v} for k, v in study.best_params.items()
    ])
    display(best_params_df.style.set_caption('🏆 Best Hyperparameters'))
    
    return trials_df


class OptunaResult:
    """Wrapper class for compatibility with the rest of the notebook"""
    def __init__(self, estimator, best_score):
        self.best_estimator_ = estimator
        self.best_score_ = best_score
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
