import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
import warnings
import json
import os

class ModelVisualizer:
    """
    Class for visualizing model performance, learning curves, and SHAP feature importance.
    Adapted from Mission 4 for classification tasks.
    
    Supports human-readable feature names via feature_mappings.json for better SHAP visualizations.
    """
    
    # Class-level cache for feature mappings
    _feature_mappings = None
    
    @classmethod
    def load_feature_mappings(cls):
        """Load feature name mappings from config file."""
        if cls._feature_mappings is not None:
            return cls._feature_mappings
        
        # Try multiple paths for flexibility
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '../../app/config/feature_mappings.json'),
            '/app/config/feature_mappings.json',  # Docker container path
            os.path.join(os.path.dirname(__file__), '../../../app/config/feature_mappings.json'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    cls._feature_mappings = json.load(f)
                    return cls._feature_mappings
        
        # Return empty mappings if file not found
        cls._feature_mappings = {"feature_names": {}, "days_to_years_columns": []}
        return cls._feature_mappings
    
    @classmethod
    def get_human_readable_name(cls, technical_name: str) -> str:
        """Convert technical feature name to human-readable format."""
        mappings = cls.load_feature_mappings()
        
        # Remove ColumnTransformer prefixes
        clean_name = technical_name
        for prefix in ['num__', 'cat__', 'ind__']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        # Remove _origin suffix from indicator columns
        if clean_name.endswith('_origin'):
            clean_name = clean_name[:-7]
        
        # Look up in mappings
        feature_names = mappings.get('feature_names', {})
        return feature_names.get(clean_name, clean_name)
    
    @classmethod
    def transform_value_for_display(cls, feature_name: str, value: float) -> tuple:
        """
        Transform value for human-readable display (e.g., DAYS to Years).
        
        Returns:
            Tuple of (display_value, unit_suffix)
        """
        mappings = cls.load_feature_mappings()
        days_columns = mappings.get('days_to_years_columns', [])
        
        # Check if this is a DAYS column (negative days to positive years)
        clean_name = feature_name
        for prefix in ['num__', 'cat__', 'ind__']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        if clean_name in days_columns:
            # DAYS_BIRTH = -15000 means ~41 years old
            # Convert: abs(days) / 365.25
            years = round(abs(value) / 365.25, 1)
            return years, "yrs"
        
        # Check if currency column
        currency_columns = mappings.get('currency_columns', [])
        if clean_name in currency_columns:
            return value, "$"
        
        # Check if percentage column
        percentage_columns = mappings.get('percentage_columns', [])
        if clean_name in percentage_columns:
            return round(value * 100, 1) if abs(value) < 10 else round(value, 1), "%"
        
        return value, ""
    
    @staticmethod
    def plot_learning_curves(models_dict, X, y, scorer=None, scoring_name='Score', cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
        """
        Generates and plots learning curves for multiple models with a dropdown selector.
        
        Parameters:
            models_dict: Dictionary of {model_name: model_object}.
            X: Training features.
            y: Training target.
            scorer: Scorer object or string (e.g., 'roc_auc').
            scoring_name: Label for the y-axis.
            cv: Cross-validation splitting strategy.
            n_jobs: Number of jobs to run in parallel.
            train_sizes: Relative or absolute numbers of training examples.
        """
        fig = go.Figure()
        
        buttons = []
        
        # Pre-compute curves for all models
        results = {}
        
        from tqdm.notebook import tqdm
        
        for i, (model_name, model) in tqdm(enumerate(models_dict.items()), total=len(models_dict), desc="Computing Curves"):
            try:
                train_sizes_abs, train_scores, test_scores = learning_curve(
                    model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scorer, verbose=0
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                results[model_name] = {
                    'train_sizes': train_sizes_abs,
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'test_mean': test_mean,
                    'test_std': test_std
                }
            except Exception as e:
                print(f"  Error computing learning curve for {model_name}: {e}")
                continue

        if not results:
            print("No learning curves could be computed.")
            return None

        # Add traces
        for i, (model_name, res) in enumerate(results.items()):
            # Visibility: only first model visible initially
            is_visible = (i == 0)
            
            # Training Score
            fig.add_trace(go.Scatter(
                x=res['train_sizes'], y=res['train_mean'],
                name='Training Score',
                mode='lines+markers',
                line=dict(color='blue'),
                visible=is_visible,
                legendgroup='train'
            ))
            
            # Training STD (Shaded Area)
            fig.add_trace(go.Scatter(
                x=np.concatenate([res['train_sizes'], res['train_sizes'][::-1]]),
                y=np.concatenate([res['train_mean'] + res['train_std'], (res['train_mean'] - res['train_std'])[::-1]]),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                visible=is_visible,
                legendgroup='train',
                hoverinfo='skip'
            ))
            
            # Cross-validation Score
            fig.add_trace(go.Scatter(
                x=res['train_sizes'], y=res['test_mean'],
                name='Cross-validation Score',
                mode='lines+markers',
                line=dict(color='green'),
                visible=is_visible,
                legendgroup='cv'
            ))
            
            # CV STD (Shaded Area)
            fig.add_trace(go.Scatter(
                x=np.concatenate([res['train_sizes'], res['train_sizes'][::-1]]),
                y=np.concatenate([res['test_mean'] + res['test_std'], (res['test_mean'] - res['test_std'])[::-1]]),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                visible=is_visible,
                legendgroup='cv',
                hoverinfo='skip'
            ))
            
            # Create button configuration
            # Each model has 4 traces
            visibility = [False] * (len(results) * 4)
            visibility[i*4 : (i+1)*4] = [True, True, True, True]
            
            buttons.append(dict(
                label=model_name,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Learning Curves: {model_name} ({scoring_name})"}]
            ))

        fig.update_layout(
            title=f"Learning Curves: {list(results.keys())[0]} ({scoring_name})",
            xaxis_title='Training Examples',
            yaxis_title=scoring_name,
            template='plotly_white',
            hovermode="x unified",
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=1.15,
                y=1.15,
                xanchor='left',
                yanchor='top'
            )]
        )
        
        return fig

    @staticmethod
    def plot_model_comparison(models_dict, X_test, y_test, business_scorer=None):
        """
        Compares multiple models on various classification metrics.
        """
        metrics = {
            'Model': [],
            'AUC': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': []
        }
        
        if business_scorer:
            metrics['Business Cost'] = []
            
        for name, model in models_dict.items():
            try:
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    auc = 0.5 # Fallback
                
                metrics['Model'].append(name)
                metrics['AUC'].append(auc)
                metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
                metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
                metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
                metrics['F1'].append(f1_score(y_test, y_pred, zero_division=0))
                
                if business_scorer:
                    cost = business_scorer.cost_function(y_test, y_pred)
                    metrics['Business Cost'].append(cost)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                
        df_metrics = pd.DataFrame(metrics)
        
        # Create subplots
        rows = 2
        cols = 3
        
        subplot_titles = ['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']
        if business_scorer:
            subplot_titles.append('Avg Business Cost (Lower is Better)')
            
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        
        # Helper to add trace
        def add_metric_bar(metric_name, row, col, color):
            if metric_name in df_metrics.columns:
                # Highlight Baseline in red
                colors = ['#d62728' if model == 'Baseline' else color for model in df_metrics['Model']]
                
                fig.add_trace(
                    go.Bar(
                        x=df_metrics['Model'],
                        y=df_metrics[metric_name],
                        name=metric_name,
                        marker_color=colors,
                        text=df_metrics[metric_name].apply(lambda x: f"{x:.3f}" if metric_name != 'Business Cost' else f"{x:.4f}"),
                        textposition='auto'
                    ),
                    row=row, col=col
                )

        add_metric_bar('AUC', 1, 1, '#1f77b4')
        add_metric_bar('Accuracy', 1, 2, '#ff7f0e')
        add_metric_bar('F1', 1, 3, '#2ca02c')
        add_metric_bar('Precision', 2, 1, '#d62728')
        add_metric_bar('Recall', 2, 2, '#9467bd')
        
        if business_scorer:
            add_metric_bar('Business Cost', 2, 3, '#8c564b')
            
        fig.update_layout(
            title='Model Performance Comparison',
            height=700,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig

    @staticmethod
    def compute_shap_values(model, X, sample_size=100):
        """
        Computes SHAP values for a given model using a sample of X.
        """
        # Sample data for speed
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
            
        # Extract best_estimator_ if it's a search object (GridSearchCV, HalvingGridSearchCV)
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
            
        # Extract model and preprocessor if pipeline
        if hasattr(model, 'named_steps'):
            # Assuming standard pipeline structure
            if 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                X_processed = preprocessor.transform(X_sample)
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
                
                # Convert to DF for easier handling
                if hasattr(X_processed, 'toarray'):
                    X_processed = X_processed.toarray()
                X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_sample.index)
            else:
                X_processed_df = X_sample
                feature_names = X.columns
            
            # Get the actual estimator
            step_name = 'model' if 'model' in model.named_steps else list(model.named_steps.keys())[-1]
            model_obj = model.named_steps[step_name]
        else:
            model_obj = model
            X_processed_df = X_sample
            feature_names = X.columns

        # Select Explainer
        try:
            # Tree-based models
            if hasattr(model_obj, 'feature_importances_'):
                explainer = shap.TreeExplainer(model_obj)
                shap_values = explainer.shap_values(X_processed_df)
            # Linear models
            elif hasattr(model_obj, 'coef_'):
                explainer = shap.LinearExplainer(model_obj, X_processed_df)
                shap_values = explainer.shap_values(X_processed_df)
            # Fallback to KernelExplainer (slow)
            else:
                print("Using KernelExplainer (this might be slow)...")
                explainer = shap.KernelExplainer(model_obj.predict, X_processed_df)
                shap_values = explainer.shap_values(X_processed_df)
        except Exception as e:
            print(f"Error creating SHAP explainer: {e}")
            return None

        # Handle different SHAP value formats (list for classification vs array)
        if isinstance(shap_values, list):
            # For binary classification, take the positive class (index 1)
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]

        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
            if len(expected_value) > 1:
                expected_value = expected_value[1]
            else:
                expected_value = expected_value[0]

        return {
            'shap_values': shap_values,
            'expected_value': expected_value,
            'feature_names': feature_names,
            'X_processed': X_processed_df
        }

    @staticmethod
    def plot_shap_summary(shap_data, top_n=20):
        """
        Creates a global feature importance plot using mean absolute SHAP values.
        """
        if shap_data is None:
            return None
            
        shap_values = np.array(shap_data['shap_values'])
        feature_names = list(shap_data['feature_names'])
        
        # Calculate mean absolute SHAP value for each feature
        # Ensure 2D array (samples x features)
        if shap_values.ndim == 1:
            mean_abs_shap = np.abs(shap_values)
        else:
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Flatten if needed
        mean_abs_shap = np.array(mean_abs_shap).flatten()
        
        # Ensure feature_names matches
        n_features = min(len(mean_abs_shap), len(feature_names))
        mean_abs_shap = mean_abs_shap[:n_features]
        feature_names = feature_names[:n_features]
        
        # Sort features and take top N
        top_n = min(top_n, n_features)
        sorted_idx = np.argsort(mean_abs_shap)[-top_n:]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = mean_abs_shap[sorted_idx].tolist()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h',
            marker=dict(
                color='#ff0051',  # SHAP-like red/pink color
            )
        ))
        
        fig.update_layout(
            title=f"Global Feature Importance (Mean |SHAP Value|) - Top {top_n}",
            xaxis_title="mean(|SHAP value|)",
            yaxis_title="",
            height=max(400, top_n * 25),
            width=800,
            template='plotly_white',
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=200)
        )
        
        return fig

    @staticmethod
    def plot_shap_local(shap_data, sample_idx=0):
        """
        Creates a local feature importance plot (Force Plot equivalent) for a specific sample.
        """
        if shap_data is None:
            return None
            
        shap_values = shap_data['shap_values'][sample_idx]
        feature_names = list(shap_data['feature_names'])  # Convert to list
        X_processed = shap_data['X_processed']
        base_value = shap_data['expected_value']
        
        # Ensure shap_values is 1D
        shap_values = np.array(shap_values).flatten()
        
        # Ensure feature_names matches shap_values length
        n_features = min(len(shap_values), len(feature_names))
        shap_values = shap_values[:n_features]
        feature_names = feature_names[:n_features]
        
        # Sort features by absolute impact
        sorted_idx = np.argsort(np.abs(shap_values))
        # Take top 15 (or fewer if less available)
        top_n = min(15, len(sorted_idx))
        sorted_idx = sorted_idx[-top_n:]
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_shap = shap_values[sorted_idx]
        sorted_values = [float(X_processed.iloc[sample_idx, i]) if i < X_processed.shape[1] else 0.0 for i in sorted_idx]
        
        final_value = base_value + np.sum(shap_values)
        
        # Create labels with human-readable names
        y_labels = []
        for val, feat in zip(sorted_values, sorted_features):
            # Get human-readable name
            human_name = ModelVisualizer.get_human_readable_name(feat)
            
            # Transform value for display (e.g., DAYS to Years)
            display_val, unit = ModelVisualizer.transform_value_for_display(feat, val)
            
            # Clean feature name and format label
            clean_feat = feat
            if clean_feat.startswith('cat__'):
                clean_feat = clean_feat.replace('cat__', '')
                # If it's a categorical flag (value 1), make it explicit
                if abs(val - 1.0) < 1e-3:
                    # Try to find the last underscore which usually separates feature name from category
                    parts = clean_feat.rsplit('_', 1)
                    if len(parts) > 1:
                        human_name = f"{ModelVisualizer.get_human_readable_name(parts[0])}: {parts[1]}"
                    y_labels.append(f"<b>YES</b> | {human_name}")
                else:
                    y_labels.append(f"<b>NO</b> | {human_name}")
            elif clean_feat.startswith('num__'):
                if unit:
                    y_labels.append(f"<b>{display_val:.1f}{unit}</b> | {human_name}")
                else:
                    y_labels.append(f"<b>{display_val:.2f}</b> | {human_name}")
            elif clean_feat.startswith('ind__'):
                clean_feat = clean_feat.replace('ind__', '').replace('_origin', '')
                status = "PRESENT" if abs(val - 1.0) < 1e-3 else "MISSING"
                y_labels.append(f"<b>{status}</b> | {human_name}")
            else:
                if unit:
                    y_labels.append(f"<b>{display_val:.1f}{unit}</b> | {human_name}")
                else:
                    y_labels.append(f"<b>{display_val:.2f}</b> | {human_name}")
        
        fig = go.Figure()
        
        # Ensure scalar comparisons - red increases risk (default), blue decreases risk
        colors = ['#d62728' if float(v) > 0 else '#1f77b4' for v in sorted_shap]
        
        fig.add_trace(go.Bar(
            x=sorted_shap.tolist(),
            y=y_labels,
            orientation='h',
            marker_color=colors,
            text=[f"{float(val):.2f}" for val in sorted_values],
            textposition='auto',
            hovertemplate="Feature: %{y}<br>Raw Value: %{text}<br>SHAP: %{x:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Feature Impact on Risk Score<br><span style='font-size:12px'>Base: {base_value:.2f} → Final: {final_value:.2f} | <span style='color:#d62728'>Red = Increases Risk</span> | <span style='color:#1f77b4'>Blue = Decreases Risk</span></span>",
            xaxis_title="SHAP Value (Impact on Default Probability)",
            yaxis_title="",
            height=600,
            template='plotly_white',
            margin=dict(l=250)  # More space for readable labels
        )
        
        return fig

    @staticmethod
    def plot_score_distribution(y_true, y_proba, threshold=0.5):
        """
        Plots the distribution of predicted probabilities for positive and negative classes,
        highlighting the decision threshold and risk zones.
        
        Parameters:
        -----------
        y_true : array-like
            True labels (0 or 1)
        y_proba : array-like
            Predicted probabilities for the positive class (1)
        threshold : float
            Decision threshold (default: 0.5)
        """
        df_scores = pd.DataFrame({'True Label': y_true, 'Probability': y_proba})
        
        fig = go.Figure()
        
        # Distribution for Negative Class (Repayment - Target=0)
        fig.add_trace(go.Histogram(
            x=df_scores[df_scores['True Label'] == 0]['Probability'],
            name='Repayment (Target=0)',
            marker_color='green',
            opacity=0.6,
            nbinsx=50
        ))
        
        # Distribution for Positive Class (Default - Target=1)
        fig.add_trace(go.Histogram(
            x=df_scores[df_scores['True Label'] == 1]['Probability'],
            name='Default (Target=1)',
            marker_color='red',
            opacity=0.6,
            nbinsx=50
        ))
        
        # Add Threshold Line
        fig.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="black", annotation_text=f"Threshold: {threshold:.2f}")
        
        # Add Background Zones
        # Green Zone (Below Threshold) -> Predicted 0 (Approve)
        fig.add_vrect(
            x0=0, x1=threshold,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Approve (Safe)", annotation_position="top left"
        )
        
        # Red Zone (Above Threshold) -> Predicted 1 (Reject)
        fig.add_vrect(
            x0=threshold, x1=1,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="Reject (Risky)", annotation_position="top right"
        )
        
        fig.update_layout(
            title=f"Credit Score Distribution & Decision Boundary (Threshold={threshold:.2f})",
            xaxis_title="Predicted Probability of Default",
            yaxis_title="Count",
            barmode='overlay',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
        )
        
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=['Repayment', 'Default']):
        """
        Plots a Plotly confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        z = cm
        x = labels
        y = labels
        
        # Text for cells
        text = [[f"Count: {cm[i][j]}<br>Rate: {cm_norm[i][j]:.2%}" for j in range(len(x))] for i in range(len(y))]
        
        fig = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            text=text,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template='plotly_white',
            width=500,
            height=500
        )
        
        return fig

    @staticmethod
    def plot_cost_curve(thresholds, costs, optimal_threshold):
        """
        Plots the business cost vs. threshold curve.
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=costs,
            mode='lines',
            name='Business Cost',
            line=dict(color='blue', width=3)
        ))
        
        # Highlight Optimal Point
        min_cost = min(costs)
        fig.add_trace(go.Scatter(
            x=[optimal_threshold],
            y=[min_cost],
            mode='markers',
            name='Optimal Threshold',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"Business Cost Optimization Curve (Optimal Threshold: {optimal_threshold:.2f})",
            xaxis_title="Probability Threshold",
            yaxis_title="Average Cost per Client",
            template='plotly_white',
            hovermode="x unified"
        )
        
        return fig
