import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ks_2samp, chi2_contingency
import mlflow
import warnings
warnings.filterwarnings('ignore')


def compute_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                 numeric_features: list, categorical_features: list):
    """
    Compute data drift using KS test (numeric) and Chi-square (categorical).
    Returns dictionary with p-values for each feature.
    NaN p-values indicate test failure (data quality issue).
    """
    drift_results = {}
    failed_tests = {}
    
    # Numeric features: Kolmogorov-Smirnov test
    for col in numeric_features:
        try:
            statistic, p_value = ks_2samp(reference_data[col], current_data[col])
            drift_results[col] = p_value
        except Exception as e:
            drift_results[col] = float('nan')  # Mark as inconclusive (NaN p-value)
            failed_tests[col] = str(e)
    
    # Categorical features: Chi-square test
    for col in categorical_features:
        try:
            ref_counts = reference_data[col].value_counts()
            curr_counts = current_data[col].value_counts()
            
            # Align indices
            all_cats = set(ref_counts.index) | set(curr_counts.index)
            ref_counts = ref_counts.reindex(all_cats, fill_value=0)
            curr_counts = curr_counts.reindex(all_cats, fill_value=0)
            
            # Chi-square test
            chi2, p_value, _, _ = chi2_contingency([ref_counts, curr_counts])
            drift_results[col] = p_value
        except Exception as e:
            drift_results[col] = float('nan')  # Mark as inconclusive (NaN p-value)
            failed_tests[col] = str(e)
    
    # Print inconclusive tests
    if failed_tests:
        print("\nINCONCLUSIVE TESTS (failed statistical test - data quality issue):")
        for col, error in sorted(failed_tests.items()):
            print(f"   {col}: {error}")
        print()
    
    return drift_results


def plot_drift_summary(drift_results: dict):
    """
    Plot p-values for all features with threshold line.
    Green = No drift (p >= 0.05)
    Red = Drift detected (p < 0.05)
    Gray = Inconclusive test (p = NaN - data quality issue)
    """
    import numpy as np
    
    fig = go.Figure()
    
    features = list(drift_results.keys())
    p_values = list(drift_results.values())
    
    # Color coding: green (no drift), red (drift), gray (inconclusive)
    colors = []
    for p in p_values:
        if pd.isna(p):
            colors.append('gray')  # Inconclusive test
        elif p < 0.05:
            colors.append('red')  # Drift detected
        else:
            colors.append('green')  # No drift
    
    # Replace NaN with 0.5 for visualization (will show as gray bar)
    p_values_plot = [0.5 if pd.isna(p) else p for p in p_values]
    
    # Create hover text with explanation
    hover_text = []
    for feature, p, color in zip(features, p_values, colors):
        if color == 'gray':
            hover_text.append(f"{feature}<br>Status: Inconclusive (Test Failed)<br>p-value: NaN")
        elif color == 'red':
            hover_text.append(f"{feature}<br>Status: DRIFT DETECTED<br>p-value: {p:.4f}")
        else:
            hover_text.append(f"{feature}<br>Status: No Drift<br>p-value: {p:.4f}")
    
    fig.add_trace(go.Bar(
        x=features, 
        y=p_values_plot, 
        marker_color=colors,
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.add_hline(
        y=0.05, 
        line_dash="dash", 
        line_color="black", 
        annotation_text="drift threshold (p=0.05)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Data Drift Summary (All Features)<br><sub>Green=✅ No Drift | Red=🔴 Drift | Gray=⚠️ Inconclusive (Test Failed)</sub>",
        xaxis_title="Features",
        yaxis_title="P-Value",
        height=450,
        hovermode='x unified'
    )
    
    return fig
    return fig


def plot_distributions_interactive(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                                  drift_results: dict, numeric_features: list,
                                  max_samples: int = 5000):
    """
    Create interactive distributions figure with dropdown to select numeric feature.
    All features shown alphabetically in dropdown menu.
    Gray = Inconclusive test (p = NaN)
    
    NOTE: Data is sampled to max_samples to reduce HTML file size.
    Statistical tests use full data, but visualizations use samples.
    """
    sorted_features = sorted(numeric_features)
    fig = go.Figure()
    
    # Sample data for visualization (keeps HTML file small)
    if len(reference_data) > max_samples:
        ref_sample = reference_data.sample(n=max_samples, random_state=42)
    else:
        ref_sample = reference_data
        
    if len(current_data) > max_samples:
        curr_sample = current_data.sample(n=max_samples, random_state=42)
    else:
        curr_sample = current_data
    
    # Add histograms for all features (hidden initially, shown via dropdown)
    for i, feature in enumerate(sorted_features):
        p_val = drift_results.get(feature, 1.0)
        
        # Determine status for display
        if pd.isna(p_val):
            status_str = "⚠️ INCONCLUSIVE (Test Failed)"
            p_display = "NaN"
        elif p_val < 0.05:
            status_str = "🔴 DRIFT DETECTED"
            p_display = f"{p_val:.4f}"
        else:
            status_str = "✅ No Drift"
            p_display = f"{p_val:.4f}"
        
        # Reference histogram (using sampled data)
        fig.add_trace(go.Histogram(
            x=ref_sample[feature],
            name="Reference",
            opacity=0.6,
            nbinsx=30,
            histnorm='percent',
            visible=(i == 0),  # First feature visible by default
            showlegend=(i == 0),
            marker_color='rgba(31, 119, 180, 0.7)'
        ))
        
        # Current histogram (using sampled data)
        fig.add_trace(go.Histogram(
            x=curr_sample[feature],
            name="Current",
            opacity=0.6,
            nbinsx=30,
            histnorm='percent',
            visible=(i == 0),  # First feature visible by default
            showlegend=(i == 0),
            marker_color='rgba(255, 127, 14, 0.7)'
        ))
    
    # Create dropdown buttons for each feature
    buttons = []
    for i, feature in enumerate(sorted_features):
        p_val = drift_results.get(feature, 1.0)
        
        # Determine status for display
        if pd.isna(p_val):
            status_str = "⚠️ INCONCLUSIVE (Test Failed)"
            p_display = "NaN"
        elif p_val < 0.05:
            status_str = "🔴 DRIFT DETECTED"
            p_display = f"{p_val:.4f}"
        else:
            status_str = "✅ No Drift"
            p_display = f"{p_val:.4f}"
        
        # Create visibility array: show only traces for this feature
        visibility = [False] * len(fig.data)
        visibility[2*i] = True      # Reference histogram
        visibility[2*i + 1] = True  # Current histogram
        
        buttons.append(
            dict(
                label=feature,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{feature} Distribution ({status_str})<br><sub>p-value: {p_display}</sub>", 
                     "xaxis.title": feature, 
                     "yaxis.title": "Percentage (%)"}
                ]
            )
        )
    
    # Set initial title with status
    first_p_val = drift_results.get(sorted_features[0], 1.0)
    if pd.isna(first_p_val):
        initial_status = "⚠️ INCONCLUSIVE (Test Failed)"
        initial_p_display = "NaN"
    elif first_p_val < 0.05:
        initial_status = "🔴 DRIFT DETECTED"
        initial_p_display = f"{first_p_val:.4f}"
    else:
        initial_status = "✅ No Drift"
        initial_p_display = f"{first_p_val:.4f}"
    
    first_feature = sorted_features[0]
    
    fig.update_layout(
        title=f"{first_feature} Distribution ({initial_status})<br><sub>p-value: {initial_p_display}</sub>",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        xaxis_title=first_feature,
        yaxis_title="Percentage (%)",
        height=500,
        barmode='overlay'
    )
    
    return fig


def plot_statistics(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                   drift_results: dict, numeric_features: list):
    """Plot statistics comparison table."""
    stats_data = []
    sorted_features = sorted(numeric_features)
    
    for col in sorted_features:
        stats_data.append({
            'Feature': col,
            'Ref Mean': reference_data[col].mean(),
            'Curr Mean': current_data[col].mean(),
            'Ref Std': reference_data[col].std(),
            'Curr Std': current_data[col].std(),
            'P-Value': drift_results[col]
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(stats_df.columns), fill_color='paleturquoise'),
        cells=dict(values=[stats_df[col] for col in stats_df.columns], fill_color='lavender')
    )])
    fig.update_layout(title="Statistics Comparison (All Numeric Features)", height=500)
    return fig


def plot_percentage_change(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                          numeric_features: list):
    """Plot percentage change heatmap."""
    sorted_features = sorted(numeric_features)
    pct_change = []
    
    for col in sorted_features:
        ref_mean = reference_data[col].mean()
        curr_mean = current_data[col].mean()
        pct = ((curr_mean - ref_mean) / abs(ref_mean) * 100) if ref_mean != 0 else 0
        pct_change.append(pct)
    
    fig = go.Figure(data=go.Heatmap(
        z=[pct_change],
        x=sorted_features,
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(
        title="Percentage Change from Reference to Current",
        height=300,
        xaxis_title="Features"
    )
    return fig


def analyze_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame,
                 numeric_features: list, categorical_features: list):
    """Run complete drift analysis with separate visualizations."""
    
    print("Step 11: Data Drift Detection")
    print(f"Reference set: {reference_data.shape[0]:,} rows")
    print(f"Current set: {current_data.shape[0]:,} rows\n")
    
    # Compute drift
    drift_results = compute_drift(reference_data, current_data, numeric_features, categorical_features)
    
    # Identify drifted features
    drifted = [f for f, p in drift_results.items() if p < 0.05]
    print(f"Features with drift (p < 0.05): {len(drifted)}/{len(drift_results)}")
    if drifted:
        for f in sorted(drifted):
            print(f"  • {f}: p-value = {drift_results[f]:.4f}")
    
    # Plot 1: Drift Summary
    print("\n📊 Drift Summary (All Features)")
    fig1 = plot_drift_summary(drift_results)
    fig1.show()
    
    # Plot 2: Distribution Comparisons (with dropdown)
    print("\n📈 Distribution Comparisons (dropdown to select feature)")
    fig2 = plot_distributions_interactive(reference_data, current_data, drift_results, numeric_features)
    fig2.show()
    
    # Plot 3: Statistics Comparison
    print("\n📋 Statistics Comparison (All Numeric Features)")
    fig3 = plot_statistics(reference_data, current_data, drift_results, numeric_features)
    fig3.show()
    
    # Plot 4: Percentage Change
    print("\n🔥 Percentage Change (All Numeric Features)")
    fig4 = plot_percentage_change(reference_data, current_data, numeric_features)
    fig4.show()
    
    # MLflow logging
    drifted_count = sum(1 for p in drift_results.values() if p < 0.05)
    
    with mlflow.start_run(run_name="Step11_Drift"):
        mlflow.log_param("reference_rows", reference_data.shape[0])
        mlflow.log_param("current_rows", current_data.shape[0])
        mlflow.log_metric("drifted_features", drifted_count)
        mlflow.set_tag("step", "11_drift")
    
    print(f"\n✅ Step 11 complete: {drifted_count} features with drift detected")
    
    return drift_results
