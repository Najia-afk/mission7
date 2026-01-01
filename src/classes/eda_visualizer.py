import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats

class EDAVisualizer:
    """
    Class for interactive Exploratory Data Analysis visualizations using Plotly.
    Inspired by mission 5 style.
    """
    
    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, target_col: str = 'TARGET'):
        """
        Plots the distribution of the target variable using Plotly.
        """
        counts = df[target_col].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=['#1f77b4', '#d62728']  # Blue for 0, Red for 1
        )])
        
        fig.update_layout(
            title_text='Target Variable Distribution',
            xaxis_title=target_col,
            yaxis_title='Count',
            template='plotly_white'
        )
        return fig

    @staticmethod
    def plot_numerical_distribution(df: pd.DataFrame, columns: list = None, max_samples: int = 5000):
        """
        Creates interactive box plots and histograms for numerical columns.
        Uses a dropdown to switch between columns.
        
        NOTE: Data is sampled to max_samples to reduce HTML file size.
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
            
        # Filter out columns with no data
        columns = [col for col in columns if df[col].count() > 0]
        
        if not columns:
            print("No valid numerical columns found.")
            return None
        
        # Sample data for visualization (keeps HTML file small)
        if len(df) > max_samples:
            df_sample = df.sample(n=max_samples, random_state=42)
        else:
            df_sample = df

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Box Plot", "Distribution"),
            horizontal_spacing=0.1
        )

        # Add traces for the first column
        first_col = columns[0]
        EDAVisualizer._add_traces_for_col(fig, df, first_col, visible=True)

        # Create dropdown buttons
        buttons = []
        for col in columns:
            # Calculate new data for this column
            # We need to reconstruct the update args. 
            # Note: Plotly dropdowns with 'update' are tricky with subplots.
            # Simpler approach: Create one trace per column and toggle visibility.
            pass
            
        # RE-STRATEGY: The dropdown approach in Plotly can be complex with subplots.
        # Let's stick to the mission 5 implementation style which adds ALL traces 
        # and then toggles visibility via buttons.
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Box Plot", "Distribution"),
            specs=[[{"type": "box"}, {"type": "histogram"}]]
        )
        
        # Add traces for all columns, but set visible=False for all except first
        for i, col in enumerate(columns):
            visible = (i == 0)
            EDAVisualizer._add_traces_for_col(fig, df_sample, col, visible)

        # Create buttons
        buttons = []
        for i, col in enumerate(columns):
            # Visibility array: 2 traces per column (box + hist)
            # We want to turn on traces 2*i and 2*i+1
            visibility = [False] * (2 * len(columns))
            visibility[2*i] = True
            visibility[2*i+1] = True
            
            button = dict(
                label=col,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Distribution analysis for {col}"}]
            )
            buttons.append(button)

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.5, xanchor="center",
                y=1.15, yanchor="top"
            )],
            title_text=f"Distribution analysis for {first_col}",
            template='plotly_white',
            height=500
        )
        
        return fig

    @staticmethod
    def _add_traces_for_col(fig, df, col, visible):
        data = df[col].dropna()
        
        # Box Plot
        fig.add_trace(
            go.Box(y=data, name=col, boxpoints='outliers', visible=visible, marker_color='#636EFA'),
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data, name=col, visible=visible, marker_color='#EF553B', opacity=0.7),
            row=1, col=2
        )

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, columns: list = None):
        """
        Plots a correlation heatmap using Plotly.
        """
        if columns:
            df_corr = df[columns].corr()
        else:
            df_corr = df.select_dtypes(include=['number']).corr()
            
        fig = px.imshow(
            df_corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Feature Correlation Matrix"
        )
        return fig
