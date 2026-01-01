# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

def plot_feature_importance(feature_importance_df, top_n=15):
    """Plot feature importance chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = feature_importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features Driving Churn')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(y_true, y_pred):
    """Plot confusion matrix as heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                           name=f'ROC curve (AUC = {roc_auc:.2f})',
                           line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                           name='Random', line=dict(color='navy', dash='dash')))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=600, height=500
    )
    return fig

def plot_metrics_comparison(metrics_dict):
    """Plot evaluation metrics comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
    bars = ax.bar(metrics_names, metrics_values, color=colors)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig