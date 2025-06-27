#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:38:26 2025

@author: kboshomaneadmin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, fbeta_score, roc_curve
)
from sklearn.calibration import calibration_curve

# Set plot style
sns.set_theme(style="whitegrid")

def load_data():
    """Loads the corrected featured train and test datasets."""
    print("Loading corrected featured datasets...")
    try:
        train_df = pd.read_csv('train_featured.csv')
        test_df = pd.read_csv('test_featured.csv')
        
        # Ensure target is in 0/1 format
        train_df['y'] = train_df['y'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else (1 if x == 1 else 0))
        test_df['y'] = test_df['y'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'yes' else (1 if x == 1 else 0))
        
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure 'train_featured_corrected.csv' and 'test_featured_corrected.csv' exist.")
        return None, None

def get_classification_metrics(y_true, y_pred_proba, y_pred_binary):
    """Calculates a dictionary of classification metrics."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    metrics = {
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
        'PR-AUC': auc(recall, precision),
        'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0),
        'F2-Score': fbeta_score(y_true, y_pred_binary, beta=2, zero_division=0)
    }
    metrics['Gini'] = 2 * metrics['ROC-AUC'] - 1
    return metrics

def plot_metrics_table(metrics_data, title='Final Random Forest Performance'):
    """Creates and saves a plot of a table with the classification metrics."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight'); ax.axis('off')
    
    table_data = [[dataset] + [f"{value:.4f}" for value in metrics.values()] for dataset, metrics in metrics_data.items()]
    columns = ['Dataset'] + list(metrics_data['Train'].keys())
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.set_fontsize(12); table.scale(1.2, 1.2)
    
    ax.set_title(title, fontsize=16, pad=20)
    plt.savefig('final_rf_metrics_report.png', bbox_inches='tight', dpi=150)
    plt.show()

def plot_roc_pr_curves(y_train, y_train_proba, y_test, y_test_proba):
    """Plots ROC and Precision-Recall curves for train and test sets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    ax1.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {roc_auc_score(y_train, y_train_proba):.3f})")
    ax1.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_score(y_test, y_test_proba):.3f})", linestyle='--')
    ax1.plot([0, 1], [0, 1], color='grey', linestyle=':')
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.set_title('ROC Curve')
    ax1.legend()

    # Precision-Recall Curve
    prec_train, rec_train, _ = precision_recall_curve(y_train, y_train_proba)
    prec_test, rec_test, _ = precision_recall_curve(y_test, y_test_proba)
    ax2.plot(rec_train, prec_train, label=f"Train PR (AUC = {auc(rec_train, prec_train):.3f})")
    ax2.plot(rec_test, prec_test, label=f"Test PR (AUC = {auc(rec_test, prec_test):.3f})", linestyle='--')
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.suptitle('Random Forest Evaluation Curves', fontsize=18)
    plt.savefig('final_rf_evaluation_curves.png', bbox_inches='tight', dpi=150)
    plt.show()

def plot_calibration_curve_and_thresholds(y_true, y_proba):
    """Plots calibration curve and threshold analysis on a single figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')
    ax1.plot(prob_pred, prob_true, marker='o', label='Random Forest')
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    ax1.set_xlabel('Mean Predicted Probability'); ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Plot (Test Set)'); ax1.legend()

    # Threshold Analysis
    thresholds = np.linspace(0.01, 0.99, 100)
    precision_scores = [precision_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
    recall_scores = [recall_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
    f1_scores = [f1_score(y_true, y_proba >= t, zero_division=0) for t in thresholds]
    
    ax2.plot(thresholds, precision_scores, label='Precision')
    ax2.plot(thresholds, recall_scores, label='Recall')
    ax2.plot(thresholds, f1_scores, label='F1-Score', linestyle='-.')
    ax2.set_xlabel('Classification Threshold'); ax2.set_ylabel('Score'); ax2.set_title('Precision, Recall, and F1 vs. Threshold (Test Set)')
    ax2.legend()

    plt.suptitle('Random Forest Diagnostics (Test Set)', fontsize=18)
    plt.savefig('final_rf_model_diagnostics.png', bbox_inches='tight', dpi=150)
    plt.show()

def main():
    """Main function to train the final Random Forest model and evaluate its performance."""
    train_df, test_df = load_data()
    if train_df is None or test_df is None: return

    # Define all valid predictors, excluding any leaky or target-related columns.
    valid_predictors = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
        'loan', 'contact', 'day', 'month', 'campaign', 'pdays',
        'previous', 'poutcome', 'balance_to_age_ratio', 'total_liabilities', 
        'is_student_or_retired', 'is_high_season', 'was_previously_contacted'
    ]
    
    target = 'y'
    final_feature_list = [col for col in valid_predictors if col in train_df.columns]
    
    print(f"\nTraining model with {len(final_feature_list)} features.")
    
    X_train = train_df[final_feature_list]
    y_train = train_df[target]
    X_test = test_df[final_feature_list]
    y_test = test_df[target]

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ], remainder='passthrough')

    print("\nTraining the final Random Forest model on the full training dataset...")
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    final_pipeline.fit(X_train, y_train)
    print("Training complete.")

    print("Generating predictions...")
    y_train_proba = final_pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    print("Calculating final performance metrics...")
    all_metrics = {
        'Train': get_classification_metrics(y_train, y_train_proba, y_train_pred),
        'Test': get_classification_metrics(y_test, y_test_proba, y_test_pred)
    }
    
    # Generate all plots
    plot_metrics_table(all_metrics)
    plot_roc_pr_curves(y_train, y_train_proba, y_test, y_test_proba)
    plot_calibration_curve_and_thresholds(y_test, y_test_proba)

    print("\n--- Final evaluation for Random Forest complete. All plots have been saved. ---")

if __name__ == '__main__':
    main()
