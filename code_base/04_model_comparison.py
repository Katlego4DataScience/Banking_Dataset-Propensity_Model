#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:24:46 2025

@author: kboshomaneadmin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Set plot style
sns.set_theme(style="whitegrid")

def load_data():
    """Loads all necessary datasets."""
    print("Loading datasets...")
    try:
        train_featured = pd.read_csv('train_featured.csv')
        train_woe = pd.read_csv('train_woe.csv')
        
        # Convert target in featured data as well
        train_featured['y'] = train_featured['y'].apply(lambda x: 1 if x == 'yes' else 0)
        
        return train_featured, train_woe
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure files exist.")
        return None, None

def main():
    """
    Main function to run the model comparison pipeline.
    """
    train_featured, train_woe = load_data()
    
    if train_featured is None or train_woe is None:
        return

    # --- 1. Define Datasets and Target ---
    target = 'y'
    
    # Define all known leaky features to be removed, including the target proxy 'y_numeric'
    leaky_features = ['duration', 'y_numeric']
    print(f"\nEnsuring leaky features are removed: {leaky_features}")
    
    # Dataset for Logistic Regression (WOE transformed)
    # The WOE script replaces original columns, so we drop the original feature name.
    X_woe = train_woe.drop(columns=[target] + leaky_features, errors='ignore')
    y_woe = train_woe[target]
    
    # Dataset for Tree-Based Models (feature engineered)
    X_featured = train_featured.drop(columns=[target] + leaky_features, errors='ignore')
    y_featured = train_featured[target]

    # --- 2. Setup Preprocessing for Tree-Based Models ---
    # Identify numerical and categorical features in the 'featured' dataset
    numerical_features = X_featured.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_featured.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create a preprocessor object using ColumnTransformer
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like binary flags) as they are
    )

    # --- 3. Define Models and Pipelines ---
    # We use pipelines to chain preprocessing and modeling steps
    
    # Pipeline for Logistic Regression (no preprocessing needed for WOE data)
    lr_pipeline = Pipeline([
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear'))
    ])

    # Pipeline for Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor_tree),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    # Pipeline for Scikit-learn's Gradient Boosting
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor_tree),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

    # --- 4. Run Cross-Validation ---
    print("\nStarting 5-fold cross-validation for each model...")
    
    models = {
        "Logistic Regression (WOE)": (lr_pipeline, X_woe, y_woe),
        "Random Forest": (rf_pipeline, X_featured, y_featured),
        "Gradient Boosting": (gb_pipeline, X_featured, y_featured)
    }
    
    results = {}
    
    # Define the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, (pipeline, X, y) in models.items():
        print(f"Evaluating {name}...")
        # Use roc_auc as the scoring metric
        scores = cross_val_score(pipeline, X, y, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
        results[name] = scores
        print(f"{name} | Mean ROC-AUC: {np.mean(scores):.4f} | Std Dev: {np.std(scores):.4f}")

    # --- 5. Visualize and Compare Results ---
    print("\nVisualizing model comparison results...")
    
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=results_df, palette='viridis', showmeans=True)
    plt.title('Model Comparison - Cross-Validated ROC-AUC Scores', fontsize=18)
    plt.ylabel('ROC-AUC Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    print("\n--- Model comparison complete. ---")
    print("The model with the highest and most stable ROC-AUC score is the best candidate.")

if __name__ == '__main__':
    main()
