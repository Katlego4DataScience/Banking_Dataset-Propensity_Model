#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:26:20 2025

@author: kboshomaneadmin
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_diagnosis_table(stats, title='Data Leakage Diagnosis Report'):
    """
    Creates and saves a plot of a table with the diagnosis statistics.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = list(stats.items())
    
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"], loc='center', cellLoc='center')
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color the status cell based on findings
    if stats['Status'] == 'Leakage Detected!':
        table.get_celld()[(1, 1)].set_facecolor("#ff4d4d")
        table.get_celld()[(1, 1)].set_text_props(color='white')
    else:
        table.get_celld()[(1, 1)].set_facecolor("#90ee90")
        table.get_celld()[(1, 1)].set_text_props(color='black')
        
    ax.set_title(title, fontsize=16, pad=20)
    plt.savefig('leakage_diagnosis_report.png', bbox_inches='tight', dpi=150)
    plt.show()

def check_for_overlap(train_filepath='train_new.csv', test_filepath='test_new.csv'):
    """
    Loads train and test datasets and checks for overlapping rows to diagnose data leakage.
    """
    print("--- Data Leakage Investigation ---")
    
    # --- 1. Load Data ---
    print(f"Loading training data from: {train_filepath}")
    print(f"Loading testing data from: {test_filepath}")
    try:
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)
    except FileNotFoundError as e:
        print(f"\nError: {e}. Please ensure both featured CSV files exist.")
        return

    print("\nDatasets loaded successfully.")
    
    # --- 2. Identify Identifier Columns ---
    identifier_cols = train_df.columns.tolist()
    potential_targets = ['y', 'y_numeric']
    leaky_cols = ['duration', 'duration_per_contact']
    
    for col in potential_targets + leaky_cols:
        if col in identifier_cols:
            identifier_cols.remove(col)
            
    print(f"\nUsing {len(identifier_cols)} columns as unique identifiers for a row.")

    # --- 3. Check for Overlap using an Inner Merge ---
    print("\nPerforming an inner merge to find overlapping rows...")
    merged_df = pd.merge(train_df, test_df, on=identifier_cols, how='inner')

    # --- 4. Report Findings ---
    overlap_count = len(merged_df)
    test_size = len(test_df)
    overlap_percentage = (overlap_count / test_size) * 100 if test_size > 0 else 0

    diagnosis_stats = {
        "Status": "Leakage Detected!" if overlap_count > 0 else "No Leakage Detected",
        "Training Set Shape": str(train_df.shape),
        "Test Set Shape": str(test_df.shape),
        "Overlapping Rows": overlap_count,
        "Overlap Percentage": f"{overlap_percentage:.2f}%"
    }

    print("\n--- DIAGNOSIS ---")
    plot_diagnosis_table(diagnosis_stats)

    if overlap_count > 0:
        print(f"\n[!!] CRITICAL: {overlap_count} rows ({overlap_percentage:.2f}%) of the test set are also in the training set.")
        print("This is a critical issue causing the perfect model scores.")
        print("\nFirst 5 overlapping rows found:")
        print(merged_df[identifier_cols].head())
    else:
        print("\n[✓] No direct row overlap was found.")
        
    print("\n--- End of Investigation ---")
    print("The 'leakage_diagnosis_report.png' has been saved.")


if __name__ == '__main__':
    check_for_overlap()
