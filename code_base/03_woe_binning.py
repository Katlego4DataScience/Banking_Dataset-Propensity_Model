#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:50:08 2025

@author: kboshomaneadmin
"""

import pandas as pd
from optbinning import OptimalBinning, BinningProcess
import warnings

# Suppress convergence warnings from optbinning if they occur
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """
    Main function to perform WOE binning and transformation on all features.
    Feature selection will be handled in a later step.
    """
    # Note: You may need to install optbinning: pip install optbinning
    
    # --- 1. Load Data ---
    print("Loading featured datasets...")
    try:
        train_df = pd.read_csv('train_featured.csv')
        test_df = pd.read_csv('test_featured.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'train_featured.csv' and 'test_featured.csv' exist.")
        return
        
    # --- 2. Define Variables ---
    # Define the target variable and convert it to binary (0/1)
    target = 'y'
    train_df[target] = train_df[target].apply(lambda x: 1 if x == 'yes' else 0)
    test_df[target] = test_df[target].apply(lambda x: 1 if x == 'yes' else 0)

    # Define all potential predictor variables (original and engineered)
    variable_names = [
        'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 
        'previous', 'poutcome', 'duration_per_contact', 'balance_to_age_ratio',
        'total_liabilities', 'is_student_or_retired', 'is_high_season',
        'was_previously_contacted'
    ]

    # Filter out any variables that might not be in the dataframe
    variable_names = [v for v in variable_names if v in train_df.columns]
    
    # --- 3. Fit and Apply WOE Transformation ---
    print("\nFitting Binning Process on all features and applying WOE transformation...")
    
    # Initialize the binning process with all variables
    woe_binning_process = BinningProcess(
        variable_names=variable_names,
        # Define special codes for pdays to handle the '-1' value correctly
        special_codes={'pdays': [-1]}
    )
    
    # Fit on the training data
    X_train = train_df[variable_names]
    y_train = train_df[target]
    woe_binning_process.fit(X_train, y_train)
    
    # Transform both train and test data. 
    # This returns a dataframe where original columns are replaced by WOE values.
    train_woe = woe_binning_process.transform(train_df, metric='woe')
    test_woe = woe_binning_process.transform(test_df, metric='woe')
    
    # Ensure the target column is present in the final dataframes
    train_woe[target] = train_df[target]
    test_woe[target] = test_df[target]
    
    # --- 4. Save Transformed Data ---
    train_output_path = 'train_woe.csv'
    test_output_path = 'test_woe.csv'
    
    train_woe.to_csv(train_output_path, index=False)
    test_woe.to_csv(test_output_path, index=False)
    
    print(f"\nWOE transformed data saved to '{train_output_path}' and '{test_output_path}'.")
    print("\nFirst 5 rows of the WOE-transformed training data (showing transformed columns):")
    
    # FIXED: The .transform() method replaces the original columns with WOE values.
    # We select the original variable names to see their new WOE values.
    print(train_woe[variable_names + [target]].head())


if __name__ == '__main__':
    main()
