#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:21:40 2025

@author: kboshomaneadmin
"""

import pandas as pd
import numpy as np

def load_data(filepath, separator=';'):
    """
    Loads and validates the training dataset, handling common separator and whitespace issues.
    """
    try:
        print(f"Attempting to load data from {filepath} with separator '{separator}'...")
        df = pd.read_csv(filepath, sep=separator)

        # If only one column is detected, the separator is likely wrong. Retry with a comma.
        if len(df.columns) == 1 and separator == ';':
            print("Warning: Only one column was detected. Retrying with a comma (',') separator.")
            df = pd.read_csv(filepath, sep=',')

        # Clean column names to remove leading/trailing whitespace, which can cause KeyErrors.
        df.columns = df.columns.str.strip()

        print("Columns found after loading and cleaning:", df.columns.tolist())

        # Now, check for 'y' again after cleaning.
        if 'y' not in df.columns:
            print(f"CRITICAL ERROR: Column 'y' is still not found.")
            print("Please check your CSV file's header for the exact spelling and case of the target column.")
            return None

        print("Data loaded and validated successfully.")
        # Convert target variable 'y' to a more intuitive 0/1 format for correlation analysis
        df['y_numeric'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please ensure it is in the correct directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def engineer_features(df):
    """
    Creates new features based on existing data to improve model performance.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with new, engineered features.
    """
    print("Starting feature engineering...")
    
    # --- 1. Contact Efficiency Ratios ---
    # Add a small epsilon to avoid division by zero, though campaign is always >= 1
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1e-6)
    
    # --- 2. Financial Health Indicators ---
    # Add a small epsilon to age in case any age is 0
    df['balance_to_age_ratio'] = df['balance'] / (df['age'] + 1e-6)
    
    # Create the total_liabilities feature
    conditions = [
        (df['housing'] == 'no') & (df['loan'] == 'no'),
        (df['housing'] == 'yes') & (df['loan'] == 'no'),
        (df['housing'] == 'no') & (df['loan'] == 'yes'),
        (df['housing'] == 'yes') & (df['loan'] == 'yes')
    ]
    choices = ['No Loans', 'Housing Loan Only', 'Personal Loan Only', 'Both Loans']
    df['total_liabilities'] = np.select(conditions, choices, default='Unknown')

    # --- 3. High-Propensity Group Flags ---
    df['is_student_or_retired'] = df['job'].isin(['student', 'retired']).astype(int)
    
    high_season_months = ['mar', 'apr', 'sep', 'oct', 'dec']
    df['is_high_season'] = df['month'].isin(high_season_months).astype(int)
    
    # --- 4. Previous Contact Flag ---
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)
    
    print("Feature engineering complete. New features added:")
    new_features = [
        'duration_per_contact', 
        'balance_to_age_ratio', 
        'total_liabilities', 
        'is_student_or_retired', 
        'is_high_season', 
        'was_previously_contacted'
    ]
    print(new_features)
    
    return df

def main():
    """
    Main function to load data, apply feature engineering, and save the results.
    """
    # Define file paths using the provided locations
    train_filepath = '/home/UNIZA/kboshomaneadmin/Probability to covert/train_new.csv'
    test_filepath = '/home/UNIZA/kboshomaneadmin/Probability to covert/test_new.csv' 
    
    # Load datasets
    train_df = load_data(train_filepath)
    test_df = load_data(test_filepath)
    
    if train_df is not None and test_df is not None:
        # Apply feature engineering to both training and testing sets
        train_df_featured = engineer_features(train_df)
        test_df_featured = engineer_features(test_df)
        
        # Define output file paths
        train_output_path = 'train_featured.csv'
        test_output_path = 'test_featured.csv'
        
        # Save the new dataframes to CSV files
        train_df_featured.to_csv(train_output_path, index=False)
        test_df_featured.to_csv(test_output_path, index=False)
        
        print(f"\nProcessed data saved to '{train_output_path}' and '{test_output_path}'.")
        print("\nFirst 5 rows of the new training data:")
        print(train_df_featured.head())

if __name__ == '__main__':
    main()
