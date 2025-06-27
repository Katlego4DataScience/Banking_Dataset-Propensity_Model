#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 13:51:58 2025

@author: kboshomaneadmin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_new_split():
    """
    Loads the original, full dataset and performs a correct, stratified
    train/test split, saving the new sets to CSV files.
    """
    # --- 1. Define File Paths ---
    # IMPORTANT: Assumes the original, full dataset is the 'train.csv'
    # and uses a semicolon separator. Please update if your file is different.
    original_data_path = 'train.csv'
    
    new_train_path = 'train_new.csv'
    new_test_path = 'test_new.csv'
    
    # --- 2. Load Original Data ---
    print(f"Attempting to load original dataset from '{original_data_path}'...")
    if not os.path.exists(original_data_path):
        print(f"\n[Error] The file '{original_data_path}' was not found.")
        print("This script requires the original, complete dataset to create a valid split.")
        print("Please place the full dataset in the directory and ensure the filename is correct.")
        return

    try:
        df = pd.read_csv(original_data_path, sep=';')
        print("Original dataset loaded successfully.")
        print(f"Full dataset shape: {df.shape}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- 3. Perform Stratified Train/Test Split ---
    # We will split the data into 80% for training and 20% for testing.
    # 'stratify=df['y']' is crucial. It ensures that the distribution of the 
    # target variable ('y') is the same in both the train and test sets.
    
    print("\nPerforming stratified 80/20 train/test split...")
    
    # Check if target column 'y' exists
    if 'y' not in df.columns:
        print("[Error] Target column 'y' not found in the dataset.")
        return
        
    X = df.drop(columns=['y'])
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        random_state=42, 
        stratify=y
    )
    
    # Re-combine features and target for saving
    train_df_new = pd.concat([X_train, y_train], axis=1)
    test_df_new = pd.concat([X_test, y_test], axis=1)
    
    print("Split complete.")
    print(f"New training set shape: {train_df_new.shape}")
    print(f"New test set shape: {test_df_new.shape}")
    
    # Verify stratification
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    print("\nVerifying target distribution:")
    print(f"Train set 'yes' proportion: {train_dist.get('yes', 0):.4f}")
    print(f"Test set 'yes' proportion:  {test_dist.get('yes', 0):.4f}")


    # --- 4. Save the New Datasets ---
    # The new files will be saved with a standard comma separator.
    print(f"\nSaving new training set to '{new_train_path}'")
    train_df_new.to_csv(new_train_path, index=False)
    
    print(f"Saving new test set to '{new_test_path}'")
    test_df_new.to_csv(new_test_path, index=False)
    
    print("\n--- Process Complete ---")
    print("You should now re-run the entire pipeline starting from the feature engineering script,")
    print(f"updating it to use '{new_train_path}' and '{new_test_path}' as the inputs.")


if __name__ == '__main__':
    create_new_split()
