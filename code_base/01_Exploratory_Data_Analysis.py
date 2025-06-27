#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:40:00 2025

@author: kboshomaneadmin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pandas.api.types import CategoricalDtype

# --- Configuration for plotting ---
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')


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

def analyze_target_variable(df, output_path='charts'):
    """
    Analyzes and visualizes the distribution of the target variable.
    """
    if 'y' not in df.columns:
        print("Target variable 'y' not found in the dataframe.")
        return

    print("\n--- Target Variable Analysis ---")
    target_percentage = df['y'].value_counts(normalize=True) * 100
    
    print("Distribution of 'y' (Subscribed to Term Deposit):")
    print(f"{target_percentage}")

    plt.figure(figsize=(8, 6))
    sns.countplot(x='y', data=df, palette="pastel", order=['no', 'yes'])
    plt.title('Distribution of Target Variable (y)', fontsize=16)
    plt.xlabel('Subscribed to Term Deposit?', fontsize=12)
    plt.ylabel('Number of Clients', fontsize=12)
    plt.savefig(f'{output_path}/target_variable_distribution.png')
    plt.show()
    print("\nInsight: The dataset is imbalanced. 'No' subscriptions significantly outnumber 'Yes' subscriptions.")
    print("This is important for model evaluation; accuracy alone might be misleading.")


def analyze_numerical_features(df, output_path='charts'):
    """
    Performs analysis on numerical features: summary, distributions, and relationship with target.
    """
    print("\n--- Numerical Feature Analysis (Boxplots) ---")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_cols.remove('y_numeric')  # Exclude the numeric target from this list of features
    print(f"Numerical columns identified: {numerical_cols}")
    
    # Box plots to see relationship with target variable
    print("\nPlotting numerical features against the target variable...")
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='y', y=col, data=df, palette="vlag", order=['no', 'yes'])
        plt.title(f'{col.capitalize()} vs. Subscription Status', fontsize=16)
        plt.xlabel('Subscribed to Term Deposit?', fontsize=12)
        plt.ylabel(col.capitalize(), fontsize=12)
        plt.savefig(f'{output_path}/boxplot_{col}_vs_y.png')
        plt.show()
    print("\nInsight: The 'duration' of the last call appears to be a strong indicator. Clients who subscribed had much longer call durations.")


def enhanced_numerical_distribution_plots(df, output_path='charts'):
    """
    NEW: Creates overlaid histograms to compare numerical feature distributions for each target class.
    """
    print("\n--- Enhanced Numerical Distribution Analysis ---")
    numerical_cols = ['age', 'balance', 'duration', 'campaign']
    
    print("Plotting overlaid distributions for key numerical features...")
    for col in numerical_cols:
        plt.figure(figsize=(12, 7))
        # Use histplot with a hue to separate distributions by the target variable
        sns.histplot(data=df, x=col, hue='y', kde=True, common_norm=False, palette='viridis', alpha=0.6)
        plt.title(f'Distribution of {col.capitalize()} by Subscription Status', fontsize=16)
        plt.xlabel(col.capitalize(), fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.savefig(f'{output_path}/histplot_hue_{col}.png')
        plt.show()
    print("\nInsight: The overlaid plots confirm that the distribution for 'duration' is vastly different for subscribers vs. non-subscribers.")


def analyze_categorical_features(df, output_path='charts'):
    """
    Performs analysis on categorical features by visualizing their relationship with the target.
    """
    print("\n--- Categorical Feature Analysis ---")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('y')  # Remove target variable from this list
    
    print(f"Categorical columns identified: {categorical_cols}")
    
    # Plot count plots for each categorical feature against the target
    print("\nPlotting categorical features against the target variable...")
    for col in categorical_cols:
        plt.figure(figsize=(12, 7))
        # Create a plot showing the proportion of 'yes' vs 'no' for each category
        prop_df = df.groupby(col)['y'].value_counts(normalize=True).unstack().fillna(0)
        
        # Special ordering for the 'month' column
        if col == 'month':
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            prop_df = prop_df.reindex([m for m in month_order if m in prop_df.index])
        
        prop_df.plot(kind='bar', stacked=True, color=['#A9CCE3', '#196F3D'])
        
        plt.title(f'Proportion of Subscriptions by {col.capitalize()}', fontsize=16)
        plt.xlabel(col.capitalize(), fontsize=12)
        plt.ylabel('Proportion of Clients', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Subscribed?')
        plt.tight_layout()
        plt.savefig(f'{output_path}/proportion_{col}_vs_y.png')
        plt.show()

def analyze_campaign_features(df, output_path='charts'):
    """
    NEW: Performs a dedicated analysis of campaign-related features.
    """
    print("\n--- Campaign Feature Analysis ---")
    
    # Analysis of 'campaign' (number of contacts during this campaign)
    df_campaign = df.copy()
    df_campaign['campaign_group'] = np.where(df_campaign['campaign'] > 5, '6+', df_campaign['campaign'].astype(str))
    
    # --- CORRECTED FIX STARTS HERE ---
    # Define the desired order for the categories.
    campaign_order = ['1', '2', '3', '4', '5', '6+']
    
    # Create a Pandas Categorical data type with the specified order.
    campaign_cat_type = CategoricalDtype(categories=campaign_order, ordered=True)
    
    # Apply this ordered categorical type to the column.
    df_campaign['campaign_group'] = df_campaign['campaign_group'].astype(campaign_cat_type)
    # --- CORRECTED FIX ENDS HERE ---
    
    plt.figure(figsize=(10, 6))
    
    # Now, plot the data. We have removed the incorrect 'order' parameter.
    # Seaborn will automatically use the correct order defined in the DataFrame's column.
    sns.lineplot(data=df_campaign, x='campaign_group', y='y_numeric', errorbar=None, marker='o', color='purple')
    
    plt.title('Conversion Rate by Number of Campaign Contacts', fontsize=16)
    plt.xlabel('Number of Contacts in This Campaign', fontsize=12)
    plt.ylabel('Conversion Rate (y=1)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'{output_path}/conversion_by_campaign_contacts.png')
    plt.show()
    print("\nInsight: Conversion rate tends to decrease after the first few contacts.")
    
    # Analysis of 'pdays' (days since last contact)
    pdays_df = df[df['pdays'] != -1]  # Filter for clients who were previously contacted
    plt.figure(figsize=(12, 7))
    sns.histplot(data=pdays_df, x='pdays', hue='y', kde=True, common_norm=False, bins=40, palette='magma')
    plt.title('Distribution of Days Since Last Contact (pdays)', fontsize=16)
    plt.xlabel('Days Since Last Contact', fontsize=12)
    plt.ylabel('Count of Clients', fontsize=12)
    plt.savefig(f'{output_path}/pdays_distribution.png')
    plt.show()
    print("\nInsight: For previously contacted clients, those contacted more recently (lower pdays) seem more likely to convert.")

def bivariate_categorical_numerical_analysis(df, output_path='charts'):
    """
    NEW: Explores the interaction between key categorical and numerical features.
    """
    print("\n--- Bivariate Analysis (Categorical vs. Numerical) ---")
    
    print("Plotting Balance distribution across Job types...")
    plt.figure(figsize=(15, 8))
    # Using a log scale for balance due to skewness
    df_plot = df.copy()
    df_plot['log_balance'] = np.log1p(df_plot['balance'] - df_plot['balance'].min()) # a robust way to log-transform
    sns.violinplot(data=df_plot, x='job', y='log_balance', hue='y', split=True, palette='coolwarm', inner='quart')
    plt.title('Log(Balance) Distribution by Job and Subscription Status', fontsize=16)
    plt.xlabel('Job Type', fontsize=12)
    plt.ylabel('Log-Transformed Balance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_path}/violin_balance_by_job.png')
    plt.show()


def analyze_correlations(df, output_path='charts'):
    """
    Calculates and visualizes the correlation matrix for numerical features.
    """
    print("\n--- Correlation Analysis ---")
    numerical_cols = df.select_dtypes(include=np.number).columns
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=18)
    plt.savefig(f'{output_path}/correlation_matrix.png')
    plt.show()
    print("\nInsight: No strong multicollinearity detected among the primary numerical predictors.")


def main():
    """
    Main function to run the complete EDA pipeline on the training data.
    """
    output_path = 'charts'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    filepath = '/home/UNIZA/kboshomaneadmin/Probability to covert/train_new.csv' 
    
    bank_df = load_data(filepath)
    
    if bank_df is not None:
        analyze_target_variable(bank_df, output_path)
        analyze_numerical_features(bank_df, output_path)
        enhanced_numerical_distribution_plots(bank_df, output_path) # New function call
        analyze_categorical_features(bank_df, output_path)
        analyze_campaign_features(bank_df, output_path) # New function call
        bivariate_categorical_numerical_analysis(bank_df, output_path) # New function call
        analyze_correlations(bank_df, output_path)
        print("\n--- Expanded Exploratory Data Analysis Complete ---")
        print(f"Generated plots have been saved to the '{output_path}' directory.")

if __name__ == '__main__':
    main()

