#!/usr/bin/env python3
"""
CSV Data Analysis Script

This script analyzes all CSV files in the current directory and provides:
1. Distribution of data types for each file
2. Number of unique categories in each non-numerical column
3. Mean and standard deviation for all columns
4. Proportion of missing values in each column

Author: Generated for table-synthesizers project
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

def analyze_csv_file(file_path):
    """
    Analyze a single CSV file and return comprehensive statistics.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing analysis results
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        # Read the CSV file with appropriate settings for large files
        print("Loading data...")
        df = pd.read_csv(file_path, low_memory=False)
        
        print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        # 1. Data types distribution
        print(f"\n{'-'*40}")
        print("1. DATA TYPES DISTRIBUTION")
        print(f"{'-'*40}")
        
        dtype_counts = df.dtypes.value_counts()
        print("Data type counts:")
        for dtype, count in dtype_counts.items():
            print(f"  {str(dtype):15}: {count:4d} columns")
        
        # 2. Missing values analysis
        print(f"\n{'-'*40}")
        print("2. MISSING VALUES ANALYSIS")
        print(f"{'-'*40}")
        
        missing_stats = df.isnull().sum()
        missing_props = (missing_stats / len(df)) * 100
        
        # Show columns with missing values
        missing_cols = missing_stats[missing_stats > 0]
        if len(missing_cols) > 0:
            print(f"Columns with missing values ({len(missing_cols)} total):")
            for col in missing_cols.head(20).index:  # Show first 20 to avoid overwhelming output
                print(f"  {col:50}: {missing_stats[col]:8,} ({missing_props[col]:6.2f}%)")
            if len(missing_cols) > 20:
                print(f"  ... and {len(missing_cols) - 20} more columns with missing values")
        else:
            print("No missing values found in any column.")
        
        # 3. Numerical columns analysis
        print(f"\n{'-'*40}")
        print("3. NUMERICAL COLUMNS STATISTICS")
        print(f"{'-'*40}")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Number of numerical columns: {len(numerical_cols)}")
        
        if len(numerical_cols) > 0:
            # Calculate statistics for numerical columns
            num_stats = df[numerical_cols].describe()
            
            print("\nSample of numerical statistics (first 10 columns):")
            print("Column".ljust(50) + "Mean".rjust(15) + "Std".rjust(15) + "Min".rjust(15) + "Max".rjust(15))
            print("-" * 110)
            
            for col in numerical_cols[:10]:  # Show first 10 to avoid overwhelming output
                try:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    print(f"{col[:49]:50} {mean_val:14.4f} {std_val:14.4f} {min_val:14.4f} {max_val:14.4f}")
                except:
                    print(f"{col[:49]:50} {'ERROR':>14} {'ERROR':>14} {'ERROR':>14} {'ERROR':>14}")
            
            if len(numerical_cols) > 10:
                print(f"\n... and {len(numerical_cols) - 10} more numerical columns")
        
        # 4. Categorical columns analysis
        print(f"\n{'-'*40}")
        print("4. CATEGORICAL COLUMNS ANALYSIS")
        print(f"{'-'*40}")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        print(f"Number of categorical columns: {len(categorical_cols)}")
        
        if len(categorical_cols) > 0:
            print("\nUnique values count for categorical columns:")
            print("Column".ljust(50) + "Unique Values".rjust(15) + "Sample Values")
            print("-" * 100)
            
            for col in categorical_cols[:15]:  # Show first 15 to avoid overwhelming output
                try:
                    unique_count = df[col].nunique()
                    # Get sample values (first 3 unique values)
                    sample_values = df[col].dropna().unique()[:3]
                    sample_str = ", ".join([str(v)[:20] for v in sample_values])
                    if len(sample_str) > 30:
                        sample_str = sample_str[:27] + "..."
                    
                    print(f"{col[:49]:50} {unique_count:14,} {sample_str}")
                except:
                    print(f"{col[:49]:50} {'ERROR':>14} {'ERROR'}")
            
            if len(categorical_cols) > 15:
                print(f"\n... and {len(categorical_cols) - 15} more categorical columns")
        
        # 5. Memory usage
        print(f"\n{'-'*40}")
        print("5. MEMORY USAGE")
        print(f"{'-'*40}")
        
        memory_usage = df.memory_usage(deep=True).sum()
        print(f"Total memory usage: {memory_usage / (1024**2):.2f} MB")
        
        # Summary statistics
        analysis_results = {
            'file_name': os.path.basename(file_path),
            'shape': df.shape,
            'data_types': dtype_counts.to_dict(),
            'missing_values_count': len(missing_cols),
            'total_missing_cells': missing_stats.sum(),
            'numerical_columns': len(numerical_cols),
            'categorical_columns': len(categorical_cols),
            'memory_usage_mb': memory_usage / (1024**2)
        }
        
        return analysis_results
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def main():
    """Main function to analyze all CSV files in the current directory."""
    
    print("CSV Data Analysis Tool")
    print("=" * 80)
    
    # Find all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for i, file in enumerate(csv_files, 1):
        file_size = os.path.getsize(file) / (1024**2)  # Size in MB
        print(f"  {i}. {file} ({file_size:.1f} MB)")
    
    # Analyze each file
    all_results = []
    
    for file_path in csv_files:
        result = analyze_csv_file(file_path)
        if result:
            all_results.append(result)
    
    # Summary across all files
    if all_results:
        print(f"\n{'='*80}")
        print("SUMMARY ACROSS ALL FILES")
        print(f"{'='*80}")
        
        total_rows = sum(result['shape'][0] for result in all_results)
        total_columns = sum(result['shape'][1] for result in all_results)
        total_memory = sum(result['memory_usage_mb'] for result in all_results)
        
        print(f"Total datasets analyzed: {len(all_results)}")
        print(f"Total rows across all files: {total_rows:,}")
        print(f"Total columns across all files: {total_columns:,}")
        print(f"Total memory usage: {total_memory:.2f} MB")
        
        print(f"\nPer-file summary:")
        print("File".ljust(60) + "Rows".rjust(12) + "Cols".rjust(8) + "Num".rjust(8) + "Cat".rjust(8) + "Missing".rjust(10))
        print("-" * 116)
        
        for result in all_results:
            print(f"{result['file_name'][:59]:60} "
                  f"{result['shape'][0]:11,} "
                  f"{result['shape'][1]:7,} "
                  f"{result['numerical_columns']:7,} "
                  f"{result['categorical_columns']:7,} "
                  f"{result['missing_values_count']:9,}")

if __name__ == "__main__":
    main()
