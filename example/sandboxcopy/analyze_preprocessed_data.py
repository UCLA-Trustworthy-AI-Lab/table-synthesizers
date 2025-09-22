#!/usr/bin/env python3
"""
Preprocessed Data Analysis Script

This script analyzes the preprocessed Amazon advertising data files to provide:
1. Data types for all columns after preprocessing
2. Distribution of unique categories among categorical columns
3. Summary statistics for numeric columns
4. Overall data quality metrics

Usage:
    python analyze_preprocessed_data.py
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class PreprocessedDataAnalyzer:
    """
    Analyzer for preprocessed Amazon advertising data.
    """
    
    def __init__(self, data_dir: str, metadata_dir: str = None):
        self.data_dir = Path(data_dir)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.data_dir
        
    def load_metadata(self, metadata_file: str) -> Dict[str, Any]:
        """Load metadata JSON file."""
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_file}: {e}")
            return {}
    
    def analyze_data_types(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data types of all columns."""
        type_analysis = {
            'column_types': {},
            'type_summary': {},
            'feature_categories': {
                'numeric_original': [],
                'numeric_transformed': [],
                'categorical': [],
                'time_derived': [],
                'segment_features': [],
                'cost_features': []
            }
        }
        
        # Get time features from metadata
        time_features = set()
        if 'time_info' in metadata and 'derived_features' in metadata['time_info']:
            time_features = set(metadata['time_info']['derived_features'])
        
        # Get segment features
        segment_features = set()
        for col in df.columns:
            if col.startswith(('seg_', 'num_segments')):
                segment_features.add(col)
        
        # Get cost features
        cost_features = set()
        for col in df.columns:
            if col.startswith(('cost_clr_', 'log_')):
                cost_features.add(col)
        
        # Analyze each column
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            
            # Classify column type
            if col in time_features:
                category = 'time_derived'
            elif col in segment_features:
                category = 'segment_features'
            elif col in cost_features:
                category = 'cost_features'
            elif df[col].dtype in ['object', 'category']:
                category = 'categorical'
            elif col in metadata.get('numeric_transforms', {}):
                category = 'numeric_transformed'
            else:
                category = 'numeric_original'
            
            type_analysis['column_types'][col] = {
                'dtype': dtype,
                'nunique': nunique,
                'category': category,
                'null_count': df[col].isnull().sum(),
                'null_pct': (df[col].isnull().sum() / len(df)) * 100
            }
            
            type_analysis['feature_categories'][category].append(col)
        
        # Create type summary
        type_counts = {}
        for col_info in type_analysis['column_types'].values():
            dtype = col_info['dtype']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        type_analysis['type_summary'] = type_counts
        
        return type_analysis
    
    def analyze_categorical_distributions(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze categorical column distributions."""
        cat_analysis = {
            'categorical_columns': {},
            'unique_counts_distribution': {},
            'vocabulary_info': {}
        }
        
        categorical_cols = [col for col in df.columns 
                          if df[col].dtype in ['object', 'category']]
        
        unique_counts = []
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            col_info = {
                'total_unique': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_pct': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                'value_distribution': dict(value_counts.head(10))  # Top 10 values
            }
            
            # Check if vocabulary exists in metadata
            if col in metadata.get('categorical_vocabs', {}):
                vocab = metadata['categorical_vocabs'][col]
                col_info['vocabulary_size'] = len(vocab)
                col_info['has_other_token'] = '<<OTHER>>' in vocab
                col_info['has_na_token'] = 'na' in vocab
            
            cat_analysis['categorical_columns'][col] = col_info
            unique_counts.append(df[col].nunique())
        
        # Analyze distribution of unique counts
        if unique_counts:
            cat_analysis['unique_counts_distribution'] = {
                'min_unique': min(unique_counts),
                'max_unique': max(unique_counts),
                'mean_unique': np.mean(unique_counts),
                'median_unique': np.median(unique_counts),
                'std_unique': np.std(unique_counts),
                'unique_counts_by_column': dict(zip(categorical_cols, unique_counts))
            }
        
        # Vocabulary info from metadata
        if 'categorical_vocabs' in metadata:
            for col, vocab in metadata['categorical_vocabs'].items():
                cat_analysis['vocabulary_info'][col] = {
                    'vocab_size': len(vocab),
                    'vocabulary': vocab
                }
        
        return cat_analysis
    
    def analyze_numeric_distributions(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze numeric column distributions."""
        numeric_analysis = {
            'numeric_columns': {},
            'transformation_summary': {},
            'statistical_summary': {}
        }
        
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            
            col_info = {
                'count': len(col_data),
                'mean': col_data.mean() if len(col_data) > 0 else 0,
                'std': col_data.std() if len(col_data) > 0 else 0,
                'min': col_data.min() if len(col_data) > 0 else 0,
                'max': col_data.max() if len(col_data) > 0 else 0,
                'median': col_data.median() if len(col_data) > 0 else 0,
                'q25': col_data.quantile(0.25) if len(col_data) > 0 else 0,
                'q75': col_data.quantile(0.75) if len(col_data) > 0 else 0,
                'zeros_count': (col_data == 0).sum(),
                'zeros_pct': ((col_data == 0).sum() / len(col_data)) * 100 if len(col_data) > 0 else 0
            }
            
            # Check if this column was transformed
            if col in metadata.get('numeric_transforms', {}):
                transform_info = metadata['numeric_transforms'][col]
                col_info['transformation'] = transform_info['type']
                col_info['original_mean'] = transform_info.get('mean')
                col_info['original_std'] = transform_info.get('std')
            else:
                col_info['transformation'] = 'none'
            
            numeric_analysis['numeric_columns'][col] = col_info
        
        # Transformation summary
        transform_types = {}
        for col_info in numeric_analysis['numeric_columns'].values():
            trans_type = col_info.get('transformation', 'none')
            transform_types[trans_type] = transform_types.get(trans_type, 0) + 1
        
        numeric_analysis['transformation_summary'] = transform_types
        
        # Overall statistical summary
        if numeric_cols:
            all_means = [info['mean'] for info in numeric_analysis['numeric_columns'].values()]
            all_stds = [info['std'] for info in numeric_analysis['numeric_columns'].values()]
            
            numeric_analysis['statistical_summary'] = {
                'total_numeric_columns': len(numeric_cols),
                'mean_of_means': np.mean(all_means),
                'mean_of_stds': np.mean(all_stds),
                'standardized_columns': sum(1 for info in numeric_analysis['numeric_columns'].values() 
                                          if abs(info['mean']) < 0.1 and abs(info['std'] - 1.0) < 0.1)
            }
        
        return numeric_analysis
    
    def analyze_segment_features(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze segment-related features."""
        segment_analysis = {
            'segment_summary': {},
            'top_segments': {},
            'hash_features': {},
            'segment_statistics': {}
        }
        
        # Get segment info from metadata
        if 'segment_info' in metadata:
            segment_info = metadata['segment_info']
            segment_analysis['segment_summary'] = {
                'total_top_segments': len(segment_info.get('top_segment_ids', [])),
                'hash_bins': segment_info.get('hash_bins', 0),
                'source_columns': segment_info.get('source_columns', [])
            }
            
            # Analyze top segment features
            top_segment_ids = segment_info.get('top_segment_ids', [])
            for seg_id in top_segment_ids[:10]:  # Top 10 most frequent
                col_name = f'seg_{seg_id}'
                if col_name in df.columns:
                    activation_rate = df[col_name].mean()
                    segment_analysis['top_segments'][seg_id] = {
                        'activation_rate': activation_rate,
                        'activation_count': df[col_name].sum()
                    }
            
            # Analyze hash features
            hash_bins = segment_info.get('hash_bins', 0)
            for i in range(min(hash_bins, 5)):  # First 5 hash bins
                col_name = f'seg_hash_{i}'
                if col_name in df.columns:
                    segment_analysis['hash_features'][f'hash_{i}'] = {
                        'mean_value': df[col_name].mean(),
                        'max_value': df[col_name].max(),
                        'nonzero_pct': ((df[col_name] > 0).sum() / len(df)) * 100
                    }
        
        # Analyze num_segments if present
        if 'num_segments' in df.columns:
            segment_analysis['segment_statistics'] = {
                'mean_segments_per_row': df['num_segments'].mean(),
                'max_segments_per_row': df['num_segments'].max(),
                'zero_segments_pct': ((df['num_segments'] == 0).sum() / len(df)) * 100
            }
        
        return segment_analysis
    
    def generate_file_summary(self, csv_file: str) -> Dict[str, Any]:
        """Generate comprehensive summary for a single file."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {csv_file}")
        print(f"{'='*60}")
        
        # Load data and metadata
        df = pd.read_csv(csv_file)
        metadata_file = csv_file.replace('.csv', '_metadata.json')
        metadata = self.load_metadata(metadata_file)
        
        print(f"Data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Perform all analyses
        type_analysis = self.analyze_data_types(df, metadata)
        cat_analysis = self.analyze_categorical_distributions(df, metadata)
        numeric_analysis = self.analyze_numeric_distributions(df, metadata)
        segment_analysis = self.analyze_segment_features(df, metadata)
        
        # Print results
        self.print_analysis_results(type_analysis, cat_analysis, numeric_analysis, segment_analysis)
        
        # Return comprehensive summary
        return {
            'file_info': {
                'filename': os.path.basename(csv_file),
                'shape': df.shape,
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
            },
            'type_analysis': type_analysis,
            'categorical_analysis': cat_analysis,
            'numeric_analysis': numeric_analysis,
            'segment_analysis': segment_analysis
        }
    
    def print_analysis_results(self, type_analysis, cat_analysis, numeric_analysis, segment_analysis):
        """Print formatted analysis results."""
        
        print(f"\n📊 DATA TYPES SUMMARY")
        print(f"{'='*40}")
        for dtype, count in type_analysis['type_summary'].items():
            print(f"{dtype:15}: {count:3d} columns")
        
        print(f"\n🏷️  FEATURE CATEGORIES")
        print(f"{'='*40}")
        for category, columns in type_analysis['feature_categories'].items():
            if columns:
                print(f"{category:20}: {len(columns):3d} columns")
        
        print(f"\n📈 CATEGORICAL DISTRIBUTIONS")
        print(f"{'='*40}")
        if cat_analysis['categorical_columns']:
            print(f"{'Column':<30} {'Unique':<8} {'Most Freq %':<12} {'Top Value'}")
            print("-" * 70)
            for col, info in list(cat_analysis['categorical_columns'].items())[:10]:
                print(f"{col:<30} {info['total_unique']:<8} {info['most_frequent_pct']:<11.1f}% {str(info['most_frequent']):<15}")
            
            if len(cat_analysis['categorical_columns']) > 10:
                print(f"... and {len(cat_analysis['categorical_columns']) - 10} more categorical columns")
            
            if 'unique_counts_distribution' in cat_analysis:
                dist = cat_analysis['unique_counts_distribution']
                print(f"\nUnique counts - Min: {dist['min_unique']}, Max: {dist['max_unique']}, Mean: {dist['mean_unique']:.1f}")
        else:
            print("No categorical columns found")
        
        print(f"\n🔢 NUMERIC TRANSFORMATIONS")
        print(f"{'='*40}")
        if numeric_analysis['transformation_summary']:
            for transform, count in numeric_analysis['transformation_summary'].items():
                print(f"{transform:20}: {count:3d} columns")
        
        if 'statistical_summary' in numeric_analysis:
            stats = numeric_analysis['statistical_summary']
            print(f"\nStandardized columns: {stats.get('standardized_columns', 0)}/{stats.get('total_numeric_columns', 0)}")
        
        print(f"\n🎯 SEGMENT FEATURES")
        print(f"{'='*40}")
        if segment_analysis['segment_summary']:
            summary = segment_analysis['segment_summary']
            print(f"Top segments: {summary.get('total_top_segments', 0)}")
            print(f"Hash bins: {summary.get('hash_bins', 0)}")
            
            if segment_analysis['top_segments']:
                print(f"\nTop 5 most active segments:")
                for seg_id, info in list(segment_analysis['top_segments'].items())[:5]:
                    print(f"  seg_{seg_id}: {info['activation_rate']:.3f} activation rate")
        else:
            print("No segment features found")
    
    def analyze_all_files(self):
        """Analyze all preprocessed files in the directory."""
        print("🔍 PREPROCESSING DATA ANALYSIS REPORT")
        print("=" * 80)
        
        # Find all preprocessed CSV files
        csv_files = list(self.data_dir.glob("preprocessed_*.csv"))
        
        if not csv_files:
            print("No preprocessed CSV files found!")
            return
        
        all_summaries = {}
        
        for csv_file in sorted(csv_files):
            try:
                summary = self.generate_file_summary(str(csv_file))
                all_summaries[csv_file.name] = summary
            except Exception as e:
                print(f"❌ Error analyzing {csv_file.name}: {e}")
        
        # Generate overall summary
        self.print_overall_summary(all_summaries)
        
        return all_summaries
    
    def print_overall_summary(self, all_summaries):
        """Print overall summary across all files."""
        print(f"\n🌟 OVERALL SUMMARY ACROSS ALL FILES")
        print(f"{'='*60}")
        
        total_files = len(all_summaries)
        total_rows = sum(summary['file_info']['shape'][0] for summary in all_summaries.values())
        total_cols = sum(summary['file_info']['shape'][1] for summary in all_summaries.values())
        total_memory = sum(summary['file_info']['memory_mb'] for summary in all_summaries.values())
        
        print(f"Files processed: {total_files}")
        print(f"Total rows: {total_rows:,}")
        print(f"Total columns: {total_cols:,}")
        print(f"Total memory: {total_memory:.2f} MB")
        
        # Aggregate type information
        all_dtypes = {}
        all_categories = {}
        
        for summary in all_summaries.values():
            for dtype, count in summary['type_analysis']['type_summary'].items():
                all_dtypes[dtype] = all_dtypes.get(dtype, 0) + count
            
            for category, columns in summary['type_analysis']['feature_categories'].items():
                all_categories[category] = all_categories.get(category, 0) + len(columns)
        
        print(f"\nData types across all files:")
        for dtype, count in sorted(all_dtypes.items()):
            print(f"  {dtype}: {count} columns")
        
        print(f"\nFeature categories across all files:")
        for category, count in sorted(all_categories.items()):
            if count > 0:
                print(f"  {category}: {count} columns")

def main():
    """Main analysis function."""
    
    # Define the preprocessed data directory
    data_dir = "/home/xiaofeng/table_reasoning/table-synthesizers/example/sandbox_preprocessed"
    
    # Initialize analyzer
    analyzer = PreprocessedDataAnalyzer(data_dir)
    
    # Analyze all files
    summaries = analyzer.analyze_all_files()
    
    print(f"\n✅ Analysis complete! Check the output above for detailed statistics.")

if __name__ == "__main__":
    main()
