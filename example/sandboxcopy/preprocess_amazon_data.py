#!/usr/bin/env python3
"""
Amazon Sandbox Data Preprocessing Script

This script implements diffusion-friendly preprocessing for Amazon advertising data
following the detailed rules in preprocessing_rules file. The goal is to prepare
data for diffusion-based tabular generators by removing tree shortcuts, adding
learnable features, and stabilizing numerics.

Usage:
    python preprocess_amazon_data.py
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AmazonDataPreprocessor:
    """
    Preprocessor for Amazon advertising data following diffusion-friendly principles.
    """
    
    def __init__(self, 
                 min_count_cat: int = 30,
                 segments_top_k: int = 100,
                 segment_hash_bins: int = 16,
                 id_like_unique_frac: float = 0.95,
                 extreme_card_threshold: int = 5000,
                 one_hot: bool = False):
        
        self.min_count_cat = min_count_cat
        self.segments_top_k = segments_top_k
        self.segment_hash_bins = segment_hash_bins
        self.id_like_unique_frac = id_like_unique_frac
        self.extreme_card_threshold = extreme_card_threshold
        self.one_hot = one_hot
        
        # Allowlist of high-cardinality columns to keep
        self.allowlist = {
            'dma_code', 'browser_family', 'product_line', 
            'supply_source', 'user_id_type', 'device_type',
            'marketplace_name', 'campaign', 'line_item'
        }
        
        # Always drop these columns
        self.always_drop = {
            'request_tag', 'user_id', 'postal_code', 'entity_id',
            'conversion_id', 'event_id', 'traffic_event_id'
        }
        
        self.metadata = {
            'numeric_transforms': {},
            'categorical_vocabs': {},
            'segment_info': {},
            'time_info': {},
            'cost_info': {},
            'dropped_columns': [],
            'kept_columns': []
        }
    
    def identify_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify datetime columns based on naming patterns."""
        datetime_patterns = ['time', 'date', 'timestamp', 'dt', 'utc']
        datetime_cols = []
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in datetime_patterns):
                datetime_cols.append(col)
        
        return datetime_cols
    
    def identify_id_like_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify ID-like columns with high uniqueness."""
        id_like_cols = []
        
        for col in df.columns:
            if col in self.always_drop:
                continue
                
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                unique_frac = df[col].nunique() / len(df)
                unique_count = df[col].nunique()
                
                if (unique_frac >= self.id_like_unique_frac or 
                    unique_count >= self.extreme_card_threshold) and col not in self.allowlist:
                    id_like_cols.append(col)
        
        return id_like_cols
    
    def parse_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns and create cyclic time features."""
        df = df.copy()
        datetime_cols = self.identify_datetime_columns(df)
        
        # Find the best datetime column (prefer impression_dt_utc)
        base_time_col = None
        for preferred in ['impression_dt_utc', 'event_dt_utc', 'click_date_utc', 'traffic_event_date_utc']:
            if preferred in datetime_cols:
                base_time_col = preferred
                break
        
        if not base_time_col and datetime_cols:
            base_time_col = datetime_cols[0]
        
        if base_time_col and base_time_col in df.columns:
            try:
                # Parse the datetime column
                df[base_time_col] = pd.to_datetime(df[base_time_col], errors='coerce')
                
                # Extract time features
                dt_series = df[base_time_col]
                df['hour'] = dt_series.dt.hour
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                
                df['dow'] = dt_series.dt.dayofweek  # 0=Monday
                df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
                df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
                
                df['is_weekend'] = (df['dow'] >= 5).astype(int)
                df['month'] = dt_series.dt.month
                
                # Store metadata
                self.metadata['time_info'] = {
                    'base_column': base_time_col,
                    'derived_features': ['hour', 'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos', 'is_weekend', 'month']
                }
                
                print(f"Created time features from {base_time_col}")
                
            except Exception as e:
                print(f"Warning: Could not parse datetime column {base_time_col}: {e}")
        
        # Drop all original datetime columns
        cols_to_drop = [col for col in datetime_cols if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        self.metadata['dropped_columns'].extend(cols_to_drop)
        
        return df
    
    def create_site_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create coarse site buckets based on supply_source."""
        df = df.copy()
        
        if 'site' in df.columns and 'supply_source' in df.columns:
            # Create site bucket based on supply_source patterns
            def classify_site_type(supply_source):
                if pd.isna(supply_source):
                    return 'unknown'
                
                supply_source = str(supply_source).lower()
                app_patterns = ['app', 'mobile', 'android', 'ios', 'iphone']
                
                if any(pattern in supply_source for pattern in app_patterns):
                    return 'app'
                else:
                    return 'web'
            
            df['site_bucket'] = df['supply_source'].apply(classify_site_type)
            
            # Apply min count filtering
            bucket_counts = df['site_bucket'].value_counts()
            valid_buckets = bucket_counts[bucket_counts >= self.min_count_cat].index
            df['site_bucket'] = df['site_bucket'].apply(
                lambda x: x if x in valid_buckets else '<<OTHER>>'
            )
            
            # Drop original site column
            df = df.drop(columns=['site'])
            self.metadata['dropped_columns'].append('site')
            
            print(f"Created site_bucket with buckets: {df['site_bucket'].value_counts().to_dict()}")
        
        return df
    
    def process_segment_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process user behavior segment columns into multi-label features."""
        df = df.copy()
        
        segment_cols = ['user_behavior_segment_ids', 'matched_behavior_segment_ids']
        all_segments = set()
        
        # Collect all segment IDs
        for col in segment_cols:
            if col in df.columns:
                for segments_str in df[col].dropna():
                    if segments_str and str(segments_str).strip():
                        try:
                            # Parse string representation of list
                            segments_str = str(segments_str).strip()
                            if segments_str.startswith('[') and segments_str.endswith(']'):
                                # Remove brackets and split by comma
                                segments_str = segments_str[1:-1]
                                if segments_str.strip():
                                    segments = [int(s.strip()) for s in segments_str.split(',') if s.strip()]
                                    all_segments.update(segments)
                        except:
                            continue
        
        if all_segments:
            # Get top-K most frequent segments
            segment_counts = {}
            for col in segment_cols:
                if col in df.columns:
                    for segments_str in df[col].dropna():
                        if segments_str and str(segments_str).strip():
                            try:
                                segments_str = str(segments_str).strip()
                                if segments_str.startswith('[') and segments_str.endswith(']'):
                                    segments_str = segments_str[1:-1]
                                    if segments_str.strip():
                                        segments = [int(s.strip()) for s in segments_str.split(',') if s.strip()]
                                        for seg in segments:
                                            segment_counts[seg] = segment_counts.get(seg, 0) + 1
                            except:
                                continue
            
            # Select top segments
            top_segments = sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)[:self.segments_top_k]
            top_segment_ids = [seg_id for seg_id, _ in top_segments]
            
            # Create one-hot features for top segments
            for seg_id in top_segment_ids:
                df[f'seg_{seg_id}'] = 0
            
            # Create hash bin features
            for i in range(self.segment_hash_bins):
                df[f'seg_hash_{i}'] = 0
            
            # Add segment count feature
            df['num_segments'] = 0
            
            # Fill features
            def extract_segments(segments_str):
                if not segments_str or str(segments_str).strip() == '[]':
                    return []
                try:
                    segments_str = str(segments_str).strip()
                    if segments_str.startswith('[') and segments_str.endswith(']'):
                        segments_str = segments_str[1:-1]
                        if segments_str.strip():
                            return [int(s.strip()) for s in segments_str.split(',') if s.strip()]
                except:
                    pass
                return []
            
            for col in segment_cols:
                if col in df.columns:
                    for idx, segments_str in df[col].items():
                        segments = extract_segments(segments_str)
                        
                        # Update segment count
                        df.loc[idx, 'num_segments'] += len(segments)
                        
                        # Update one-hot features
                        for seg_id in segments:
                            if seg_id in top_segment_ids:
                                df.loc[idx, f'seg_{seg_id}'] = 1
                            
                            # Update hash features
                            hash_bin = hash(seg_id) % self.segment_hash_bins
                            df.loc[idx, f'seg_hash_{hash_bin}'] += 1
            
            # Store metadata
            self.metadata['segment_info'] = {
                'top_segment_ids': top_segment_ids,
                'hash_bins': self.segment_hash_bins,
                'source_columns': segment_cols
            }
            
            # Drop original segment columns
            df = df.drop(columns=[col for col in segment_cols if col in df.columns])
            self.metadata['dropped_columns'].extend([col for col in segment_cols if col in df.columns])
            
            print(f"Created segment features: {len(top_segment_ids)} one-hot + {self.segment_hash_bins} hash bins + count")
        
        return df
    
    def process_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical columns with frequency filtering."""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        for col in categorical_cols:
            if col.startswith('seg_') or col in ['site_bucket']:  # Skip already processed
                continue
                
            # Get value counts
            value_counts = df[col].value_counts()
            
            # Keep values with sufficient frequency
            valid_values = value_counts[value_counts >= self.min_count_cat].index.tolist()
            
            # Add special tokens
            vocab = ['na', '<<OTHER>>'] + valid_values
            
            # Map values
            df[col] = df[col].fillna('na')
            df[col] = df[col].apply(lambda x: x if x in valid_values else '<<OTHER>>')
            
            if self.one_hot:
                # Create one-hot encoding
                for val in vocab:
                    df[f'{col}_{val}'] = (df[col] == val).astype(int)
                df = df.drop(columns=[col])
                self.metadata['dropped_columns'].append(col)
            else:
                # Store vocabulary for embedding
                self.metadata['categorical_vocabs'][col] = vocab
            
            print(f"Processed categorical {col}: {len(vocab)} categories")
        
        return df
    
    def process_cost_composition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process cost columns using CLR transformation."""
        df = df.copy()
        
        # Define cost columns based on common patterns across files
        total_cost_cols = ['total_cost', 'impression_cost', 'click_cost', 'spend']
        part_cost_cols = ['audience_fee', 'platform_fee', 'supply_cost', 'third_party_fees']
        
        # Find available cost columns
        available_total = [col for col in total_cost_cols if col in df.columns]
        available_parts = [col for col in part_cost_cols if col in df.columns]
        
        if available_total and available_parts:
            total_col = available_total[0]  # Use first available
            
            # Create log total cost
            df[f'log_{total_col}'] = np.log1p(df[total_col].fillna(0))
            
            # Calculate proportions and apply CLR
            total_parts = df[available_parts].fillna(0).sum(axis=1)
            
            # Avoid division by zero
            total_parts = total_parts.replace(0, 1e-8)
            
            for col in available_parts:
                proportions = df[col].fillna(0) / total_parts
                # Add small constant to avoid log(0)
                proportions = proportions + 1e-8
                df[f'cost_clr_{col}'] = np.log(proportions) - np.log(proportions).mean()
            
            # Store metadata
            self.metadata['cost_info'] = {
                'total_column': total_col,
                'part_columns': available_parts,
                'log_total_feature': f'log_{total_col}',
                'clr_features': [f'cost_clr_{col}' for col in available_parts]
            }
            
            print(f"Created cost composition features from {total_col} and {len(available_parts)} parts")
        
        return df
    
    def transform_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate transformations to numeric columns."""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith(('seg_', 'hour_', 'dow_', 'cost_clr_', 'log_')):
                continue  # Skip already processed features
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            transform_info = {'column': col}
            
            # Determine transformation based on data characteristics
            if all(values >= 0) and all(values <= 1):
                # Rates in [0,1] -> logit + standardize
                # Add small epsilon to avoid inf
                eps = 1e-7
                values_adj = np.clip(values, eps, 1-eps)
                logit_values = np.log(values_adj / (1 - values_adj))
                
                mean_val = logit_values.mean()
                std_val = logit_values.std()
                
                df[col] = df[col].fillna(0.5)  # Fill with median for rates
                df[col] = np.clip(df[col], eps, 1-eps)
                df[col] = np.log(df[col] / (1 - df[col]))
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)
                
                transform_info.update({
                    'type': 'logit_standardize',
                    'mean': mean_val,
                    'std': std_val
                })
                
            elif all(values >= 0):
                # Non-negative -> log1p + standardize
                log_values = np.log1p(values)
                mean_val = log_values.mean()
                std_val = log_values.std()
                
                df[col] = df[col].fillna(0)
                df[col] = np.log1p(df[col])
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)
                
                transform_info.update({
                    'type': 'log1p_standardize', 
                    'mean': mean_val,
                    'std': std_val
                })
                
            else:
                # General case -> standardize
                mean_val = values.mean()
                std_val = values.std()
                
                df[col] = df[col].fillna(mean_val)
                df[col] = (df[col] - mean_val) / (std_val + 1e-8)
                
                transform_info.update({
                    'type': 'standardize',
                    'mean': mean_val,
                    'std': std_val
                })
            
            self.metadata['numeric_transforms'][col] = transform_info
            print(f"Transformed {col} using {transform_info['type']}")
        
        return df
    
    def final_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final data quality cleanup similar to preprocess_dataset()."""
        df = df.copy()
        print(f"  Post-transform shape: {df.shape}")
        print(f"  Post-transform dtypes: {df.dtypes.value_counts().to_dict()}")

        # Fill NaN values appropriately
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            print(f"  Columns with NaN: {len(nan_cols)}")
            for col in nan_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    df[col] = df[col].fillna(fill_val)

        # Drop problematic columns with all NaN or constant values
        problematic_cols = []
        for col in df.columns:
            try:
                if df[col].nunique(dropna=False) <= 1:
                    problematic_cols.append(col)
            except Exception:
                # If nunique fails for some reason, skip the column
                continue
        if problematic_cols:
            print(f"  Removing {len(problematic_cols)} constant/problematic columns")
            df = df.drop(columns=problematic_cols)
            self.metadata['dropped_columns'].extend(problematic_cols)

        # Ensure all object columns are strings
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        print(f"  Post-cleanup shape: {df.shape}")
        return df
    
    def preprocess_file(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """Preprocess a single CSV file."""
        print(f"\nProcessing {input_path}...")
        
        # Reset metadata for this file
        self.metadata = {
            'numeric_transforms': {},
            'categorical_vocabs': {},
            'segment_info': {},
            'time_info': {},
            'cost_info': {},
            'dropped_columns': [],
            'kept_columns': []
        }
        
        # Load data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        original_columns = set(df.columns)
        
        # Step 1: Drop always-drop columns
        cols_to_drop = [col for col in self.always_drop if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.metadata['dropped_columns'].extend(cols_to_drop)
            print(f"Dropped always-drop columns: {cols_to_drop}")
        
        # Step 2: Identify and drop ID-like columns
        id_like_cols = self.identify_id_like_columns(df)
        if id_like_cols:
            df = df.drop(columns=id_like_cols)
            self.metadata['dropped_columns'].extend(id_like_cols)
            print(f"Dropped ID-like columns: {id_like_cols}")
        
        # Step 3: Process datetime columns
        df = self.parse_datetime_columns(df)
        
        # Step 4: Create site buckets
        df = self.create_site_bucket(df)
        
        # Step 5: Process segment columns
        df = self.process_segment_columns(df)
        
        # Step 6: Process cost composition
        df = self.process_cost_composition(df)
        
        # Step 7: Process categorical columns
        df = self.process_categorical_columns(df)
        
        # Step 8: Transform numeric columns
        df = self.transform_numeric_columns(df)
        
        # Step 9: Final data-quality cleanup (fill NaNs, drop constants, ensure strings)
        df = self.final_quality_checks(df)
        
        # Final cleanup - record kept columns
        self.metadata['kept_columns'] = list(df.columns)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        print(f"Final shape: {df.shape}")
        
        # Save metadata
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Saved metadata to {metadata_path}")
        
        return self.metadata

def main():
    """Main preprocessing function."""
    
    # Define input and output directories
    input_dir = Path('/home/xiaofeng/table_reasoning/table-synthesizers/example/sandboxcopy')
    output_dir = Path('/home/xiaofeng/table_reasoning/table-synthesizers/example/sandbox_preprocessed')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AmazonDataPreprocessor(
        min_count_cat=30,
        segments_top_k=100,
        segment_hash_bins=16,
        one_hot=False  # Keep categorical as indices for embedding
    )
    
    # Detect subfolders with CSV subsets and process each as one table
    subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
    processed_any_subdir = False
    for subdir in sorted(subdirs):
        csv_paths = sorted(subdir.glob('*.csv'))
        if not csv_paths:
            continue
        processed_any_subdir = True
        print(f"\nConcatenating {len(csv_paths)} CSVs from {subdir.name} ...")
        frames = []
        for p in csv_paths:
            try:
                df_part = pd.read_csv(p)
                frames.append(df_part)
            except Exception as e:
                print(f"  ⚠ Skipping {p.name}: {e}")
        if not frames:
            print(f"  ⚠ No readable CSVs in {subdir}")
            continue
        # Align columns (union) before concat
        all_cols = set()
        for f in frames:
            all_cols.update(list(f.columns))
        all_cols = list(all_cols)
        frames = [f.reindex(columns=all_cols) for f in frames]
        combined_df = pd.concat(frames, ignore_index=True, sort=False)
        print(f"  Combined shape for {subdir.name}: {combined_df.shape}")
        # Write to a temp combined CSV
        temp_combined = output_dir / f"__combined_{subdir.name}.csv"
        combined_df.to_csv(temp_combined, index=False)
        # Preprocess and write one output per subfolder
        output_path = output_dir / f"preprocessed_{subdir.name}.csv"
        try:
            preprocessor.preprocess_file(str(temp_combined), str(output_path))
            print(f"✓ Successfully processed folder {subdir.name}")
        finally:
            try:
                os.remove(temp_combined)
            except Exception:
                pass
    
    # If no subfolders with CSVs found, fall back to legacy single-file names in root
    if not processed_any_subdir:
        # Define the legacy CSV files to process if they exist at root
        csv_files = [
            'sponsored_ads_traffic_7-29-25.csv',
            'conversions_all_8-1-25.csv', 
            'dsp_impressions_7-29-25.csv',
            'amazon_attributed_events_by_traffic_time_7-29-25.csv'
        ]
        # Process each file
        for csv_file in csv_files:
            input_path = input_dir / csv_file
            output_path = output_dir / f"preprocessed_{csv_file}"
            
            if input_path.exists():
                try:
                    metadata = preprocessor.preprocess_file(str(input_path), str(output_path))
                    print(f"✓ Successfully processed {csv_file}")
                except Exception as e:
                    print(f"✗ Error processing {csv_file}: {e}")
            else:
                print(f"⚠ File not found: {input_path}")
    
    print(f"\n🎉 Preprocessing complete! Check output directory: {output_dir}")

if __name__ == "__main__":
    main()
