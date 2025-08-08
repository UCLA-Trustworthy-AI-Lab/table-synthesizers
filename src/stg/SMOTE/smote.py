import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

# -----------------------------------------------------------------------------
# Custom SMOTE classes with adjustable lam parameters
# -----------------------------------------------------------------------------

class MySMOTE(SMOTE):
    def __init__(
        self,
        lam1: float = 0.0,
        lam2: float = 1.0,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors: int = 3,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.lam1 = lam1
        self.lam2 = lam2

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)
        # np.newaxis for backwards compatibility with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

class MySMOTENC(SMOTENC):
    def __init__(
        self,
        lam1: float = 0.0,
        lam2: float = 1.0,
        *,
        categorical_features,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors: int = 3,
        n_jobs: Optional[int] = None
    ):
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.lam1 = lam1
        self.lam2 = lam2

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)
        # np.newaxis for backwards compatibility with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

# -----------------------------------------------------------------------------
# New sample_smote function that accepts a pandas DataFrame and returns a
# combined DataFrame with the same column order, names and types as input.
# -----------------------------------------------------------------------------

def sample_smote(
    df: pd.DataFrame,
    target: str,
    categorical_features: Optional[List[str]] = None,
    eval_type: str = "synthetic",  # "synthetic" returns only new samples, "real" returns all
    k_neighbors: int = 3,
    frac_samples: float = 1.0,
    frac_lam_del: float = 0.0,
    is_regression: bool = False,
    seed: int = 0,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    n_samples: int = -1
) -> pd.DataFrame:
    """
    Perform SMOTE sampling on a pandas DataFrame and return a new DataFrame that
    contains both features and target in the same order as the input, with the same
    column names and original data types.
    
    Parameters:
      - df: Input DataFrame.
      - target: Name of the target column.
      - categorical_features: Optional list of feature names that should be treated as categorical.
        If not provided, any non-numeric column (among the non-target features) is assumed categorical.
      - eval_type: 'synthetic' returns only the synthetic samples; 'real' returns the entire (resampled) dataset.
      - k_neighbors: Number of neighbors to use in SMOTE.
      - frac_samples: Fraction to scale the desired number of samples per class.
      - frac_lam_del: Fraction by which to narrow the interpolation interval [lam1, lam2].
      - is_regression: If True, the regression target is temporarily appended for scaling and later recovered.
      - seed: Random seed.
      - save: If True, save the final resampled DataFrame to CSV.
      - save_path: Directory path where to save the CSV if save is True.
      - n_samples: Total number of data points to resample. If -1, uses the same size as input df.
    
    Returns:
      - A DataFrame with the resampled data, having the same column order, names and dtypes as the input.
    """
    # Determine the order of feature columns (all columns except target) as in the original df.
    features_order = [col for col in df.columns if col != target]
    
    # Determine which feature columns are categorical.
    if categorical_features is None:
        # Any non-numeric column among features is considered categorical.
        cat_cols = [col for col in features_order if not pd.api.types.is_numeric_dtype(df[col])]
    else:
        # Ensure the order is as in the original DataFrame.
        cat_cols = [col for col in features_order if col in categorical_features]
    num_cols = [col for col in features_order if col not in cat_cols]
    
    # Separate features and target.
    X = df[features_order].copy()
    y_orig = df[target].copy()
    
    # --- Process target ---
    # For classification, if the target is non-numeric, encode it.
    original_target_dtype = y_orig.dtype
    target_is_numeric = pd.api.types.is_numeric_dtype(y_orig)
    target_mapping = None
    if not is_regression and (not target_is_numeric):
        y_cat = pd.Categorical(y_orig)
        target_mapping = list(y_cat.categories)
        y = pd.Series(y_cat.codes, index=y_orig.index, name=target)
    else:
        y = y_orig.copy()
    
    # --- Process numeric features ---
    X_num = X[num_cols].copy()
    if is_regression:
        # Append the target to numeric features so that scaling can be inverted later.
        X_num_target = pd.concat([X_num, y.rename("target")], axis=1)
        scaler = MinMaxScaler().fit(X_num_target)
        X_num_scaled = pd.DataFrame(scaler.transform(X_num_target),
                                    columns = list(X_num.columns) + ["target"],
                                    index=X_num_target.index)
    else:
        scaler = MinMaxScaler().fit(X_num)
        X_num_scaled = pd.DataFrame(scaler.transform(X_num),
                                    columns = X_num.columns,
                                    index=X_num.index)
    
    # --- Process categorical features ---
    cat_mapping = {}  # to store the mapping from codes back to original categories
    if len(cat_cols) > 0:
        X_cat = X[cat_cols].copy()
        X_cat_encoded = X_cat.copy()
        for col in X_cat.columns:
            cat_series = pd.Categorical(X_cat[col])
            cat_mapping[col] = list(cat_series.categories)
            X_cat_encoded[col] = cat_series.codes
        # Make sure categorical values are integer.
        X_cat_encoded = X_cat_encoded.astype(int)
    else:
        X_cat_encoded = None
    
    # --- Combine numeric and categorical features for SMOTE ---
    if X_cat_encoded is not None:
        # When concatenating, numeric columns come first.
        X_combined = pd.concat([X_num_scaled.reset_index(drop=True),
                                X_cat_encoded.reset_index(drop=True)], axis=1)
    else:
        X_combined = X_num_scaled.reset_index(drop=True)
    
    # Convert to NumPy array for SMOTE.
    X_array = X_combined.to_numpy()
    
    # Determine categorical feature indices (in the combined array, these come after numeric columns).
    if X_cat_encoded is not None:
        cat_indices = list(range(X_num_scaled.shape[1], X_num_scaled.shape[1] + len(X_cat_encoded.columns)))
    else:
        cat_indices = None
    
    # --- Prepare target for SMOTE ---
    if is_regression:
        # For regression, use a binary version of the target for SMOTE (thresholded by the median).
        median_val = df[target].median()
        y_for_smote = (df[target] > median_val).astype(int).to_numpy()
    else:
        y_for_smote = y.to_numpy()
    
    # --- Define the sampling strategy ---
    classes = np.unique(y_for_smote)
    
    if n_samples == -1:
        # Use original logic with frac_samples
        strat = {cls: int((1 + frac_samples) * np.sum(y_for_smote == cls)) for cls in classes}
    else:
        # Distribute n_samples across all classes proportionally to their current representation
        if eval_type == 'real':
            # For 'real' mode, n_samples should be the total size including original data
            original_size = len(y_for_smote)
            synthetic_samples = max(0, n_samples - original_size)
            if synthetic_samples == 0:
                strat = {cls: np.sum(y_for_smote == cls) for cls in classes}
            else:
                # Distribute synthetic samples proportionally
                class_counts = {cls: np.sum(y_for_smote == cls) for cls in classes}
                total_current = sum(class_counts.values())
                strat = {}
                for cls in classes:
                    proportion = class_counts[cls] / total_current
                    additional_samples = int(proportion * synthetic_samples)
                    strat[cls] = class_counts[cls] + additional_samples
        else:
            # For 'synthetic' mode, n_samples is the total number of synthetic samples to generate
            class_counts = {cls: np.sum(y_for_smote == cls) for cls in classes}
            total_current = sum(class_counts.values())
            strat = {}
            for cls in classes:
                proportion = class_counts[cls] / total_current
                additional_samples = int(proportion * n_samples)
                strat[cls] = class_counts[cls] + additional_samples
    
    # Set the lam parameters.
    lam1 = 0.0 + frac_lam_del / 2
    lam2 = 1.0 - frac_lam_del / 2
    
    # --- Apply SMOTE ---
    if eval_type != 'real':
        if cat_indices is not None:
            sm = MySMOTENC(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbors,
                categorical_features=cat_indices,
                sampling_strategy=strat
            )
        else:
            sm = MySMOTE(
                lam1=lam1,
                lam2=lam2,
                random_state=seed,
                k_neighbors=k_neighbors,
                sampling_strategy=strat
            )
        print("SMOTE training data shape:", X_array.shape, y_for_smote.shape)
        X_res, y_res = sm.fit_resample(X_array, y_for_smote)
        # Keep only the synthetic samples (those added after the original data).
        original_size = X_array.shape[0]
        X_res = X_res[original_size:]
        y_res = y_res[original_size:]
    else:
        X_res, y_res = X_array, y_for_smote

    # --- Inverse-transform the numeric features ---
    if is_regression:
        # The numeric part includes the appended target as the last column.
        n_num = len(X_num.columns) + 1
        X_num_res_scaled = X_res[:, :n_num]
        X_num_res = scaler.inverse_transform(X_num_res_scaled)
        # Separate numeric features and the target.
        numeric_features_res = X_num_res[:, :-1]
        target_res_numeric = X_num_res[:, -1]
    else:
        n_num = len(X_num.columns)
        X_num_res_scaled = X_res[:, :n_num]
        X_num_res = scaler.inverse_transform(X_num_res_scaled)
        numeric_features_res = X_num_res

    # --- Process categorical features from SMOTE output ---
    if X_cat_encoded is not None:
        X_cat_res = X_res[:, n_num:]
    else:
        X_cat_res = None

    # --- Reconstruct features with original column order and data types ---
    # Create DataFrames for numeric and categorical parts.
    num_df = pd.DataFrame(numeric_features_res, columns=num_cols)
    # Convert numeric columns to original dtypes (round if integer).
    for col in num_df.columns:
        orig_dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(orig_dtype):
            num_df[col] = num_df[col].round().astype(orig_dtype)
        else:
            num_df[col] = num_df[col].astype(orig_dtype)
    
    if X_cat_res is not None:
        cat_df = pd.DataFrame(X_cat_res, columns=cat_cols)
        # Map the encoded integers back to original categories.
        for col in cat_df.columns:
            cats = cat_mapping[col]
            # Use from_codes to recover the categorical series.
            cat_series = pd.Categorical.from_codes(cat_df[col].astype(int), categories=cats)
            # If the original column was of type 'category', preserve that type.
            if str(df[col].dtype) == 'category':
                cat_df[col] = cat_series
            else:
                cat_df[col] = cat_series.astype(df[col].dtype)
    else:
        cat_df = pd.DataFrame(index=num_df.index)
    
    # Reassemble the features in their original order.
    features_res = {}
    for col in features_order:
        if col in num_df.columns:
            features_res[col] = num_df[col]
        elif col in cat_df.columns:
            features_res[col] = cat_df[col]
    features_res_df = pd.DataFrame(features_res)
    # Ensure the column order matches the original.
    features_res_df = features_res_df[features_order]
    
    # --- Process the target column ---
    if is_regression:
        target_res = target_res_numeric
        if pd.api.types.is_integer_dtype(original_target_dtype):
            target_res = np.rint(target_res).astype(original_target_dtype)
        else:
            target_res = target_res.astype(original_target_dtype)
        target_res_series = pd.Series(target_res, name=target)
    else:
        if target_mapping is not None:
            # Decode the target codes back to the original categorical values.
            target_res_series = pd.Categorical.from_codes(y_res.astype(int), categories=target_mapping)
            if str(df[target].dtype) == 'category':
                target_res_series = pd.Series(target_res_series, name=target, dtype="category")
            else:
                target_res_series = pd.Series(target_res_series.astype(df[target].dtype), name=target)
        else:
            target_res_series = pd.Series(y_res.astype(original_target_dtype), name=target)
    
    # --- Combine features and target in the original column order ---
    # Determine the original full column order.
    final_columns = list(df.columns)
    res_df = features_res_df.copy()
    res_df[target] = target_res_series.values
    # Reorder columns to match the input.
    res_df = res_df[final_columns]
    
    # Create save directory if it doesn't exist and save is True
    if save and save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Get the input filename without extension
        input_name = Path(df.attrs.get('filename', 'resampled')).stem
        output_filename = f"{input_name}_SMOTE_default_1.csv"
        output_path = save_path / output_filename
        
        # Save the resampled data
        res_df.to_csv(output_path, index=False)
        print(f"Saved resampled data to: {output_path}")
    
    return res_df

# -----------------------------------------------------------------------------
# Main: Parse command-line arguments and run sample_smote
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply SMOTE to a CSV file and output a resampled DataFrame."
    )
    parser.add_argument('csv_path', type=str, help="Path to the input CSV file.")
    parser.add_argument('target', type=str, help="Name of the target column.")
    parser.add_argument('--categorical', type=str, default="",
                        help="Comma-separated list of feature names to treat as categorical.")
    parser.add_argument('--force_categorical', type=str, default="",
                        help="Comma-separated list of column names to force convert to categorical type before processing.")
    parser.add_argument('--eval_type', type=str, default="synthetic",
                        help="Either 'synthetic' (only new synthetic samples) or 'real' (entire resampled dataset).")
    parser.add_argument('--k_neighbors', type=int, default=5, help="Number of neighbors for SMOTE.")
    parser.add_argument('--frac_samples', type=float, default=1.0, help="Scaling factor for resampling class counts.")
    parser.add_argument('--frac_lam_del', type=float, default=0.0, help="Fraction to narrow the [lam1, lam2] interval.")
    parser.add_argument('--is_regression', action='store_true',
                        help="Set this flag if the task is regression.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--save', action='store_true', help="If set, the resampled output is saved to CSV.")
    parser.add_argument('--save_path', type=str, default="outputs", help="Directory to save the CSV if --save is set.")
    parser.add_argument('--n_samples', type=int, default=-1, help="Total number of data points to resample. Default -1 means same size as input df.")
    args = parser.parse_args()
    
    # Read CSV file into a DataFrame
    df = pd.read_csv(args.csv_path)
    # Store the input filename as an attribute
    df.attrs['filename'] = Path(args.csv_path).name
    
    # Handle missing values
    print("Checking for missing values...")
    if df.isnull().any().any():
        print("Found missing values. Imputing...")
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns with mean
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Imputed {df[col].isnull().sum()} missing values in {col} with mean: {mean_val:.2f}")
        
        # Impute categorical columns with mode (most frequent)
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Imputed {df[col].isnull().sum()} missing values in {col} with mode: {mode_val}")
    else:
        print("No missing values found.")
    
    # Force convert specified columns to categorical type
    if args.force_categorical:
        force_cat_list = [col.strip() for col in args.force_categorical.split(',')]
        print(f"Force converting columns to categorical: {force_cat_list}")
        for col in force_cat_list:
            if col in df.columns:
                df[col] = df[col].astype('category')
                print(f"Converted column '{col}' to categorical type")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
    
    if args.categorical:
        cat_list = [col.strip() for col in args.categorical.split(',')]
    else:
        cat_list = None

    if "SOURCE_LABEL" in df.columns:
        print("Dropping SOURCE_LABEL column...")
        df = df[df['SOURCE_LABEL'] == 'train']
        df.drop(columns=["SOURCE_LABEL"], inplace=True)

    if args.target == "LAST_COLUMN":
        target = df.columns[-1]
        # If the last column is numerical or doesn't have at least 2 unique values,
        # find the rightmost categorical column with at least 2 unique values
        if pd.api.types.is_numeric_dtype(df[target]) or df[target].nunique() < 2:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            print("categorical_cols", categorical_cols)
            found_valid_target = False
            
            if len(categorical_cols) > 0:
                # Search from right to left for a categorical column with at least 2 unique values
                for col in reversed(categorical_cols):
                    if df[col].nunique() >= 2:
                        target = col
                        print(f"Using rightmost categorical column '{target}' with {df[target].nunique()} unique values as target.")
                        found_valid_target = True
                        break
                
                if not found_valid_target:
                    raise ValueError("No categorical columns found with at least 2 unique values.")
            else:
                raise ValueError("No categorical columns found in the dataset.")
    else:
        target = args.target

    # Print initial value counts of target column
    print("\nInitial target column distribution:")
    value_counts = df[target].value_counts()
    print(value_counts)
    print(f"\nInitial number of classes: {len(value_counts)}")
    print(f"Initial smallest class size: {value_counts.min()}")
    print(f"Initial largest class size: {value_counts.max()}")

    # Filter out rare categories
    min_samples_needed = args.k_neighbors + 1
    rare_categories = value_counts[value_counts < min_samples_needed].index
    if len(rare_categories) > 0:
        print(f"\nRemoving {len(rare_categories)} categories with fewer than {min_samples_needed} samples:")
        for cat in rare_categories:
            print(f"- {cat}: {value_counts[cat]} samples")
        
        # Drop rows with rare categories
        original_size = len(df)
        df = df[~df[target].isin(rare_categories)]
        print(f"\nRemoved {original_size - len(df)} rows with rare categories")
        
        # Print updated value counts
        print("\nUpdated target column distribution:")
        value_counts = df[target].value_counts()
        print(value_counts)
        print(f"\nUpdated number of classes: {len(value_counts)}")
        print(f"Updated smallest class size: {value_counts.min()}")
        print(f"Updated largest class size: {value_counts.max()}")

    print("SMOTE training data shape:", df.shape)
    
    resampled_df = sample_smote(
        df,
        target=target,
        categorical_features=cat_list,
        eval_type=args.eval_type,
        k_neighbors=args.k_neighbors,
        frac_samples=args.frac_samples,
        frac_lam_del=args.frac_lam_del,
        is_regression=args.is_regression,
        seed=args.seed,
        save=args.save,
        save_path=args.save_path,
        n_samples=args.n_samples
    )
    
    print("Resampled data shape:", resampled_df.shape)
    print("Resampled data head:\n", resampled_df.head())

if __name__ == '__main__':
    main()
