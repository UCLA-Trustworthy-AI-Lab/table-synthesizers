import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
from sklearn.model_selection import train_test_split
import warnings

TYPE_TRANSFORM ={
    'float', np.float32,
    'str', str,
    'int', int
}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INFO_PATH = os.path.join(BASE_DIR, 'data', 'Info')

parser = argparse.ArgumentParser(description='process dataset')

# General configs
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')

# Only parse args if this file is run directly, not imported
if __name__ == '__main__':
    args = parser.parse_args()
else:
    args = None

def preprocess_beijing():
    with open(os.path.join(INFO_PATH, 'beijing.json'), 'r') as f:
        info = json.load(f)
    
    data_path = info['raw_data_path']

    data_df = pd.read_csv(data_path)
    columns = data_df.columns

    data_df = data_df[columns[1:]]


    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info['data_path'], index = False)

def preprocess_news():
    with open(os.path.join(INFO_PATH, 'news.json'), 'r') as f:
        info = json.load(f)

    data_path = info['raw_data_path']
    data_df = pd.read_csv(data_path)
    data_df = data_df.drop('url', axis=1)

    columns = np.array(data_df.columns.tolist())

    cat_columns1 = columns[list(range(12,18))]
    cat_columns2 = columns[list(range(30,38))]

    cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis = 1)
    cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis = 1)

    data_df = data_df.drop(cat_columns2, axis=1)
    data_df = data_df.drop(cat_columns1, axis=1)

    data_df['data_channel'] = cat_col1
    data_df['weekday'] = cat_col2
    
    data_save_path = os.path.join(BASE_DIR, 'data', 'news', 'news.csv')
    data_df.to_csv(f'{data_save_path}', index = False)

    columns = np.array(data_df.columns.tolist())
    num_columns = columns[list(range(45))]
    cat_columns = ['data_channel', 'weekday']
    target_columns = columns[[45]]

    info['num_col_idx'] = list(range(45))
    info['cat_col_idx'] = [46, 47]
    info['target_col_idx'] = [45]
    info['data_path'] = data_save_path
    
    name = 'news'
    with open(os.path.join(INFO_PATH, f'{name}.json'), 'w') as file:
        json.dump(info, file, indent=4)


def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0, max_attempts=1000):
    """
    Robust train_val_test_split that prevents infinite loops.
    
    Strategies implemented:
    1. Stratified sampling for categorical variables (when possible)
    2. Limited attempts with fallback (prevents infinite loops)
    3. Best-effort category preservation
    4. Graceful degradation for edge cases
    
    Args:
        data_df: Input DataFrame
        cat_columns: List of categorical column names
        num_train: Number of training samples
        num_test: Number of test samples
        max_attempts: Maximum random attempts before fallback (default: 1000)
    """
    total_num = data_df.shape[0]
    
    # Validation checks
    if num_train + num_test > total_num:
        raise ValueError(f"Requested samples ({num_train + num_test}) exceed dataset size ({total_num})")
    
    if num_train == 0 or num_test == 0:
        raise ValueError("Both num_train and num_test must be > 0")
    
    print(f"TabSyn split: train={num_train}, test={num_test}, total={total_num}")
    
    # Strategy 1: Try stratified sampling if we have categorical columns
    if cat_columns:
        try:
            # Create stratification key from categorical columns
            if len(cat_columns) == 1:
                stratify_col = data_df[cat_columns[0]]
            else:
                # Combine multiple categorical columns
                stratify_col = data_df[cat_columns].apply(
                    lambda x: '_'.join(x.astype(str)), axis=1
                )
            
            # Check if stratification is viable
            value_counts = stratify_col.value_counts()
            min_count = value_counts.min()
            
            if min_count >= 2:  # Need at least 2 samples per category
                test_size = num_test / total_num
                train_idx, test_idx = train_test_split(
                    range(total_num), 
                    test_size=test_size, 
                    stratify=stratify_col, 
                    random_state=42
                )
                
                # Adjust for exact sizes
                if len(train_idx) != num_train:
                    diff = num_train - len(train_idx)
                    if diff > 0 and len(test_idx) >= diff:
                        # Move samples from test to train
                        extra_idx = test_idx[:diff]
                        train_idx = np.concatenate([train_idx, extra_idx])
                        test_idx = test_idx[diff:]
                    elif diff < 0 and abs(diff) <= len(train_idx):
                        # Move samples from train to test
                        extra_idx = train_idx[num_train:]
                        train_idx = train_idx[:num_train]
                        test_idx = np.concatenate([test_idx, extra_idx])
                
                train_df = data_df.iloc[train_idx].copy()
                test_df = data_df.iloc[test_idx].copy()
                
                # Check if all categories are preserved
                categories_preserved = True
                for col in cat_columns:
                    train_cats = set(train_df[col])
                    all_cats = set(data_df[col])
                    if train_cats != all_cats:
                        categories_preserved = False
                        break
                
                if categories_preserved:
                    print("✅ Stratified sampling successful - all categories preserved")
                    return train_df, test_df, 42
                else:
                    print("⚠️ Stratified sampling preserved most but not all categories")
                    
        except Exception as e:
            print(f"⚠️ Stratified sampling failed: {e}, using random sampling...")
    
    # Strategy 2: Limited random attempts (prevents infinite loops)
    print(f"Trying random sampling with up to {max_attempts} attempts...")
    
    best_train_df, best_test_df, best_seed = None, None, None
    best_missing_categories = float('inf')
    
    for attempt in range(max_attempts):
        seed = 1234 + attempt
        np.random.seed(seed)
        
        # Random shuffle and split
        idx = np.arange(total_num)
        np.random.shuffle(idx)
        
        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]
        
        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]
        
        # Check category preservation
        missing_categories = 0
        for col in cat_columns:
            train_cats = set(train_df[col])
            all_cats = set(data_df[col])
            missing_categories += len(all_cats - train_cats)
        
        # Perfect split found
        if missing_categories == 0:
            print(f"✅ Perfect split found at attempt {attempt + 1}")
            return train_df, test_df, seed
        
        # Track best attempt
        if missing_categories < best_missing_categories:
            best_missing_categories = missing_categories
            best_train_df, best_test_df, best_seed = train_df.copy(), test_df.copy(), seed
        
        # Early stopping for reasonable splits
        if missing_categories <= 2 and attempt >= 100:
            print(f"✅ Good split found at attempt {attempt + 1} (missing {missing_categories} categories)")
            return train_df, test_df, seed
        
        # Progress update
        if attempt % 200 == 199:
            print(f"   Attempt {attempt + 1}/{max_attempts}, best missing: {best_missing_categories}")
    
    # Strategy 3: Use best split found (prevents infinite loops)
    if best_train_df is not None:
        print(f"🔄 Using best available split (missing {best_missing_categories} categories)")
        
        if best_missing_categories > 0:
            print("⚠️ WARNING: Not all categorical values present in training set")
            print("   TabSyn may have reduced performance on unseen categories")
            
            # Report missing categories
            for col in cat_columns:
                train_cats = set(best_train_df[col])
                all_cats = set(data_df[col])
                missing = all_cats - train_cats
                if missing:
                    print(f"   {col}: missing {len(missing)} categories")
                    if len(missing) <= 5:  # Only show details for small numbers
                        print(f"      Missing: {missing}")
        
        return best_train_df, best_test_df, best_seed
    
    # Strategy 4: Emergency fallback (should never reach here)
    print("🚨 EMERGENCY FALLBACK: Using simple random split")
    warnings.warn(
        "Could not preserve categorical distributions after maximum attempts. "
        "TabSyn may have significantly reduced performance."
    )
    
    np.random.seed(1234)
    idx = np.arange(total_num)
    np.random.shuffle(idx)
    
    train_idx = idx[:num_train]  
    test_idx = idx[-num_test:]
    
    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]
    
    return train_df, test_df, 1234    


def process_data(name):

    if name == 'news':
        preprocess_news()
    elif name == 'beijing':
        preprocess_beijing()

    with open(os.path.join(INFO_PATH, f'{name}.json'), 'r') as f:
        info = json.load(f)

    data_path = info['data_path']
    print("info:", info)
    if info['file_type'] == 'csv':
        data_df = pd.read_csv(data_path, header = info['header'])

    elif info['file_type'] == 'xls':
        data_df = pd.read_excel(data_path, sheet_name='Data', header=1)
        data_df = data_df.drop('ID', axis=1)

    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    print("target_columns:",target_columns)

    if info['test_path']:

        # if testing data is given
        test_path = info['test_path']

        with open(test_path, 'r') as f:
            lines = f.readlines()[1:]
            test_save_path = os.path.join(BASE_DIR, 'data', name, 'test.data')
            if not os.path.exists(test_save_path):
                with open(test_save_path, 'a') as f1:     
                    for line in lines:
                        save_line = line.strip('\n').strip('.')
                        f1.write(f'{save_line}\n')

        test_df = pd.read_csv(test_save_path, header = None)
        train_df = data_df

    else:  
        # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)

        num_train = int(num_data*0.9)
        num_test = num_data - num_train
        print(num_train,num_test)

        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    print(name, train_df.shape, test_df.shape, data_df.shape)

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info
    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'


    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()


    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    print("NP array created!")

 
    save_dir = os.path.join(BASE_DIR, 'data', name)
    np.save(os.path.join(save_dir, 'X_num_train.npy'), X_num_train)
    np.save(os.path.join(save_dir, 'X_cat_train.npy'), X_cat_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)

    np.save(os.path.join(save_dir, 'X_num_test.npy'), X_num_test)
    np.save(os.path.join(save_dir, 'X_cat_test.npy'), X_cat_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)


    train_df.to_csv(os.path.join(save_dir, 'train.csv'), index = False)
    test_df.to_csv(os.path.join(save_dir, 'test.csv'), index = False)

    synthetic_dir = os.path.join(BASE_DIR, 'synthetic', name)
    if not os.path.exists(synthetic_dir):
        os.makedirs(synthetic_dir)
    
    train_df.to_csv(os.path.join(synthetic_dir, 'real.csv'), index = False)
    test_df.to_csv(os.path.join(synthetic_dir, 'test.csv'), index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(os.path.join(save_dir, 'info.json'), 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)


if __name__ == "__main__":

    if args.dataname:
        process_data(args.dataname)
    else:
        for name in ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news']:    
            process_data(name)

        
