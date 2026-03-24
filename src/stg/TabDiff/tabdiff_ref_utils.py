"""
Utilities adapted from the upstream TabDiff repository.

Source:
https://github.com/MinkaiXu/TabDiff
File adapted from: process_dataset.py (MIT License)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_column_name_mapping(
    data_df: pd.DataFrame,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    target_col_idx: List[int],
    column_names: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """
    Build index/name mappings used by TabDiff-style preprocessing.

    Returns:
        idx_mapping: Original column index -> grouped index
        inverse_idx_mapping: Grouped index -> original column index
        idx_name_mapping: Original index -> column name
    """
    if column_names is None:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping: Dict[int, int] = {}

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
        elif idx in target_col_idx:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1
        else:
            # Keep behavior stable when target columns are not explicitly provided.
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping: Dict[int, int] = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping: Dict[int, str] = {}
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = str(column_names[i])

    return idx_mapping, inverse_idx_mapping, idx_name_mapping
