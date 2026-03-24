import os
import json
import numpy as np
import pandas as pd
import torch
from .vae.model import Decoder_model

# Fix relative import issue for subprocess execution
try:
    from ..utils_train import preprocess
except ImportError:
    # Subprocess context: load utils_train module directly
    import os
    import importlib.util
    tabsyn_dir = os.path.dirname(os.path.dirname(__file__))
    utils_train_path = os.path.join(tabsyn_dir, 'utils_train.py')
    if os.path.exists(utils_train_path):
        spec = importlib.util.spec_from_file_location("utils_train", utils_train_path)
        utils_train_module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules["utils_train"] = utils_train_module
        spec.loader.exec_module(utils_train_module)
        from utils_train import preprocess
    else:
        raise ImportError("Could not find utils_train module in subprocess context") 

def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # Align with dataset prepared by process_dataset (under TabSyn/data/<name>)
    base_dir = os.path.abspath(os.path.join(curr_dir, '..'))
    dataset_dir = os.path.join(base_dir, 'data', dataname)

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'data/TabSyn/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(curr_dir, '..'))
    dataset_dir = os.path.join(base_dir, 'data', dataname)
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    with open(os.path.join(dataset_dir, 'info.json'), 'r') as f:
        info = json.load(f)

    task_type = info['task_type']


    ckpt_dir = f'{curr_dir}/ckpt/{dataname}'

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(dataset_dir, task_type = task_type, inverse = True)

    embedding_save_path = f'data/TabSyn/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)

    decoder_save_path = f'data/TabSyn/ckpt/{dataname}/decoder.pt'
    pre_decoder.load_state_dict(torch.load(decoder_save_path, weights_only=False))

    info['pre_decoder'] = pre_decoder
    info['token_dim'] = token_dim

    return train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


 
@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, cat_inverse, device):
    task_type = info['task_type']

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    pre_decoder = info['pre_decoder']
    token_dim = info['token_dim']

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)
    
    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    # Initialize syn_num and syn_cat as empty tensors if there are no numerical or categorical features
    if n_num_feat > 0:
        syn_num = x_hat_num.cpu().numpy()
        syn_num = num_inverse(syn_num)
    else:
        syn_num = torch.empty((syn_data.shape[0], 0)).cpu().numpy()  # Empty array for syn_num

    if n_cat_feat > 0:
        syn_cat = []
        for pred in x_hat_cat:
            syn_cat.append(pred.argmax(dim=-1))

        # Handle case where x_hat_cat is empty (no actual categorical features, only categorical target)
        if len(syn_cat) > 0:
            syn_cat = torch.stack(syn_cat).t().cpu().numpy()
            syn_cat = cat_inverse(syn_cat)
        else:
            syn_cat = torch.empty((syn_data.shape[0], 0)).cpu().numpy()
    else:
        syn_cat = torch.empty((syn_data.shape[0], 0)).cpu().numpy()  # Empty array for syn_cat

    # Split target and features based on task type
    if task_type == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):] if n_num_feat > 0 else syn_num
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):] if n_cat_feat > 0 else syn_cat

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']


    idx_mapping = info['idx_mapping']
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}
    print(num_col_idx, cat_col_idx, target_col_idx)
    print(idx_mapping)

    syn_df = pd.DataFrame()

    if info['task_type'] == 'regression':
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]] 
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat
