import torch
import numpy as np
import sys
# Apply zero workaround
from stg import zero_workaround as zero
sys.modules['zero'] = zero
from ..tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .utils_train import get_model


num_col_types = ['numerical', 'ordinal','continuous', 'datetime']

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def combine_columns(X_features, y, data_info, target, num_cols):
    X_features = torch.tensor(X_features)
    y = torch.tensor(y)
    print("y.shape",y.shape, y[:20])
    
    transform_info = data_info['transform_info']
    total_columns = sum(info['end_idx'] - info['start_idx'] for info in transform_info.values())
    #print("Desired shape:",(int(X_features.shape[0]), int(total_columns)))

    # Initialize a new tensor with the same number of columns as the original tensor
    combined_tensor = torch.zeros((int(X_features.shape[0]), int(total_columns)), dtype=X_features.dtype)
    #combined_tensor = torch.zeros((int(X_features.shape[0]), int(total_columns)))
    
    cur_num_idx, cur_cat_idx = 0, num_cols
    
    # Place the X_features and y values into the appropriate columns
    for column_name, info in transform_info.items():
        start_idx = info['start_idx']
        end_idx = info['end_idx']
        is_num = info['original_dtype'] in num_col_types
        col_range = range(start_idx, end_idx)
        print(column_name,start_idx, end_idx)
        
        if column_name == target:
            combined_tensor[:, start_idx:end_idx] = y.unsqueeze(1)
        elif is_num:
            combined_tensor[:, col_range] = X_features[:, cur_num_idx:cur_num_idx + len(col_range)]
            cur_num_idx += len(col_range)
        else:
            combined_tensor[:, col_range] = X_features[:, cur_cat_idx:cur_cat_idx + len(col_range)]
            cur_cat_idx += len(col_range)
    
    return combined_tensor

def sample(
    data_info,
    target,
    empirical_class_dist,
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    diffusion_fn_state = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    disbalance = None,
    device = None,
    seed = 0,
):
    zero.improve_reproducibility(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    category_sizes = []
    num_numerical_features_ = 0
    target_idx = 0
    disc_cols = []
    for i,element in enumerate(data_info['transform_info'].items()):
        column,info = element
        if info['original_dtype'] == 'orginal':
            disc_cols.append(num_numerical_features_)
        if info['original_dtype'] in num_col_types:
            num_numerical_features_ += 1
        elif i < len(data_info['transform_info']) - 1:
            category_sizes.append(len(info['empirical_dist'])) # only append non-target cat variables
    
        if i == len(data_info['transform_info']) - 1 and target is None:
            target = column
    K = np.array(category_sizes)
    if len(K) == 0:
        K = np.array([0])

    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model_params['num_classes'] = len(empirical_class_dist)
    model_params['is_y_cond'] = data_info['transform_info'][target]['original_dtype'] not in num_col_types

    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=category_sizes
    )

    model.load_state_dict(
        diffusion_fn_state
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()


    num_numerical_features = num_numerical_features_ + int(data_info['transform_info'][target] in num_col_types)
    print(num_numerical_features)

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        #np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        #if T_dict['cat_encoding'] == 'one-hot':
        #    X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        #X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])
        X_cat = X_gen[:, num_numerical_features:]
    else:
        X_cat = None

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        #np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        #X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        #X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        print("data_info['transform_info'].keys()")
        print(data_info['transform_info'])
        
        disc_cols = []
        for c, info in data_info['transform_info'].items():
            if info['original_dtype'] in ['ordinal']:
                disc_cols.append(info['start_idx'])
        #print("Discrete cols:", disc_cols)
        #print("data_info['transform_info'][target]", data_info['transform_info'][target])
        if data_info['transform_info'][target]['original_dtype'] in num_col_types:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            #X_num = round_columns(X_num_real, X_num, disc_cols)
            for c in disc_cols:
                X_num[:,c] = np.round(X_num[:, c])

    if X_cat is not None:
        print(X_num.shape, X_cat.shape)
        X_features = np.concatenate([X_num, X_cat],axis=1)
    else: 
        X_features = X_num
    return combine_columns(X_features, y_gen, data_info, target, X_num.shape[1])

    #if num_numerical_features != 0:
    #    print("Num shape: ", X_num.shape)
    #    np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    #if num_numerical_features < X_gen.shape[1]:
    #    np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    #np.save(os.path.join(parent_dir, 'y_train'), y_gen)