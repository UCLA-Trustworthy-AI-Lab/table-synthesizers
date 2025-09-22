from copy import deepcopy
import torch
import os
import numpy as np
import zero
from ..tab_ddpm import GaussianMultinomialDiffusion
from .utils_train import get_model, update_ema
import pandas as pd


class InfiniteDataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # If the iterator is exhausted, create a new iterator
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

def process_encoded_table(tensor, data_info, target):
    transform_info = data_info['transform_info']
    
    X_num = []
    X_cat = []
    y = None
    
    # Split the tensor into X_num, X_cat, y
    for column_name, info in transform_info.items():
        start_idx = info['start_idx']
        end_idx = info['end_idx']
        
        if column_name == target:
            y = tensor[:, start_idx:end_idx]
            y_original_dtype = info['original_dtype']
        else:
            if any([t in info['original_dtype'] for t in ['numerical', 'ordinal', 'continuous', 'datetime']]):
                X_num.append(tensor[:, start_idx:end_idx])
                #print("appending to X_num:", column_name, tensor[:, start_idx:end_idx].shape, start_idx, end_idx)
            else:
                X_cat.append(tensor[:, start_idx:end_idx])
    
    
    # Concatenate tensors along the column axis
    if X_num:
        X_num = torch.cat(X_num, dim=1)
    else:
        X_num = None
    #print("X_num shape:", X_num.shape)
    
    if X_cat:
        X_cat = torch.cat(X_cat, dim=1)
    else:
        X_cat = None
    
    #print(type(X_num), type(X_cat), type(y))
    if y_original_dtype in ['numerical', 'ordinal','continuous', 'datetime']:
        # Concatenate X_num, y, X_cat
        if X_num is not None:
            X_num = torch.cat([X_num, y], dim=1)
            #print("concating y to x_num!")
            #print("X_num shape", X_num.shape)
        else:
            X_num = y
        out_dict = torch.tensor([])
        if X_cat is not None:
            X_features = torch.cat([X_num, X_cat], dim=1)
        else:
            X_features = X_num
    else:
        # Concatenate X_num, X_cat and return (X_num, X_cat), y
        if X_num is not None and X_cat is not None:
            X_features = torch.cat([X_num, X_cat], dim=1)
        elif X_num is not None:
            X_features = X_num
        else:
            X_features = X_cat
        out_dict = y
    #print("X_features.shape",X_features.shape)
    return X_features, out_dict

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, data_info, target, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = InfiniteDataLoaderIterator(train_iter) # Use custom iterator to loop indefinitely
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

        self.data_info = data_info
        self.target = target

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        print("self.steps:",self.steps)
        while step < self.steps:
            data = next(self.train_iter) # Expect input is an encoded tensor, but target, num and cate features are mixed together
            # we split and reorder them. 
            #print("Data.shape", data.shape)
            x, out_dict = process_encoded_table(data, self.data_info, self.target)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    train_loader,
    data_info,
    target=None,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    device = None,
    seed = 0,
):
    """
        data_info: dict with fomat  {transform_info: {column_name: {original_dtype, start_idx, end_idx, transformed_dtypes, empirical_dist}, encoded_width, integer}
    """
    #parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category_sizes = []
    num_dtypes = ['numerical', 'ordinal','continuous', 'datetime']
    num_numerical_features = 0
    for i,element in enumerate(data_info['transform_info'].items()):
        column,info = element
        if info['original_dtype'] in num_dtypes:
            num_numerical_features += 1
        else:
            # Only add to category_sizes if it's not the target column
            if i != len(data_info['transform_info']) - 1:  # Not the last column (target)
                category_sizes.append(len(info['empirical_dist']))
        if i == len(data_info['transform_info']) - 1 and target is None:
            target = column
        if target == column:
            empirical_class_dist = torch.tensor(info['empirical_dist'])
            print("Empirical dist of target is recorded!")
            print(type(empirical_class_dist), empirical_class_dist.shape)
    K = np.array(category_sizes)
    if len(K) == 0:
        K = np.array([0])
    print("K value in train:",K)

    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    model_params['num_classes'] = len(empirical_class_dist)
    model_params['is_y_cond'] = data_info['transform_info'][target]['original_dtype'] not in num_dtypes
    
    print("Model params in train")
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=category_sizes
    )
    model.to(device)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        data_info=data_info,
        target=target,
        device=device
    )
    trainer.run_loop()

    return diffusion._denoise_fn, trainer.ema_model, empirical_class_dist

    #trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    #torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    #torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))