import torch
import torch.nn as nn
import math
import numpy as np


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def pate(data, teachers, lap_scale, device="cpu"):
    """PATE implementation for GANs.
    """
    num_teachers = len(teachers)
    # First check the actual output shape to determine proper label tensor size
    with torch.no_grad():
        sample_output = teachers[0](data)
        actual_output_size = sample_output.shape[0]

    labels = torch.Tensor(num_teachers, actual_output_size).type(torch.int64).to(device)
    for i in range(num_teachers):
        output = teachers[i](data)
        pred = (output > 0.5).type(torch.Tensor).squeeze().to(device)
        # Ensure pred matches the expected size
        if pred.numel() != actual_output_size:
            # If shapes don't match, expand pred to match data.shape[0] by repeating
            pred = pred.repeat(data.shape[0] // pred.numel())[:data.shape[0]]
        labels[i] = pred

    votes = torch.sum(labels, dim=0).unsqueeze(1).type(torch.DoubleTensor).to(device)
    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1 / lap_scale, size=votes.size())).to(
        device
    )
    noisy_votes = votes + noise
    noisy_labels = (noisy_votes > num_teachers / 2).type(torch.DoubleTensor).to(device)

    return noisy_labels, votes


def moments_acc(num_teachers, votes, lap_scale, l_list, device="cpu"):
    q = (2 + lap_scale * torch.abs(2 * votes - num_teachers)) / (
        4 * torch.exp(lap_scale * torch.abs(2 * votes - num_teachers))
    ).to(device)

    alpha = []
    for l_val in l_list:
        a = 2 * lap_scale ** 2 * l_val * (l_val + 1)
        t_one = (1 - q) * torch.pow((1 - q) / (1 - math.exp(2 * lap_scale) * q), l_val)
        t_two = q * torch.exp(2 * lap_scale * l_val)
        t = t_one + t_two
        alpha.append(torch.clamp(t, max=a).sum())

    return torch.DoubleTensor(alpha).to(device)