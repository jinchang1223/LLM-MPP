import numpy as np
import torch
import torch.nn as nn
from configure import *
from sklearn.metrics import roc_auc_score, mean_squared_error
import sys
import torch.nn.functional as F


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).unsqueeze(1)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
    
def rmse(y_pred_list, y_true_list):
    y_true_list = y_true_list.numpy()
    y_pred_list = y_pred_list.numpy()
    if y_true_list.ndim == 1:
        y_true_list = y_true_list.reshape(-1, 1)
        y_pred_list = y_pred_list.reshape(-1, 1)
    elif y_true_list.ndim == 3:
        y_true_list = np.squeeze(y_true_list, axis=-1)
        y_pred_list = np.squeeze(y_pred_list, axis=-1)
        
    # mse_loss_fn = nn.MSELoss()
    rmse_loss = mean_squared_error(y_true_list, y_pred_list, squared=False)
    return rmse_loss


def roc_auc_function(y_pred_list, y_true_list):
    y_true_list = y_true_list.numpy()
    y_pred_list = y_pred_list.numpy()
    if y_true_list.ndim == 1:
        y_true_list = y_true_list.reshape(-1, 1)
        y_pred_list = y_pred_list.reshape(-1, 1)
    elif y_true_list.ndim == 3:
        y_true_list = np.squeeze(y_true_list, axis=-1)
        y_pred_list = np.squeeze(y_pred_list, axis=-1)
        
    roc_list = []
    invalid_count = 0
    for i in range(y_true_list.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true_list[:, i] == 1) > 0 and np.sum(y_true_list[:, i] == -1) > 0:
            is_valid = y_true_list[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true_list[is_valid, i] + 1) / 2, y_pred_list[is_valid, i]))
        else:
            invalid_count += 1

    print('Invalid task count:\t', invalid_count)

    if len(roc_list) < y_true_list.shape[1]:
        print('Some target is missing!')
        print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true_list.shape[1]))

    roc_list = np.array(roc_list)
    roc_value = np.mean(roc_list)
    return roc_value


LOSS_FUNCTION_MATCH_DICT = {
    'BACE': nn.BCEWithLogitsLoss(),
    'BBBP': nn.BCEWithLogitsLoss(),
    'HIV':  nn.BCEWithLogitsLoss(),
    'ClinTox': nn.BCEWithLogitsLoss(),
    'Sider': nn.BCEWithLogitsLoss(),
    'Tox21': nn.BCEWithLogitsLoss(),
    'ToxCast': nn.BCEWithLogitsLoss(),
    'QM9': nn.MSELoss(),
    'QM8': nn.MSELoss(),
    'QM7': nn.L1Loss(),
    'ESOL': nn.MSELoss(),
    'Lipo': nn.MSELoss(),
    'FreeSolv': nn.MSELoss()
}

EVALUE_FUNCTION_MATCH_DICT = {
    'BACE': roc_auc_function,
    'BBBP': roc_auc_function,
    'HIV':  roc_auc_function,
    'ClinTox': roc_auc_function,
    'Sider': roc_auc_function,
    'Tox21': roc_auc_function,
    'ToxCast': roc_auc_function,
    'QM9': nn.L1Loss(),
    'QM8': nn.L1Loss(),
    'QM7': nn.L1Loss(),
    'ESOL': rmse,
    'Lipo': rmse,
    'FreeSolv': rmse
}
