import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def generate_binomial_mask(target_matrix, p=0.5):
    # np.random.seed(seed_)
    return torch.from_numpy(np.random.binomial(1, p, size=(target_matrix.shape)))


class Linear_probe(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.linear = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        return self.linear(x)

class Conv_Pyram_model(nn.Module):
    def __init__(self, input_dims, output_dims, dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernel_size = 7
        self.dropout = dropout
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(self.input_dims, 32, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding=(3)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)

        ).cuda()

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, self.output_dims, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(self.output_dims),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()


    def forward(self, x, p=1, seed_=42):
        x = x
        x = x.transpose(1, 2)
        x1 = self.conv_block1_t(x)
        if p < 1:
            mask_ = generate_binomial_mask(torch.ones(x1.shape), p=p).cuda()
            x1 = x1 * mask_
        x2 = self.conv_block2_t(x1)
        if p < 1:
            mask_2 = generate_binomial_mask(torch.ones(x2.shape), p=p).cuda()
            x2 = x2 * mask_2
        x3 = self.conv_block3_t(x2)
        if p < 1:
            mask_3 = generate_binomial_mask(torch.ones(x3.shape), p=p).cuda()
            x3 = x3 * mask_3
        z = nn.functional.max_pool1d(x3, x3.shape[2])
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        return x1, x2, x3, z_norm

class Conv_Pyram_model_EDF(nn.Module):
    def __init__(self, input_dims, output_dims, dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernel_size = 7
        self.dropout = dropout
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(self.input_dims, 32, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding=(3)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)

        ).cuda()

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, self.output_dims, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(self.output_dims),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()


    def forward(self, x, p=1, seed_=42):
        x = x
        x = x.transpose(1, 2)
        x1 = self.conv_block1_t(x)
        if p < 1:
            mask_ = generate_binomial_mask(torch.ones(x1.shape), p=p).cuda()
            x1 = x1 * mask_
        x2 = self.conv_block2_t(x1)
        if p < 1:
            mask_2 = generate_binomial_mask(torch.ones(x2.shape), p=p).cuda()
            x2 = x2 * mask_2
        x3 = self.conv_block3_t(x2)
        if p < 1:
            mask_3 = generate_binomial_mask(torch.ones(x3.shape), p=p).cuda()
            x3 = x3 * mask_3
        z = nn.functional.max_pool1d(x3, x3.shape[2])
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        return x1, x2, x3, z_norm


class Conv_Pyram_model_HAR(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial', dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernel_size = 7
        self.dropout = dropout
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(self.input_dims, 64, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding=(3)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
            nn.Dropout(self.dropout)

        ).cuda()

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(128, self.output_dims, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(self.output_dims),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()


    def forward(self, x, p=1, seed_=42):

        x = x
        x = x.transpose(1, 2)
        x1 = self.conv_block1_t(x)
        if p < 1:
            mask_ = generate_binomial_mask(torch.ones(x1.shape), p=p).cuda()
            x1 = x1 * mask_
        x2 = self.conv_block2_t(x1)
        if p < 1:
            mask_2 = generate_binomial_mask(torch.ones(x2.shape), p=p).cuda()
            x2 = x2 * mask_2
        x3 = self.conv_block3_t(x2)
        if p < 1:
            mask_3 = generate_binomial_mask(torch.ones(x3.shape), p=p).cuda()
            x3 = x3 * mask_3
        z = nn.functional.max_pool1d(x3, x3.shape[2])
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        return x1, x2, x3, z_norm

class Conv_Pyram_model_Epi(nn.Module):
    def __init__(self, input_dims, output_dims, dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernel_size = 7
        self.dropout = dropout
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(self.input_dims, 32, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding=(3)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)

        ).cuda()

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, self.output_dims, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(self.output_dims),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()


    def forward(self, x, p=1, seed_=42):
        x = x
        x = x.transpose(1, 2)
        x1 = self.conv_block1_t(x)
        if p < 1:
            mask_ = generate_binomial_mask(torch.ones(x1.shape), p=p).cuda()
            x1 = x1 * mask_
        x2 = self.conv_block2_t(x1)
        if p < 1:
            mask_2 = generate_binomial_mask(torch.ones(x2.shape), p=p).cuda()
            x2 = x2 * mask_2
        x3 = self.conv_block3_t(x2)
        if p < 1:
            mask_3 = generate_binomial_mask(torch.ones(x3.shape), p=p).cuda()
            x3 = x3 * mask_3
        z = nn.functional.max_pool1d(x3, x3.shape[2])
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        return x1, x2, x3, z_norm

class Conv_Pyram_model_Waveform(nn.Module):
    def __init__(self, input_dims, output_dims, dropout=0.0):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernel_size = 7
        self.dropout = dropout
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(self.input_dims, 32, kernel_size=self.kernel_size,
                      stride=1, bias=False, padding=(3)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(self.dropout)

        ).cuda()

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, self.output_dims, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(self.output_dims),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(self.dropout)
        ).cuda()


    def forward(self, x, p=1, seed_=42):
        x = x
        x = x.transpose(1, 2)
        x1 = self.conv_block1_t(x)
        if p < 1:
            mask_ = generate_binomial_mask(torch.ones(x1.shape), p=p).cuda()
            x1 = x1 * mask_
        x2 = self.conv_block2_t(x1)
        if p < 1:
            mask_2 = generate_binomial_mask(torch.ones(x2.shape), p=p).cuda()
            x2 = x2 * mask_2
        x3 = self.conv_block3_t(x2)
        if p < 1:
            mask_3 = generate_binomial_mask(torch.ones(x3.shape), p=p).cuda()
            x3 = x3 * mask_3
        z = nn.functional.max_pool1d(x3, x3.shape[2])
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        return x1, x2, x3, z_norm


