from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class LabelOptimizer():

    def __init__(self, N, K, lambd, device):
        self.N = N
        self.K = K
        self.lambd = lambd
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Q = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N, K).to(device)
        self.r = 1.
        self.c = 1. * N / K

    def update_P(self, p_t, index):
        # p_batch = p_t / self.N
        self.P[index, :] = p_t

    def update_Q(self):
        # solve label assignment via sinkhorn-knopp
        self.P = self.P ** self.lambd
        v = (torch.ones(self.K, 1) / self.K).to(self.device)
        err = 1.
        cnt = 0
        while err > 0.1:
            u = self.r / (self.P @ v)
            new_v = self.c / (self.P.T @ u)
            err = torch.sum(torch.abs(new_v / v - 1))
            v = new_v
            cnt += 1
        # print(f'error: {err}, step: {cnt}')
        self.Q = u * self.P * v.squeeze()
        # Q = torch.diag(u.squeeze()) @ self.P @ torch.diag(v.squeeze())
        
    def update_Labels(self):
        self.Labels = self.Q.argmax(dim=1)
