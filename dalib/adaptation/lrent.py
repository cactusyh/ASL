from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DomainAdversarialLoss']


class ImageClassifier(nn.Module):

    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self._features_dim = bottleneck_dim
        self.head = nn.Linear(self._features_dim, num_classes)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        x = self.backbone(x)
        f = self.bottleneck(x)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params


class DataSetIdx(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


class LabelOptimizer():

    def __init__(self, N, K, lambd, device):
        self.N = N
        self.K = K
        self.lambd = lambd
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N).to(device)
        self.r = 1.
        self.c = 1. * N / K

    def update_P(self, p_t, index):
        # p_batch = p_t / self.N
        self.P[index, :] = p_t

    def update_Labels(self):
        self.P = self.P ** (1 / self.lambd)
        self.Labels = self.P / self.P.sum(dim=1, keepdim=True)


def cross_entropy_soft(y_t, soft_label):
    return - (soft_label * F.log_softmax(y_t, dim=1)).sum() / y_t.size(0)