from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dalib.modules.grl import WarmStartGradientReverseLayer, GradientReverseLayer
from dalib.modules.domain_discriminator import DomainDiscriminator
from ._util import binary_accuracy

__all__ = ['DomainAdversarialLoss']


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator: nn.Module, no_warm_start: Optional[bool] = False, 
                 reduction: Optional[str] = 'mean'):
        super(DomainAdversarialLoss, self).__init__()
        if no_warm_start:
            self.grl = GradientReverseLayer()
        else:
            self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.domain_discriminator = domain_discriminator
        self.bce = nn.BCELoss(reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        return 0.5 * (self.bce(d_s, d_label_s) + self.bce(d_t, d_label_t))


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

    def __init__(self, N, K, beta, device):
        self.N = N
        self.K = K
        self.beta = beta
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N).to(device)
        self.r = 1.
        self.c = 1. * N / K

    def update_P(self, p_s, p_t, index):
        p_batch = p_t * (p_s ** self.beta)
        self.P[index, :] = p_batch

    def update_Labels(self):
        # solve label assignment via sinkhorn-knopp
        self.P = self.P ** (1. / self.beta)
        v = (torch.ones(self.K, 1) / self.K).to(self.device)
        err = 1.
        cnt = 0
        while err > 0.01:
            u = self.r / (self.P @ v)
            new_v = self.c / (self.P.T @ u)
            err = torch.sum(torch.abs(new_v / v - 1))
            v = new_v
            cnt += 1
        print(f'error: {err}, step: {cnt}')
        Q = u * self.P * v.squeeze()
        # Q = torch.diag(u.squeeze()) @ self.P @ torch.diag(v.squeeze())
        self.Labels = Q.argmax(dim=1)
