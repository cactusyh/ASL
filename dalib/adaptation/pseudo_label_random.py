from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
from torch._C import dtype
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

    def __init__(self, N, K, lambd, device, threshold=0.5):
        self.N = N
        self.K = K
        self.lambd = lambd
        self.device = device
        self.P = torch.zeros(N, K).to(device)
        self.Labels = torch.zeros(N, dtype=torch.long).to(device)
        self.r = 1.
        self.c = 1. * N / K
        self.threshold = threshold

    def update_P(self, p_t, index):
        # p_batch = p_t / self.N
        self.P[index, :] = p_t

    def update_Labels(self):
        for i in range(self.P.size(0)):
            if self.P[i].max() > self.threshold:
                self.Labels[i] = self.P[i].argmax(dim=0)
            else:
                self.Labels[i] = torch.randint(low=0, high=self.K, size=(1,))


def entropy_loss(p_t):
    return - (p_t * torch.log(p_t)).sum() / p_t.size(0)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VirtualAdversarialLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VirtualAdversarialLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.randn(x.shape).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat, _ = model(x + self.xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            model.zero_grad()
    
        # calc VAT loss
        r_adv = d * self.eps
        pred_hat, _ = model(x + r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        loss = F.kl_div(logp_hat, pred, reduction='batchmean')

        return loss
