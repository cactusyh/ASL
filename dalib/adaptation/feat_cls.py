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
        f1 = self.backbone(x)
        # f = f.view(-1, self.backbone.out_features)
        f2 = self.bottleneck(f1)
        predictions = self.head(f2)
        return predictions, f1, f2

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


class DomainClassifier(nn.Module):

    def __init__(self, in_feature: int, bottleneck_dim: Optional[int] = 256):
        super(DomainClassifier, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_feature, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.domain_discriminator = DomainDiscriminator(in_feature=bottleneck_dim, hidden_size=1024)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.bottleneck(x)
        predictions = self.domain_discriminator(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
        ] + self.domain_discriminator.get_parameters()
        return params


class DomainClsLoss(nn.Module):

    def __init__(self):
        super(DomainClsLoss, self).__init__()
        self.iter_num = 0
        self.max_iters = 1000
        self.domain_cls_acc = 0

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((d_s.size(0), 1)).to(d_s.device)
        d_label_t = torch.zeros((d_t.size(0), 1)).to(d_t.device)
        self.domain_cls_acc = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        coeff = np.float(2.0 / (1.0 + np.exp(-self.iter_num / self.max_iters)) - 1)
        loss = coeff * 0.5 * (F.binary_cross_entropy(d_s, d_label_s) + F.binary_cross_entropy(d_t, d_label_t))
        self.step()
        return loss
    
    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class FeatureClsLoss(nn.Module):

    def __init__(self):
        super(FeatureClsLoss, self).__init__()
        self.iter_num = 0
        self.max_iters = 1000
        self.feature_cls_acc = 0

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f_c, f_d = f.chunk(2, dim=0)
        f_label_c = torch.ones((f_c.size(0), 1)).to(f_c.device)
        f_label_d = torch.zeros((f_d.size(0), 1)).to(f_d.device)
        self.feature_cls_acc = 0.5 * (binary_accuracy(f_c, f_label_c) + binary_accuracy(f_d, f_label_d))
        coeff = np.float(2.0 / (1.0 + np.exp(-self.iter_num / self.max_iters)) - 1)
        loss = coeff * 0.5 * (F.binary_cross_entropy(f_c, f_label_c) + F.binary_cross_entropy(f_d, f_label_d))
        self.step()
        return loss

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
