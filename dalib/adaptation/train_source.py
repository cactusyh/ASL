import torch
import torch.nn as nn
import torch.nn.functional as F


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


# def cross_entropy_ls(pred, label, alpha=0.1):
#     ce_loss = F.cross_entropy(pred, label)
#     kl_loss = - torch.mean(F.log_softmax(pred, dim=1))
#     return (1 - alpha) * ce_loss + alpha * kl_loss


def cross_entropy_ls(pred, label, alpha=0.1):
    log_probs = F.log_softmax(pred, dim=1)
    targets = torch.zeros(log_probs.size()).to(pred.device)
    targets = targets.scatter_(1, label.unsqueeze(1), 1)
    targets = (1 - alpha) * targets + alpha / pred.size(1)
    loss = (- targets * log_probs).sum(dim=1).mean()
    return loss


class DatasetTransform(torch.utils.data.Dataset):
    def __init__(self, dt, transform):
        self.dt = dt
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.dt[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.dt)
