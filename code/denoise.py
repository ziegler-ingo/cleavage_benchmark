import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import linear_rampup


class NoiseAdaptation(nn.Module):
    """
    Based on:

    Goldberger, J., & Ben-Reuven, E. (2017, April).
    Training deep neural-networks using a noise adaptation layer.
    In International conference on learning representations.

    Unofficial Implementation:
    https://github.com/Billy1900/Noise-Adaption-Layer
    """

    def __init__(self, theta, k, device):
        super().__init__()
        self.theta = nn.Linear(k, k, bias=False)
        self.theta.weight.data = theta
        self.eye = torch.eye(k).to(device)

    def forward(self, x):
        theta = self.theta(self.eye)
        theta = F.softmax(theta, dim=0)
        out = torch.matmul(x, theta)
        return out


class CoteachingLoss:
    """
    Based on:

    Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., ... & Sugiyama, M. (2018).
    Co-teaching: Robust training of deep neural networks with extremely noisy labels.
    Advances in neural information processing systems, 31.
    """

    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, y1, y2, t, forget_rate):
        l1 = self.criterion(y1, t)
        idx1 = torch.argsort(l1)

        l2 = self.criterion(y2, t)
        idx2 = torch.argsort(l2)

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * l1.shape[0])

        idx1_update = idx1[:num_remember]
        idx2_update = idx2[:num_remember]

        # exchange the samples
        l1_update = self.criterion(y1[idx2_update], t[idx2_update])
        l2_update = self.criterion(y2[idx1_update], t[idx1_update])

        return l1_update.sum() / num_remember, l2_update.sum() / num_remember


class JoCoRLoss:
    """
    Based on:

    Wei, H., Feng, L., Chen, X., & An, B. (2020).
    Combating noisy labels by agreement: A joint training method with co-regularization.
    In Proceedings of the IEEE/CVF conference on
    computer vision and pattern recognition (pp. 13726-13735).
    """

    def __init__(self):
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.co_lambda = 0.1

    def kl_loss(self, pred, soft_target):
        return F.kl_div(F.logsigmoid(pred), torch.sigmoid(soft_target), reduction="sum")

    def __call__(self, y1, y2, lbls, forget_rate):
        l1 = self.criterion(y1, lbls) * (1 - self.co_lambda)
        l2 = self.criterion(y2, lbls) * (1 - self.co_lambda)
        losses = (
            l1
            + l2
            + (self.co_lambda * self.kl_loss(y1, y2))
            + (self.co_lambda * self.kl_loss(y2, y1))
        )

        idx = torch.argsort(losses)
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * losses.shape[0])

        idx_update = idx[:num_remember]
        loss = losses[idx_update].mean()
        return loss


class SemiLoss:
    def __call__(
        self, out_x, lbl_x, out_u, lbl_u, lambda_u, epoch, warm_up, rampup_len
    ):
        probs_u = F.softmax(out_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(out_x, dim=1) * lbl_x, dim=1))
        Lu = torch.mean((probs_u - lbl_u) ** 2)
        return Lx, Lu, linear_rampup(lambda_u, epoch, warm_up, rampup_len)


class NegEntropy:
    def __call__(self, out):
        probs = F.softmax(out, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))
