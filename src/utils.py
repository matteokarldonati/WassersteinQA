import pandas as pd
import torch
from torch import nn


class SinkhornDistance(nn.Module):
    """
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on reduction
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        C = self.cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        actual_nits = 0

        thresh = 1e-1

        for i in range(self.max_iter):

            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1

            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        """
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) epsilon$
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    def cost_matrix(self, x, y, p=2):
        """Returns the matrix of $|x_i-y_j|^p$."""

        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)

        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """Barycenter subroutine, used by kinetic acceleration through extrapolation."""
        return tau * u + (1 - tau) * u1


def get_distance_matrix(embeddings, dataset_names):
    wasserstein = SinkhornDistance(eps=1, max_iter=100)

    distance_matrix = pd.DataFrame(columns=dataset_names, index=dataset_names)

    for i in embeddings:
        for j in embeddings:
            distance_matrix.loc[i, j] = round(
                wasserstein(torch.tensor(embeddings[i]), torch.tensor(embeddings[j])).item(), 2)
            if i == j:
                distance_matrix.loc[i, j] = 0.0

    distance_matrix = distance_matrix.astype(float)

    return distance_matrix
