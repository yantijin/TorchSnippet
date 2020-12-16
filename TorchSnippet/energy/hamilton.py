import torch
import torch.nn as nn
import numpy as np

__all__ = ['HNN', 'nonsep_symint']


class HNN(nn.Module):
    """Hamiltonian Neural ODE

    :param net: function parametrizing the vector field.
    :type net: nn.Module
    """
    def __init__(self, net:nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            n = x.shape[1] // 2
            gradH = torch.autograd.grad(self.net(x).sum(), x,
                                        create_graph=True)[0]
        return torch.cat([gradH[:, n:], -gradH[:, :n]], 1).to(x)


def nonsep_symint(q, p, x, y, dt, K_t, eps, w=20, last=True):
    '''
    :param q: inputs
    :param p: inputs
    :param x: inputs
    :param y: inputs
    :param dt: total time for integration
    :param K_t: hamilton function
    :param eps: time step
    :param w: parameters for R^{\delta}
    :param last: whether output total integration or only the last one
    :return: integration results
    '''
    n_steps = np.round((torch.abs(dt) / eps).max().item())
    h = dt / n_steps
    h = h.unsqueeze(1).unsqueeze(1)
    res = []
    for i_step in range(int(n_steps)):
        x1, p1 = K_t(q, y)
        p = p - x1 * h * 0.5
        x = x + p1 * h * 0.5
        q1, y1 = K_t(x, p)
        q = q + y1 * h * 0.5
        y = y - q1 * h * 0.5
        q1 = 0.5 * (q - x)
        p1 = 0.5 * (p - y)
        x1 = torch.cos(2 * w * h) * q1 + torch.sin(2 * w * h) * p1
        y1 = -torch.sin(2 * w * h) * q1 + torch.cos(2 * w * h) * p1
        q1 = 0.5 * (q + x)
        p1 = 0.5 * (p + y)
        q = q1 + x1
        p = p1 + y1
        x = q1 - x1
        y = p1 - y1
        q1, y1 = K_t(x, p)
        q = q + y1 * h * 0.5
        y = y - q1 * h * 0.5
        x1, p1 = K_t(q, y)
        p = p - x1 * h * 0.5
        x = x + p1 * h * 0.5
        res.append(torch.cat([q, p, x, y], dim=1))
    if last:
        return [q, p, x, y]
    else:
        res = torch.stack(res)
        return res
