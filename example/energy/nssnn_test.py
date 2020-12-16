import os
import torch
import torch.nn as nn
import math
import torch.utils.data as data
import pytorch_lightning as pl
import TorchSnippet as tsp
from TorchSnippet.energy import nonsep_symint
import matplotlib.pyplot as plt
import h5py

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
plot_points = 20000
plot_Np = 2000000

##############################################
# generate data
##############################################
def analyH(q, p):
    return 0.5 * (q ** 2 + 1) * (p ** 2 + 1)


class KAnalysis(nn.Module): # 生成数据的解析形式
    def __init__(self, ):
        super(KAnalysis, self).__init__()

    def forward(self, q, p):
        with torch.enable_grad():
            q = q.requires_grad_(True)
            p = p.requires_grad_(True)
            K = analyH(q, p)
            K = K.sum()
            dq = torch.autograd.grad(K, q, retain_graph=True, create_graph=False)[0]
            dp = torch.autograd.grad(K, p, retain_graph=True, create_graph=False)[0]
        return dq, dp


def Gen_Data(d_t, eps, w=20, n_sample=1280, N=1, n_steps=1):
    datau = [[] for i in range(n_steps+1)]
    for i in range(n_sample):
        q0 = 8. * (torch.rand(1, 1, N) - 0.5)
        p0 = 8. * (torch.rand(1, 1, N) - 0.5)
        x0 = q0
        y0 = p0
        datau[0].append(torch.cat([q0, p0, x0, y0], dim=1).unsqueeze(-1))
        f_true = KAnalysis()
        f_true.eval()
        with torch.no_grad():
            for j in range(n_steps):
                [q0, p0, x0, y0] = nonsep_symint(q0, p0, x0, y0, d_t, f_true, eps=eps, w=w)
                datau[j+1].append(torch.cat([q0, p0, x0, y0], dim=1).unsqueeze(-1))
    datau = [torch.cat(datau[k]) for k in range(n_steps+1)]
    datau = torch.cat(datau, dim=-1).float()

    qT = torch.tensor([[[0.]]])
    pT = torch.tensor([[[-3.]]])
    xT = qT
    yT = pT
    qpxyT = [torch.cat([qT, pT, xT, yT], dim=1)]
    print('test data')
    with torch.no_grad():
        for i in range(plot_Np):
            qT, pT, xT, yT = nonsep_symint(qT, pT, xT, yT, torch.tensor([plot_points / plot_Np]).to(device), f_true.forward, 0.02)
            qpxyT.append(torch.cat([qT, pT, xT, yT], dim=1))
    qpxyT = torch.cat(qpxyT)
    return datau, qpxyT

datau, qpxyT = Gen_Data(d_t=torch.tensor([0.02]), eps=0.02)
train = data.TensorDataset(datau)
trainloader = data.DataLoader(train, batch_size=512, shuffle=True)


##############################################
# prepare for training
##############################################

class basis_learner(pl.LightningModule):
    def __init__(self, model, d_t, n_steps=1, eps=0.02, w=20):
        super(basis_learner, self).__init__()
        self.model = model
        self.d_t = d_t
        self.eps = eps
        self.n_steps = n_steps
        self.w = w

    def training_step(self, batch, batch_idx):
        res = []
        batch, = batch
        for j in range(self.n_steps):
            q0, p0, x0, y0 = batch[:,0:1, :, j],batch[:,1:2, :, j],batch[:,2:3, :, j],batch[:,3:4, :, j]
            y_hat = nonsep_symint(q0, p0, x0, y0, self.d_t, self.model.forward_train, self.eps, self.w)
            res.append(torch.cat(y_hat, dim=1).unsqueeze(-1))
        y_hat = torch.cat(res, dim=-1)
        loss = nn.L1Loss()(batch[..., 1:], y_hat)
        logs = {'loss': loss}
        return {'loss': loss, 'logs': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.005)

    def train_dataloader(self):
        return trainloader


class LinearBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(LinearBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Linear(inchannel, outchannel),
            # nn.Tanh(),
            nn.Sigmoid(),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        return out


class KTrained(nn.Module): # 先计算H，而后计算 \dot p, \dot q
    def __init__(self, N, hidden_dim):
        super(KTrained, self).__init__()
        self.N = N
        self.cal_H = nn.Sequential(LinearBlock(2 * self.N, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   nn.Linear(hidden_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-math.sqrt(6. / m.in_features), math.sqrt(6. / m.in_features))

    def forward_train(self, q, p):
        with torch.enable_grad():
            x = torch.cat([q, p], dim=2)
            x = x.requires_grad_(True)
            K = self.cal_H(x.squeeze(1))
            dK = torch.autograd.grad(K.sum(), x, retain_graph=True, create_graph=True)[0]
        return dK[:, :, :self.N], dK[:, :, self.N:self.N * 2]

    def forward(self, q, p):
        with torch.enable_grad():
            x = torch.cat([q, p], dim=2)
            x = x.requires_grad_(True)
            K = self.cal_H(x.squeeze(1))
            dK = torch.autograd.grad(K.sum(), x, retain_graph=True, create_graph=False)[0]
        return dK[:, :, :self.N], dK[:, :, self.N:self.N * 2]

class nonseparate_hnn(nn.Module):
    def __init__(self, dt, eps, net, w=20, last=True):
        super(nonseparate_hnn, self).__init__()
        self.dt = dt
        self.eps = eps
        self.net = net
        self.w = w
        self.last = last

    def forward(self, q, p, x, y):
        res = nonsep_symint(q, p, x, y, self.dt, self.net.forward_train, eps=self.eps, w=self.w, last=self.last)
        return res

func = KTrained(N=1, hidden_dim=64)
learn = basis_learner(func, d_t=torch.tensor([0.02]))
trainer = pl.Trainer(min_epochs=800, max_epochs=1000)
trainer.fit(learn)

# plot figures
qN = qpxyT[0:1, 0:1, :].to(device)
pN = qpxyT[0:1, 1:2, :].to(device)
xN = qpxyT[0:1, 2:3, :].to(device)
yN = qpxyT[0:1, 3:4, :].to(device)
qpxyN = [torch.cat([qN, pN, xN, yN], dim=1).detach().cpu()]

for i in range(20000):
    qN, pN, xN, yN = nonsep_symint(qN, pN, xN, yN, torch.tensor([2000 / 20000]).to(device), func.forward, 0.02)
    qpxyN.append(torch.cat([qN, pN, xN, yN], dim=1).detach().cpu())
qpxyN = torch.cat(qpxyN)
qpxyN = qpxyN.detach().cpu().numpy()
qpxyT = qpxyT.detach().cpu().numpy()
plt.clf()
plt.plot(qpxyN[:, 0, 0], qpxyN[:, 1, 0], c="r", linestyle='--')
plt.plot(qpxyT[:, 0, 0], qpxyT[:, 1, 0], c="b")
# plt.scatter(qpxyT[:, 0, 0], qpxyT[:, 1, 0], s=40, c="b")
# plt.draw()
plt.show()
# plt.pause(1.)





# n_steps = 1
# d_t = torch.tensor([0.01])
# optim = torch.optim.Adam(func.parameters(), lr=0.001)
# for i in range(1001):
#     for batch in trainloader:
#         optim.zero_grad()
#         res = []
#         batch, = batch
#         for j in range(n_steps):
#             q0, p0, x0, y0 = batch[:, 0:1, :, j], batch[:, 1:2, :, j], batch[:, 2:3, :, j], batch[:, 3:4, :, j]
#             y_hat = nonsep_symint(q0, p0, x0, y0, d_t, func.forward_train, eps=0.01, w=20)
#             res.append(torch.cat(y_hat, dim=1).unsqueeze(-1))
#         y_hat = torch.cat(res, dim=-1)
#         loss = nn.L1Loss()(batch[..., 1:], y_hat)
#         loss.backward(retain_graph=True)
#         optim.step()
#         print('loss:', loss.detach().cpu().numpy())




# if __name__ == "__main__":
#     res=Gen_Data(d_t=torch.tensor([0.01]), eps=0.01)
#     print(res.shape)