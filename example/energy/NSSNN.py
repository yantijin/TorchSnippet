# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as func
import numpy as np
import math
import h5py
import os
import itertools

# from torch.utils.tensorboard import SummaryWriter

epsT = 0.01
epsN = 0.01
epsD = 0.01
nsteps = 1
N = 1
device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
points_validation = 200
dtp_validation = torch.tensor([0.01])
plot_points = 10
lt = 0.1
dtp = torch.tensor([lt / plot_points])
hidden_dim = 64
n_sample = 1280
lt_Np = 100
plot_Np = 1000
dt_Np = torch.tensor([lt_Np / plot_Np])
l_r = 5 * epsN


def to_np(x):
    return x.detach().cpu().numpy()


def analyH(q, p):
    return 0.5 * (q ** 2 + 1) * (p ** 2 + 1)


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


class KTrained_baseline(nn.Module): # 直接从p,q到\dot p, \dot q
    def __init__(self, N, hidden_dim):
        super(KTrained_baseline, self).__init__()
        self.N = N
        self.cal_H = nn.Sequential(LinearBlock(2 * self.N, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   LinearBlock(hidden_dim, hidden_dim),
                                   nn.Linear(hidden_dim, 2 * self.N))
        self.b = nn.Parameter(torch.zeros(1, 1, 2 * self.N), requires_grad=True).to(device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-math.sqrt(6. / m.in_features), math.sqrt(6. / m.in_features))

    def forward(self, q, p):
        with torch.enable_grad():
            x = torch.cat([q, p], dim=2)
            x = x.requires_grad_(True)
            K = self.cal_H(x) + self.b
        return K[:, :, :self.N], K[:, :, self.N:self.N * 2]


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


def RK2(q, p, dt, K_t, eps):
    n_steps = np.round((torch.abs(dt) / eps).max().item())
    h = dt / n_steps
    h = h.unsqueeze(1).unsqueeze(1)
    for i_step in range(int(n_steps)):
        dp, dq = K_t(q, p)
        q1 = q + 0.5 * dq * h
        p1 = p - 0.5 * dp * h
        dp, dq = K_t(q1, p1)
        q = q + dq * h
        p = p - dp * h
    return q, p


def Nonsep_SymInt(q, p, x, y, dt, K_t, eps): # 根据alg 1 生成p, q, x, y
    n_steps = np.round((torch.abs(dt) / eps).max().item())
    h = dt / n_steps
    h = h.unsqueeze(1).unsqueeze(1)
    w = 20
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
    return q, p, x, y


def Gen_Data():
    datau = [[] for i in range(nsteps + 1)]
    datau_hnn = [[] for i in range(2)]
    datat = []
    for i in range(n_sample):
        t1_t2 = torch.tensor([epsD])
        q0 = 8. * (torch.rand(1, 1, N) - 0.5)
        p0 = 8. * (torch.rand(1, 1, N) - 0.5)
        datau_hnn[0].append(torch.cat([q0, p0], dim=1).unsqueeze(-1))
        x0 = q0
        y0 = p0
        datau[0].append(torch.cat([q0, p0, q0, p0], dim=1).unsqueeze(-1))
        f_true = KAnalysis()
        f_true.eval()
        with torch.no_grad():
            q1, p1, x1, y1 = Nonsep_SymInt(q0, p0, x0, y0, t1_t2, f_true.forward, epsT)
            q1 = q1 + 0.0 * (torch.rand(1, 1, N) - 0.5)
            p1 = p1 + 0.0 * (torch.rand(1, 1, N) - 0.5)
            dq0 = (q1 - q0) / t1_t2
            dp0 = (p1 - p0) / t1_t2
            datau_hnn[1].append(torch.cat([-dp0, dq0], dim=1).unsqueeze(-1))
            q0 = q1
            p0 = p1
            x0 = q1
            y0 = p1
            datau[1].append(torch.cat([q1, p1, q1, p1], dim=1).unsqueeze(-1))
            for j in range(1, nsteps):
                q1, p1, x1, y1 = Nonsep_SymInt(q0, p0, x0, y0, t1_t2, f_true.forward, epsT)
                q0 = q1
                p0 = p1
                x0 = q1
                y0 = p1
                # print(q1-x1,p1-y1)
                # q1 = q1 + 0.5*(torch.rand(1, 1, N)-0.5)
                # p1 = p1 + 0.5*(torch.rand(1, 1, N)-0.5)
                # q1 = q1 + 0.01*(torch.rand(1, 1, N)-0.5)
                # p1 = p1 + 0.01*(torch.rand(1, 1, N)-0.5)
                datau[j + 1].append(torch.cat([q1, p1, q1, p1], dim=1).unsqueeze(-1))
            datat.append(t1_t2)
    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    datau = [torch.cat(datau[j]) for j in range(nsteps + 1)]
    datau = torch.cat(datau, dim=-1).float()
    datat = torch.tensor(datat).float()
    hf = h5py.File(os.path.join(data_root, "data.h5"), "w")
    hf.create_dataset('u', data=datau)
    hf.create_dataset('dt', data=datat)
    hf.close()

    data_root_hnn = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    datau_hnn = [torch.cat(datau_hnn[j]) for j in range(2)]
    datau_hnn = torch.cat(datau_hnn, dim=-1).float()
    hf_hnn = h5py.File(os.path.join(data_root_hnn, "data_hnn.h5"), "w")
    hf_hnn.create_dataset('u', data=datau_hnn)
    hf_hnn.close()

    qT = torch.tensor([[[0.]]])
    pT = torch.tensor([[[-3.]]])
    xT = qT
    yT = pT
    qpxyT = [torch.cat([qT, pT, xT, yT], dim=1)]
    print('test data')
    with torch.no_grad():
        for i in range(plot_points):
            qT, pT, xT, yT = Nonsep_SymInt(qT, pT, xT, yT, dtp, f_true.forward, epsT)
            qpxyT.append(torch.cat([qT, pT, xT, yT], dim=1))
    torch.save(torch.cat(qpxyT), 'test.dat')
    qpxyT0 = []
    qpxyT1 = []
    print('validation data')
    with torch.no_grad():
        for i in range(points_validation):
            qT0 = 8. * (torch.rand(1, 1, N) - 0.5)
            pT0 = 8. * (torch.rand(1, 1, N) - 0.5)
            xT0 = qT0
            yT0 = pT0
            qT1, pT1, xT1, yT1 = Nonsep_SymInt(qT0, pT0, xT0, yT0, dtp_validation, f_true.forward, epsT)
            # print(qT1 - xT1, pT1 - yT1)
            # print(analyH(qT1, pT1)-analyH(qT0, pT0))
            qpxyT0.append(torch.cat([qT0, pT0, xT0, yT0], dim=1))
            qpxyT1.append(torch.cat([qT1, pT1, xT1, yT1], dim=1))
    torch.save(torch.cat(qpxyT0), 'validation0.dat')
    torch.save(torch.cat(qpxyT1), 'validation1.dat')


def Gen_Data_HNN():
    datau = [[] for i in range(2)]
    f_true = KAnalysis()
    f_true.eval()
    for i in range(n_sample):
        q0 = 6. * (torch.rand(1, 1, N) - 0.5)
        p0 = 6. * (torch.rand(1, 1, N) - 0.5)
        datau[0].append(torch.cat([q0, p0], dim=1).unsqueeze(-1))
        dq0, dp0 = f_true.forward(q0, p0)
        datau[1].append(torch.cat([dq0, dp0], dim=1).unsqueeze(-1))
    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    datau = [torch.cat(datau[j]) for j in range(2)]
    datau = torch.cat(datau, dim=-1).float()
    hf = h5py.File(os.path.join(data_root, "data_hnn.h5"), "w")
    hf.create_dataset('u', data=datau)
    hf.close()


class Dataset_HNN(torch.utils.data.Dataset):
    def __init__(self, data_type):
        f = h5py.File('data_hnn.h5')
        self.u = f['u'][:]
        split = int(self.u.shape[0] * 0.9)
        if data_type == 'train':
            self.u = torch.from_numpy(self.u[:split]).to(device)
        else:
            self.u = torch.from_numpy(self.u[split:]).to(device)
        f.close()

    def __getitem__(self, index):
        return self.u[index]

    def __len__(self):
        return self.u.shape[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        f = h5py.File('data.h5')
        self.u = f['u'][:]
        self.dt = f['dt'][:]
        split = int(self.u.shape[0] * 0.9)
        if data_type == 'train':
            self.u = torch.from_numpy(self.u[:split]).to(device)
            self.dt = torch.from_numpy(self.dt[:split]).to(device)
        else:
            self.u = torch.from_numpy(self.u[split:]).to(device)
            self.dt = torch.from_numpy(self.dt[split:]).to(device)
        f.close()

    def __getitem__(self, index):
        return self.u[index], self.dt[index]

    def __len__(self):
        return self.u.shape[0]


f_neur_hnn = KTrained(N, hidden_dim)
f_neur_hnn.to(device)
f_neur_nssnn = KTrained(N, hidden_dim)
f_neur_nssnn.to(device)
f_neur_baseline = KTrained_baseline(N, hidden_dim)
f_neur_baseline.to(device)
loss_func = func.l1_loss


# loss_func = func.mse_loss


def train(model):
    # writer = SummaryWriter()

    if (model == "model_baseline.pt"):
        optimizer = torch.optim.Adam(f_neur_baseline.parameters(), lr=l_r)
    elif (model == "model_hnn.pt"):
        optimizer = torch.optim.Adam(f_neur_hnn.parameters(), lr=l_r)
    elif (model == "model_nssnn.pt"):
        optimizer = torch.optim.Adam(f_neur_nssnn.parameters(), lr=l_r)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    if (model == "model_hnn.pt"):
        train_data_loader = torch.utils.data.DataLoader(Dataset_HNN('train'), batch_size=512, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(Dataset_HNN('test'), batch_size=512, shuffle=True)
    else:
        train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=512, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(Dataset('test'), batch_size=512, shuffle=True)
    lowest_test_loss = 99999
    loss_data = []
    for i in range(1001):
        train_loss = 0
        train_sample = 0
        test_loss = 0
        test_sample = 0
        if (model == "model_baseline.pt"):
            f_neur_baseline.train()
        elif (model == "model_hnn.pt"):
            f_neur_hnn.train()
        elif (model == "model_nssnn.pt"):
            f_neur_nssnn.train()
        for batch_index, data_batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            if (model == "model_hnn.pt"):
                qpxyT = data_batch
                # print(qpxyT.shape)
                q0T = qpxyT[:, 0:1, :, 0]
                p0T = qpxyT[:, 1:2, :, 0]
                dq0T = qpxyT[:, 0:1, :, 1]
                dp0T = qpxyT[:, 1:2, :, 1]
                dq0N, dp0N = f_neur_hnn.forward_train(q0T, p0T)
                loss = loss_func(dq0T, dq0N) + loss_func(dp0T, dp0N)
            else:
                qpxyT, dt = data_batch
                qpxyN = []
                for j in range(nsteps):
                    q0T = qpxyT[:, 0:1, :, j]
                    p0T = qpxyT[:, 1:2, :, j]
                    if (model == "model_nssnn.pt"):
                        x0T = qpxyT[:, 2:3, :, j]
                        y0T = qpxyT[:, 3:4, :, j]
                        q1N, p1N, x1N, y1N = Nonsep_SymInt(q0T, p0T, x0T, y0T, dt, f_neur_nssnn.forward_train, epsN)
                        qpxyN.append(torch.cat([q1N, p1N, x1N, y1N], dim=1).unsqueeze(-1))
                    elif (model == "model_baseline.pt"):
                        q1N, p1N = RK2(q0T, p0T, dt, f_neur_baseline.forward, epsN)
                        qpxyN.append(torch.cat([q1N, p1N, q1N, p1N], dim=1).unsqueeze(-1))
                qpxyN = torch.cat(qpxyN, dim=-1)
                loss = loss_func(qpxyT[:, :, :, 1:], qpxyN[:, :, :, :])
            train_loss += loss.detach().cpu().item()
            # print(loss.detach().cpu().item())
            train_sample += 1
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (model == "model_baseline.pt"):
            f_neur_baseline.eval()
        elif (model == "model_hnn.pt"):
            f_neur_hnn.eval()
        elif (model == "model_nssnn.pt"):
            f_neur_nssnn.eval()
        with torch.no_grad():
            for batch_index, data_batch in enumerate(test_data_loader):
                if (model == "model_hnn.pt"):
                    qpxyT = data_batch
                    q0T = qpxyT[:, 0:1, :, 0]
                    p0T = qpxyT[:, 1:2, :, 0]
                    dq0T = qpxyT[:, 0:1, :, 1]
                    dp0T = qpxyT[:, 1:2, :, 1]
                    dq0N, dp0N = f_neur_hnn.forward(q0T, p0T)
                    loss = loss_func(dq0T, dq0N) + loss_func(dp0T, dp0N)
                else:
                    qpxyT, dt = data_batch
                    qpxyN = []
                    for j in range(nsteps):
                        q0T = qpxyT[:, 0:1, :, j]
                        p0T = qpxyT[:, 1:2, :, j]
                        if (model == "model_nssnn.pt"):
                            x0T = qpxyT[:, 2:3, :, j]
                            y0T = qpxyT[:, 3:4, :, j]
                            q1N, p1N, x1N, y1N = Nonsep_SymInt(q0T, p0T, x0T, y0T, dt, f_neur_nssnn.forward, epsN)
                            qpxyN.append(torch.cat([q1N, p1N, x1N, y1N], dim=1).unsqueeze(-1))
                        elif (model == "model_baseline.pt"):
                            q1N, p1N = RK2(q0T, p0T, dt, f_neur_baseline.forward, epsN)
                            qpxyN.append(torch.cat([q1N, p1N, q1N, p1N], dim=1).unsqueeze(-1))
                    qpxyN = torch.cat(qpxyN, dim=-1)
                    loss = loss_func(qpxyT[:, :, :, 1:], qpxyN[:, :, :, :])
                test_loss += loss.detach().cpu().item()
                test_sample += 1

        # writer.add_scalars('loss', {'train_loss': train_loss / train_sample,
        #                            'test_loss': test_loss / test_sample}, i + 1)
        # if i % 50 == 0:
        print(model)
        print(i, train_loss / train_sample, test_loss / test_sample)
        loss_data.append([i, train_loss / train_sample, test_loss / test_sample])
        #    test(model)
        if lowest_test_loss > test_loss / test_sample:
            if (model == "model_nssnn.pt"):
                torch.save(f_neur_nssnn.state_dict(), "model_nssnn.pt")
            elif (model == "model_hnn.pt"):
                torch.save(f_neur_hnn.state_dict(), "model_hnn.pt")
            elif (model == "model_baseline.pt"):
                torch.save(f_neur_baseline.state_dict(), "model_baseline.pt")
            lowest_test_loss = test_loss / test_sample
    with open(model + 'loss.dat', 'w+') as f:
        lendata = len(loss_data)
        for i in range(lendata):
            f.write(str(loss_data[i][0]))
            f.write('\t\t')
            f.write(str(loss_data[i][1]))
            f.write('\t\t')
            f.write(str(loss_data[i][2]))
            f.write('\n')


def test(model):
    qpxyT = torch.load('test.dat')
    qN = qpxyT[0:1, 0:1, :].to(device)
    pN = qpxyT[0:1, 1:2, :].to(device)
    xN = qpxyT[0:1, 2:3, :].to(device)
    yN = qpxyT[0:1, 3:4, :].to(device)
    qpxyN = [torch.cat([qN, pN, xN, yN], dim=1).detach().cpu()]
    if (model == "model_baseline.pt"):
        f_neur_baseline.eval()
    elif (model == "model_hnn.pt"):
        f_neur_hnn.eval()
    elif (model == "model_nssnn.pt"):
        f_neur_nssnn.eval()
    with torch.no_grad():
        for i in range(plot_points):
            if (model == "model_nssnn.pt"):
                qN, pN, xN, yN = Nonsep_SymInt(qN, pN, xN, yN, dtp.to(device), f_neur_nssnn.forward, epsN)
            elif (model == "model_hnn.pt"):
                qN, pN = RK2(qN, pN, dtp.to(device), f_neur_hnn.forward_train, epsN)
                xN = qN
                yN = pN
            elif (model == "model_baseline.pt"):
                qN, pN = RK2(qN, pN, dtp.to(device), f_neur_baseline.forward, epsN)
                xN = qN
                yN = pN
            qpxyN.append(torch.cat([qN, pN, xN, yN], dim=1).detach().cpu())
    qpxyN = torch.cat(qpxyN)
    qpxyN = to_np(qpxyN)
    qpxyT = to_np(qpxyT)

    plt.clf()
    plt.scatter(qpxyN[:, 0, 0], qpxyN[:, 1, 0], s=80, c="r", marker="s")
    plt.plot(qpxyT[:, 0, 0], qpxyT[:, 1, 0], c="b")
    plt.scatter(qpxyT[:, 0, 0], qpxyT[:, 1, 0], s=40, c="b")
    plt.draw()
    plt.show()
    plt.pause(1.)
    plt.savefig(model[:9])


def validation(model):
    if (model == "model_baseline.pt"):
        f_neur_baseline.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_baseline.to(device)
        f_neur_baseline.eval()
    elif (model == "model_hnn.pt"):
        f_neur_hnn.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_hnn.to(device)
        f_neur_hnn.eval()
    elif (model == "model_nssnn.pt"):
        f_neur_nssnn.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_nssnn.to(device)
        f_neur_nssnn.eval()

    qpxyT0 = torch.load('validation0.dat')
    qpxyT1 = torch.load('validation1.dat')
    Geo = 0.
    Err = 0.
    with torch.no_grad():
        for i in range(points_validation):
            qT0 = qpxyT0[i:i + 1, 0:1, :].to(device)
            pT0 = qpxyT0[i:i + 1, 1:2, :].to(device)
            xT0 = qpxyT0[i:i + 1, 2:3, :].to(device)
            yT0 = qpxyT0[i:i + 1, 3:4, :].to(device)
            qT1 = qpxyT1[i:i + 1, 0:1, :].to(device)
            pT1 = qpxyT1[i:i + 1, 1:2, :].to(device)
            xT1 = qpxyT1[i:i + 1, 2:3, :].to(device)
            yT1 = qpxyT1[i:i + 1, 3:4, :].to(device)
            if (model == "model_nssnn.pt"):
                qN1, pN1, xN1, yN1 = Nonsep_SymInt(qT0, pT0, xT0, yT0, dtp_validation.to(device), f_neur_nssnn.forward,
                                                   epsN)
            elif (model == "model_hnn.pt"):
                qN1, pN1 = RK2(qT0, pT0, dtp_validation.to(device), f_neur_hnn.forward, epsN)
            elif (model == "model_baseline.pt"):
                qN1, pN1 = RK2(qT0, pT0, dtp_validation.to(device), f_neur_baseline.forward, epsN)
            Geotemp = torch.abs((analyH(qN1, pN1) - analyH(qT0, pT0)) / torch.sqrt(
                analyH(qT0, pT0) ** 2 + 0.0000001)).sum().detach().cpu().item()
            Errtemp = (torch.abs(qN1 - qT1) + torch.abs(pN1 - pT1)).sum().detach().cpu().item()
            # print(analyH(qT0, pT0),analyH(qT1, pT1),analyH(qN1, pN1),Geotemp)
            Geo = Geo + Geotemp
            Err = Err + Errtemp
    print(model, Geo / points_validation, Err / points_validation)
    with open(model + 'Hamiltonian.dat', 'w') as f:
        f.write(str(Geo / points_validation))
        f.write('\t\t')
        f.write(str(Err / points_validation))


def Gen_Data_plot():
    qT = torch.tensor([[[0.]]])
    pT = torch.tensor([[[-3.]]])
    xT = qT
    yT = pT
    qpxyT = [torch.cat([qT, pT, xT, yT], dim=1)]
    f_true = KAnalysis()
    f_true.eval()
    with torch.no_grad():
        for i in range(plot_Np):
            qT, pT, xT, yT = Nonsep_SymInt(qT, pT, xT, yT, dt_Np, f_true.forward, epsT)
            qpxyT.append(torch.cat([qT, pT, xT, yT], dim=1))
    torch.save(torch.cat(qpxyT), 'test_plot.dat')


def validation_plot(model):
    if (model == "model_baseline.pt"):
        f_neur_baseline.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_baseline.to(device)
        f_neur_baseline.eval()
    elif (model == "model_hnn.pt"):
        f_neur_hnn.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_hnn.to(device)
        f_neur_hnn.eval()
    elif (model == "model_nssnn.pt"):
        f_neur_nssnn.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
        f_neur_nssnn.to(device)
        f_neur_nssnn.eval()
    qpxyT = torch.load('test_plot.dat')
    # print(qpxyT.shape)
    qN = qpxyT[0:1, 0:1, 0:1].to(device)
    pN = qpxyT[0:1, 1:2, 0:1].to(device)
    xN = qpxyT[0:1, 2:3, 0:1].to(device)
    yN = qpxyT[0:1, 3:4, 0:1].to(device)
    qpxyN = [torch.cat([qN, pN, xN, yN], dim=1).detach().cpu()]

    with torch.no_grad():
        for i in range(plot_Np):
            if (model == "model_nssnn.pt"):
                qN, pN, xN, yN = Nonsep_SymInt(qN, pN, xN, yN, dt_Np.to(device), f_neur_nssnn.forward, epsN)
            elif (model == "model_hnn.pt"):
                qN, pN = RK2(qN, pN, dt_Np.to(device), f_neur_hnn.forward, epsN)
            elif (model == "model_baseline.pt"):
                qN, pN = RK2(qN, pN, dt_Np.to(device), f_neur_baseline.forward, epsN)
            qpxyN.append(torch.cat([qN, pN, qN, pN], dim=1).detach().cpu())
    qpxyN = torch.cat(qpxyN)
    qpxyN = to_np(qpxyN)
    print(qpxyT.shape)
    qpxyT = to_np(qpxyT)
    dpq = np.abs(
        (qpxyN[:, 0, 0] ** 2 + 1) * (qpxyN[:, 1, 0] ** 2 + 1) - (qpxyT[:, 0, 0] ** 2 + 1) * (qpxyT[:, 1, 0] ** 2 + 1))
    plt.clf()
    plt.scatter(qpxyN[:, 0, 0], qpxyN[:, 1, 0], s=80, c="r", marker="s")
    plt.plot(qpxyT[:, 0, 0], qpxyT[:, 1, 0], c="b")
    plt.scatter(qpxyT[:, 0, 0], qpxyT[:, 1, 0], s=40, c="b")
    plt.draw()
    plt.show()
    plt.pause(1.)
    plt.savefig('out')
    with torch.no_grad():
        with open(model + 'qp.dat', 'w') as f:
            for i in range(plot_Np):
                f.write(str(qpxyN[i, 0, 0]))
                f.write('\t\t')
                f.write(str(qpxyN[i, 1, 0]))
                f.write('\t\t')
                f.write(str(qpxyT[i, 0, 0]))
                f.write('\t\t')
                f.write(str(qpxyT[i, 1, 0]))
                f.write('\n')
        with open(model + 'dqp.dat', 'w') as f:
            for i in range(plot_Np):
                f.write(str(i * lt_Np / plot_Np))
                f.write('\t\t')
                f.write(str(dpq[i]))
                f.write('\n')


if __name__ == '__main__':
    Gen_Data()
    # train("model_baseline.pt")
    # train("model_hnn.pt")
    train("model_nssnn.pt")
    # validation("model_baseline.pt")
    # validation("model_hnn.pt")
    validation("model_nssnn.pt")
    # Gen_Data_plot()
    # validation_plot("model_baseline.pt")
    # validation_plot("model_hnn.pt")
    # validation_plot("model_nssnn.pt")




