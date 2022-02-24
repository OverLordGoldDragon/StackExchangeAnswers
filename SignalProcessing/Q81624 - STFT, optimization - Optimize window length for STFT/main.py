# -*- coding: utf-8 -*-
"""Optimize window length for STFT, plain window demo."""
# https://dsp.stackexchange.com/q/81624/50076 ################################
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

def plot(x, title):
    plt.plot(x)
    plt.title(title, loc='left', weight='bold', fontsize=18)
    plt.gcf().set_size_inches(9, 7)
    plt.show(x)


def hann(N):
    Nint = torch.floor(N).int()
    Nc = torch.clamp(N, min=Nint, max=Nint + 1)
    t = torch.arange(Nint) / Nc
    w = .5 * (1 - torch.cos(2*torch.pi *t))
    return w


N = nn.Parameter(torch.tensor(129.))

# optimal window length
N_ref = 160
w_ref = hann(torch.tensor(float(N_ref)))
# add room for overshoot
x = F.pad(w_ref, [40, 40])

LR = 100000
Ns, grads, outs = [], [], []
for i in range(20):
    w = hann(N)
    w = w / torch.norm(w)  # L2 norm to ensure `max(conv)` peaks at `N_ref`
    conv = torch.conv1d(x[None, None], w[None, None])
    out = 1. / torch.max(conv)  # inverse of peak cross-correlation to minimize
    out.backward()

    with torch.no_grad():
        N -= LR * N.grad

        Ns.append(float(N.detach().numpy()))
        grads.append(float(N.grad.detach().numpy()))
        outs.append(float(out.detach().numpy()))


    # manual `optimizer.zero_grad()`
    N.grad.requires_grad_(False)
    N.grad.zero_()

plot(Ns,    "N vs iteration")
plot(grads, "N.grad vs iteration")
plot(outs,  "loss vs iteration")
