# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/85745/50076
import numpy as np
import matplotlib.pyplot as plt
# from ?.toolkit import fft_upsample  # to be released soon

def E(x):
    return np.sum(np.abs(x)**2)

def viz(t1, t2, x1, x2, mode='bar', title=True):
    tkw = dict(fontsize=18, weight='bold', loc='left')
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 6))

    if mode == 'bar':
        Dt1 = t1[1] - t1[0]  # sampling period
        Dt2 = t2[1] - t2[0]
        axes[0].bar(t1, x1, .9*Dt1)
        axes[1].bar(t2, x2, .9*Dt2)
    else:
        axes[0].plot(t1, x1)
        axes[1].plot(t2, x2)

    if title:
        if title is True:
            title1 = "sum(|x|^2)={:.3g}, N={}".format(E(x1), len(x1))
            title2 = "sum(|x|^2)={:.3g}, N={}".format(E(x2), len(x2))
        elif isinstance(title, tuple):
            title1, title2 = title
        axes[0].set_title(title1, **tkw)
        axes[1].set_title(title2, **tkw)

    fig.subplots_adjust(wspace=.03)
    plt.show()

def viz_sines(N1, N2, T1, T2, f1=1, f2=1):
    t1 = np.linspace(0, T1, N1, endpoint=False)
    t2 = np.linspace(0, T2, N2, endpoint=False)
    x1 = np.cos(2*np.pi * f1 * t1)
    x2 = np.cos(2*np.pi * f2 * t2)

    viz(t1, t2, x1, x2, title=True)

#%%
N1, N2 = 10, 20
for duration in (1, 1.25):
    viz_sines(N1, N2, duration, duration)

#%%
N = 20
T1, T2 = 1, 2
f1, f2 = 2, 1
viz_sines(N, N, T1, T2, f1, f2)

#%%
# t = np.linspace(0, 1.25, 20, endpoint=False)
# x = np.cos(2*np.pi * t)
# x_up = fft_upsample(x, factor=128, time_to_time=True, real=True)
# t_up = np.linspace(0, 1.25, len(x_up), endpoint=False)
# x_up4 = np.hstack([x_up]*4)
# t_up4 = np.linspace(0, 1.25*4, len(x_up)*4, endpoint=False)

# viz(t_up, t_up4, x_up, x_up4, mode='plot',
#     title=("x_upsampled", "x_upsampled, 4 periods"))
