# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76754/50076 ################################
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

def analytic(x):
    N = len(x)
    xf = fft(x)
    xf[1:N//2] *= 2
    if N % 2 == 1:
        xf[N//2] *= 2
    xf[N//2 + 1:] = 0
    xa = ifft(xf)
    assert np.allclose(xa.real, x)
    return xa

def plot(x, title=None, show=0):
    plt.plot(x)
    if title is not None:
        plt.title(title, loc='left', weight='bold', fontsize=17)
    if show:
        plt.show()

#%%###########################################################################
# load
x = np.load('data.npy')
N = len(x)
xa = analytic(x)

# pad + take hilbert
xp = np.pad(x, N, mode='reflect')
xpa = analytic(xp)
# unpad
xpu = xp[N:-N]
xpau = xpa[N:-N]

#%% visualize
plot(x, title="original")
plot(np.abs(xa), show=1)

plot(xpu, title="reflect-padded + unpadded")
plot(np.abs(xpau), show=1)
