# -*- coding: utf-8 -*-
"""Second-order unaveraged scattering on A.M. cosine and White Gaussian Noise."""
# https://dsp.stackexchange.com/q/78508/50076 ################################
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D
from kymatio.visuals import (plot, imshow,
                             filterbank_heatmap, filterbank_scattering)

#%%
def cwt_and_viz(x, ts, show_max_rows=False):
    # do manually to override `j2 > j1` and see all coeffs
    xp = np.pad(x, [ts.pad_left, ts.pad_right], mode='reflect')
    U0_f = fft(xp)

    # |CWT|
    U1 = []
    for p1 in ts.psi1_f:
        U1.append(np.abs(ifft(U0_f * p1[0])))
    U1 = np.array(U1)
    U1_f = fft(U1, axis=-1)

    # U2
    U2 = []
    for p2 in ts.psi2_f:
        U2.append(np.abs(ifft(U1_f * p2[0])))
    U2 = np.array(U2)

    # S1 = ifft(U1_f * ts.phi_f[0]).real
    # U2_f = fft(U2, axis=-1)
    # S2 = ifft(U2_f * ts.phi_f[0][None, None]).real

    # unpad all
    s, e = ts.ind_start[0], ts.ind_end[0]
    U1, U2 = [g[..., s:e] for g in (U1, U2)]
    # S1, S2 = [g[..., s:e] for g in (S1, S2)]

    # viz first-order ########################################################
    fs = N
    xi1s = np.array([p['xi'] for p in ts.psi1_f]) * fs
    imshow(U1, abs=1, title="|CWT(x)|", yticks=xi1s, ylabel="frequency [Hz]",
            xlabel="time", xticks=t, w=.8, h=.6)

    if show_max_rows:
        mx_idx = np.argmax(np.sum(U1**2, axis=-1))
        xi1 = xi1s[mx_idx]
        title = "|CWT(x, xi1={:.1f} Hz)| (max row)".format(xi1)
        plot(t, U1[mx_idx], title=title, ylims=(0, None), show=1,
             ylabel="A.M. rate [Hz]", xlabel="time [sec]", w=.7, h=.9)

    # viz second-oder ########################################################
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    xi2s = np.array([p['xi'] for p in ts.psi2_f]) * fs

    # U2 /= (U1 + U1.max()/10000)[None]
    # U2 = np.log(1 + U2)
    mx = np.max(U2) * .9
    for U2_idx, ax in enumerate(axes.flat):
        U2_idx += 1
        xi2 = xi2s[U2_idx]
        title = "xi2 = {:.1f} Hz".format(xi2)
        imshow(U2[U2_idx], abs=1, title=title, norm=(0, mx),
               show=0, ax=ax, ticks=0)

    label_kw = dict(weight='bold', fontsize=17)
    axes[0, 0].set_ylabel("xi1",  **label_kw)
    axes[1, 0].set_ylabel("xi1",  **label_kw)
    axes[2, 0].set_ylabel("xi1",  **label_kw)
    axes[2, 0].set_xlabel("time", **label_kw)
    axes[2, 1].set_xlabel("time", **label_kw)
    axes[2, 2].set_xlabel("time", **label_kw)

    plt.subplots_adjust(wspace=.02, hspace=.1)
    plt.show()

    if show_max_rows:
        mx_idx2 = np.argmax(np.sum(U2**2, axis=(1, 2)))
        mx_idx1 = np.argmax(np.sum(U2[mx_idx2]**2, axis=-1))
        xi1, xi2 = xi1s[mx_idx1], xi2s[mx_idx2]
        title = "|CWT(|CWT(x, xi1={:.1f} Hz)|, xi2={:.1f} Hz)| (max row)".format(
            xi1, xi2)
        U2_max_row = U2[mx_idx2, mx_idx1]

        plot(t, U2_max_row, title=title, show=1, xlabel="time [sec]",
             ylabel="Rate of A.M. rate [Hz^2]",
             ylims=(0, U2_max_row.max() * 1.03))

#%%
N = 2049
kw = dict(shape=N, J=9, Q=8, max_pad_factor=None, oversampling=999, max_order=1)
ts = Scattering1D(**kw)

#%%
filterbank_scattering(ts, second_order=1, zoom=-1)
filterbank_heatmap(ts, second_order=True)

#%%# AM cosine ###############################################################
f0, f1 = 64, 3
t = np.linspace(0, 1, N, 1)
c = np.cos(2*np.pi * f0 * t)
a = (1 + (np.cos(2*np.pi * f1 * t))) / 2
x = a * c

title = "$\cos(2\pi {} t) \cdot (1 + \cos(2\pi {} t))/2$".format(f0, f1)
plot(t, x, show=1, title=title, xlabel="time [sec]")
#%%
cwt_and_viz(x, ts, show_max_rows=1)

#%%# WGN #####################################################################
np.random.seed(0)
x = np.random.randn(N)
plot(t, x, show=1, title="White Gaussian Noise", xlabel="time [sec]")

#%%
cwt_and_viz(x, ts, show_max_rows=0)
