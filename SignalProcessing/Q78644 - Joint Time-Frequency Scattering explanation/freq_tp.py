# -*- coding: utf-8 -*-
"""Visualize log-frequency shift."""
import numpy as np
from kymatio.numpy import Scattering1D
from kymatio.visuals import imshow, make_gif
from kymatio.toolkit import fdts
import matplotlib.pyplot as plt

#%% Make CWT object ##########################################################
N = 2048
cwt = Scattering1D(shape=N, J=7, Q=8, average=False, out_type='list',
                   r_psi=.85, oversampling=999, max_order=1)

#%% Make signals & take CWT ##################################################
x0 = fdts(N, n_partials=4, seg_len=N//4, f0=N/12)[0]
x1 = fdts(N, n_partials=4, seg_len=N//4, f0=N/20)[0]

Wx0 = np.array([c['coef'].squeeze() for c in cwt(x0)[1:]])
Wx1 = np.array([c['coef'].squeeze() for c in cwt(x1)[1:]])

#%% Make GIF #################################################################
fig0, ax0 = plt.subplots(1, 2, figsize=(12, 5))
fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
# imshows
kw = dict(abs=1, ticks=0, show=0)
imshow(Wx0, ax=ax0[1], fig=fig0, **kw)
imshow(Wx1, ax=ax1[1], fig=fig1, **kw)

# plots
s, e = N//3, -N//3  # zoom
ax0[0].plot(x0[s:e])
ax1[0].plot(x1[s:e])
# ticks & ylims
mx = max(np.abs(x0).max(), np.abs(x1).max()) * 1.03
for ax in (ax0, ax1):
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylim(-mx, mx)
# titles
title_kw = dict(weight='bold', fontsize=20, loc='left')
for ax in (ax0, ax1):
    ax[0].set_title("x (zoomed)", **title_kw)
    ax[1].set_title("|CWT(x)|",   **title_kw)

# finalize & save
base_name = 'freq_tp'
fig0.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.02)
fig1.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.02)
fig0.savefig(f'{base_name}0.png', bbox_inches='tight')
fig1.savefig(f'{base_name}1.png', bbox_inches='tight')
plt.close(fig0)
plt.close(fig1)

# make GIF
make_gif('', f'{base_name}.gif', duration=1000, start_end_pause=0,
         delimiter=base_name, verbose=1, overwrite=1)
