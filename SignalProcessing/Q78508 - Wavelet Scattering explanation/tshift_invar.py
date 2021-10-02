# -*- coding: utf-8 -*-
"""Illustrating time-shift invariance."""
import numpy as np
import os
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D
from kymatio.visuals import plot, make_gif, plotscat
from kymatio.toolkit import l2
from scipy.signal.windows import tukey

#%%## Set params & create scattering object ##################################
N = 2048
J, Q = 8, 10
T = 2**J
# make CWT
average, oversampling = False, 999
ts = Scattering1D(shape=N, J=J, Q=Q, average=average, oversampling=oversampling,
                  out_type='list', max_order=1, T=T, max_pad_factor=None)
meta = ts.meta()

#%%# Create signal & warp it #################################################
f = 128
width = N//4
shift = width // 4

window = tukey(width)
window = np.pad(window, (N - width)//2)
t = np.linspace(0, 1, N, 0)
x = np.cos(2*np.pi * f * t) * window

x0 = np.roll(x, -shift//2)
x1 = np.roll(x,  shift//2)

#%%# CWT
_Scx0, _Scx1 = ts(x0), ts(x1)
Scx0 = np.vstack([c['coef'] for c in _Scx0])[meta['order'] == 1]
Scx1 = np.vstack([c['coef'] for c in _Scx1])[meta['order'] == 1]
freqs = N * meta['xi'][meta['order'] == 1][:, 0]

x_all = [x0, x1]
cwt_all = [Scx0, Scx1]

#%% make scattering objects
T0 = 2**(J - 2) * 2
T1 = 2**J * 2
kw = dict(shape=N, J=J, Q=Q, average=True,
          out_type='list', max_order=1, max_pad_factor=3)
ts0 = Scattering1D(**kw, T=T0, oversampling=0)
ts1 = Scattering1D(**kw, T=T1, oversampling=0)
meta = ts0.meta()

#%% scatter
_Scx00, _Scx01, _Scx10, _Scx11 = ts0(x0), ts0(x1), ts1(x0), ts1(x1)
Scx00 = np.vstack([c['coef'] for c in _Scx00])[meta['order'] == 1]
Scx01 = np.vstack([c['coef'] for c in _Scx01])[meta['order'] == 1]
Scx10 = np.vstack([c['coef'] for c in _Scx10])[meta['order'] == 1]
Scx11 = np.vstack([c['coef'] for c in _Scx11])[meta['order'] == 1]
freqs = N * meta['xi'][meta['order'] == 1][:, 0]

x_all = [x0, x1]
Scx_all = [Scx00, Scx10, Scx01, Scx11]

#%%# Time-shift GIF ##########################################################
# configure
base_name = "tshift"
images_ext = ".png"
savedir = r"C:\Desktop\School\Deep Learning\DL_Code\signals\viz_gen"
overwrite = True

# common plot kwargs
title_kw  = dict(weight='bold', fontsize=20, loc='left')
label_kw  = dict(weight='bold', fontsize=18)
imshow_kw = dict(cmap='turbo', aspect='auto', vmin=0)
vmax00 = np.array(cwt_all).max()*.95
vmax0 = max(Scx00.max(), Scx01.max())
vmax1 = max(Scx10.max(), Scx11.max())

for i in (0, 2):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # cwt ####################################################################
    # plot
    ax = axes[0, 0]
    ax.plot(t, x_all[i//2])
    # style
    txt = ("" if i == 0 else
           " - %.2f" % (shift/N))
    ax.set_title("x(t%s)" % txt, **title_kw)
    xticks = np.array([*np.linspace(0, N, 6, 1)[:-1], N - 1])
    xticklabels = ['%.2f' % tk for tk in t[np.round(xticks).astype(int)]]
    ax.set_xticks(xticks / N)
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_xlim(-.03, 1.03)
    ax.set_yticks([])

    # imshow
    ax = axes[0, 1]
    ax.imshow(cwt_all[i//2], **imshow_kw, vmax=vmax00)
    # style
    ax.set_title("|CWT(x)|", **title_kw)
    ax.set_yticks([])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14)

    # scattering #############################################################
    Scx0, Scx1 = Scx_all[i], Scx_all[i + 1]

    # imshow
    ax = axes[1, 0]
    ax.imshow(Scx0, **imshow_kw, vmax=vmax0)
    # style
    ax.set_title("S(x) | T=%.2f sec" % (T0 / N), **title_kw)
    ax.set_xticks([]); ax.set_yticks([])

    # # imshow
    ax = axes[1, 1]
    ax.imshow(Scx1, **imshow_kw, vmax=vmax1)
    # style
    ax.set_title("S(x) | T=%.2f sec" % (T1 / N), **title_kw)
    ax.set_xticks([]); ax.set_yticks([])

    # finalize
    plt.subplots_adjust(hspace=.15, wspace=.04)

    # save
    path = os.path.join(savedir, f'{base_name}{i}{images_ext}')
    if os.path.isfile(path) and overwrite:
        os.unlink(path)
    if not os.path.isfile(path):
        fig.savefig(path, bbox_inches='tight')
    plt.close()

# make into gif
savepath = os.path.join(savedir, f'{base_name}.gif')
make_gif(loaddir=savedir, savepath=savepath, ext=images_ext, duration=750,
         overwrite=overwrite, delimiter=base_name, verbose=1, start_end_pause=0,
         delete_images=True)

#%%# plot superimposed #######################################################
def plot_superimposed(g0, g1, T, ax):
    t_idxs = np.arange(len(g0))
    for t_idx in t_idxs:
        plot([t_idx, t_idx], [g0[t_idx], g1[t_idx]],
             color='tab:red', linestyle='--', ax=ax)

    title = ("T={:.2f} | reldist={:.2f}".format(T / N, float(l2(g0, g1))),
             {'fontsize': 20})
    plotscat(g0, title=title, ax=ax)
    plotscat(g1, xlims=(-len(g0)/40, 1.03*(len(g0) - 1)),
             ylims=(0, 1.03*max(g0.max(), g1.max())), ax=ax)

mx_idx = np.argmax(Scx00.sum(axis=-1))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot_superimposed(Scx00[mx_idx], Scx01[mx_idx], T0, axes[0])
plot_superimposed(Scx10[mx_idx], Scx11[mx_idx], T1, axes[1])

plt.subplots_adjust(wspace=.09)
plt.show()

#%% Reldist vs T #############################################################
kw = dict(shape=N, J=6, Q=8, average=True,
          out_type='array', max_order=1, max_pad_factor=4)
# implementation switches to simple average with unpad which
# detracts from curve slightly in log space, so do N-1 instead of N
T_all = list(range(1, N + 1, 16)) + [N - 1]
ts_all = [Scattering1D(**kw, T=T) for T in T_all]

#%%
Scx0_all = [ts(x0) for ts in ts_all]
Scx1_all = [ts(x1) for ts in ts_all]
reldist_all = [l2(S0, S1).squeeze() for S0, S1 in zip(Scx0_all, Scx1_all)]

#%% plot
T_all_sec = np.array(T_all) / N

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
plot(T_all_sec, reldist_all, ax=axes[0],
     hlines=(0, dict(color='tab:red', linestyle='--', linewidth=1)),
     title=("Scattering coefficient relative distance", {'fontsize': 20}),
     xlabel="T [sec]", ylabel="reldist")

T_all_sec_log = np.log2(T_all_sec)
reldist_all_log = np.log10(reldist_all)
plot(T_all_sec_log, reldist_all_log, ax=axes[1],
     title=("logscaled", {'fontsize': 20}),
     xlabel="log2(T) [sec]", ylabel="log10(reldist)")

plt.subplots_adjust(wspace=.1)
plt.show()
