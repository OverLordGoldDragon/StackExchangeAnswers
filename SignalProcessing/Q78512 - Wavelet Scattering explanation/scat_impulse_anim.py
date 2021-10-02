# -*- coding: utf-8 -*-
"""Scattering vs CWT: impulse response GIF, echirp warping GIF."""
# https://dsp.stackexchange.com/q/78512/50076 ################################
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kymatio.torch import Scattering1D
from kymatio.visuals import imshow, plot, make_gif
from kymatio.toolkit import _echirp_fn

import torch
USE_GPU = bool('cuda' if torch.cuda.is_available() else 'cpu')

#%%###########################################################################
N = 2048
x = np.zeros(N)
x[N//2] = 1  # discrete Dirac delta

#%%###########################################################################
J, Q = 8, 8
T_all = np.arange(1, N/2 + 1, 8)
T_all = np.logspace(np.log2(1), np.log2(N - 1), N//8, base=2, endpoint=True)

kw = dict(shape=N, J=J, Q=Q, oversampling=999, max_order=1, max_pad_factor=None)
ts0 = Scattering1D(**kw, average=False, out_type='list')
ts_all = [Scattering1D(**kw, average=True, out_type='array', T=T)
          for T in T_all]
if USE_GPU:
    for ts in ts_all:
        ts.cuda()
    ts0.cuda()
meta_all = [ts.meta() for ts in ts_all]

#%%
Scx0 = np.array([np.array([c['coef'].cpu().numpy() for c in ts0(x)[1:]])])
Scx_all = np.array([ts(x)[1:].cpu().numpy() for ts in ts_all])

#%% extend
def extend(x):
    return np.array(list(x) + 6*[x[-1]])

Scx0, Scx_all, T_all = [extend(h) for h in (Scx0, Scx_all, T_all)]

#%%# Animate #################################################################
class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, imshow_frames0, imshow_frames1, T_all, mx, rescaled):
        self.imshow_frames0 = imshow_frames0
        self.imshow_frames1 = imshow_frames1
        self.T_all = T_all
        self.mx = mx
        self.rescaled = rescaled

        self.N = imshow_frames0[0].shape[-1]
        self.n_rows = len(imshow_frames0[0])
        self.n_frames = len(imshow_frames1)

        self.title_kw = dict(weight='bold', fontsize=15, loc='left')
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left", weight='bold')

        fig, axes = plt.subplots(1, 2, figsize=(18/1.7, 8/1.7))

        # color norm
        mx = max(np.max(imshow_frames0), np.max(imshow_frames1))

        # plots ##############################################################
        ax = axes[0]
        im = ax.imshow(self.imshow_frames0[0], cmap='turbo', animated=True,
                       aspect='auto', vmin=0, vmax=self.mx)
        self.ims0 = [im]
        nm = "CWT'" if self.rescaled else "CWT"
        ax.set_title(f"|{nm}(delta(t))|", **self.title_kw)
        ax.set_xticks([]); ax.set_yticks([])

        # imshows ############################################################
        ax = axes[1]
        im = ax.imshow(self.imshow_frames1[0], cmap='turbo', animated=True,
                       aspect='auto', vmin=0, vmax=self.mx)
        self.ims1 = [im]

        ax.set_xticks([]); ax.set_yticks([])
        self.txt1 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        # finalize #######################################################
        fig.subplots_adjust(left=.01, right=.99, bottom=.02, top=.94,
                            wspace=.05)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        # imshows ############################################################
        self.ims1[0].set_array(self.imshow_frames1[frame_idx])

        T = self.T_all[frame_idx]
        if T == N - 1:
            T = N
        nm = "S1'" if self.rescaled else "S1"
        self.txt1.set_text(f"{nm}(delta(t)) | T=%.1f" % T)

        # finalize ###########################################################
        self._drawn_artists = [*self.ims1, self.txt1]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

#%%
def run(T_target, T_idx, rescaled):
    # gif ####################################################################
    imshow_frames0 = Scx0
    imshow_frames1 = Scx_all

    mx = max(np.max(imshow_frames0), np.max(imshow_frames1)) * .98
    if rescaled:
        imshow_frames0 = np.array([r * (mx / r.max(axis=-1)[:, None])
                                   for r in Scx0])
        imshow_frames1 = np.array([r * (mx / r.max(axis=-1)[:, None])
                                   for r in Scx_all])

    ani = PlotImshowAnimation(imshow_frames0, imshow_frames1, T_all, mx, rescaled)
    nm = "_resc" if rescaled else ""
    ani.save(f'scat_impulse{nm}.mp4', fps=20)
    plt.show()

    # plots - compare slices #################################################
    ridx0, ridx1 = 2, 20
    S0, S1 = imshow_frames0[0], imshow_frames1[T_idx]
    r00, r01 = S0[ridx0], S0[ridx1]
    r10, r11 = S1[ridx0], S1[ridx1]

    def _tt(title):
        return (title, {'fontsize': 22})

    plot(r00, title=_tt("row {} & {}, CWT".format(ridx0, ridx1)))
    plot(r01, show=1, w=.8, ticks=0)

    plot(r10, title=_tt("row {} & {}, Scattering, T={}".format(
        ridx0, ridx1, T_target)))  # show `T_target` since it's nicer
    plot(r11, show=1, w=.8, ticks=0)

#%% set slice comparison target
T_target = 128
# fetch closest to `T_target`
T_idx = np.argmin(np.abs(T_all - T_target))
T_comp = T_all[T_idx]
assert abs(T_comp - T_target) < 4

#%%
run(T_target, T_idx, rescaled=True)
#%%
run(T_target, T_idx, rescaled=False)

#%%# Compare warping on freq-scaled echirps ##################################
def tau(t, K=.08):
    return (np.cos(2*np.pi * t**2) * K + 1)

t = np.linspace(0, 1, N, 0)
K = .0316  # `K_all[40]`
_tau = tau(t, K=K)

g0 = np.cos(_echirp_fn(fmin=128, fmax=N/4)( _tau * t))
g1 = np.cos(_echirp_fn(fmin=32,  fmax=N/16)(_tau * t))

T = 256
kw2 = dict(shape=N, J=10, Q=16, T=T, oversampling=999, max_order=1,
           max_pad_factor=None, out_type='list')
_ts0 = Scattering1D(**kw2, average=False)
_ts1 = Scattering1D(**kw2, average=True)
if USE_GPU:
    _ts0.cuda(); _ts1.cuda()

#%%# Make gif ################################################################
savedir = r"C:\Desktop\School\Deep Learning\DL_Code\signals\warp\\"
base_name = 'warp_sc'
ext = '.png'

def sc(x, ts):
    out = ts(x)[1:]
    return np.array([c['coef'].cpu().numpy() for c in out])

ts1 = ts_all[T_idx]
Wx0,  Wx1  = sc(g0, _ts0), sc(g1, _ts0)
Scx0, Scx1 = sc(g0, _ts1), sc(g1, _ts1)

shift = -32
Wx1, Scx1 = [np.roll(S, shift, axis=0) for S in (Wx1, Scx1)]
mx = max(W.max() for W in (Wx0, Wx1, Scx0, Scx1))*.95
pkw = dict(abs=1, ticks=0, show=0, norm=(0, mx))

Wx_all, Scx_all = [Wx0, Wx1], [Scx0, Scx1]
for i in (0, 1):
    Wx, Scx = Wx_all[i], Scx_all[i]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    name = "x" if i == 0 else "x_tau"
    if i == 0:
        title0 = "CWT(x_tau)"
        title1 = f"S(x_tau), T={T}"
    else:
        title0 = "CWT(y_tau) | shifted to overlap"
        title1 = f"S(y_tau), T={T} | shifted to overlap"
    title0, title1 = (title0, {'fontsize': 18.5}), (title1, {'fontsize': 18.5})

    imshow(Wx,  ax=axes[0], title=title0, **pkw)
    imshow(Scx, ax=axes[1], title=title1, **pkw)
    plt.subplots_adjust(wspace=.03)

    path = os.path.join(savedir, f'{base_name}{i}{ext}')
    # plt.show()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# make into gif
savepath = os.path.join(savedir, 'warp_sc.gif')
make_gif(savedir, savepath, duration=500, start_end_pause=0, overwrite=1,
         delimiter=base_name, ext=ext, delete_images=True)

#%%# Instantaneous ridges ####################################################
from ssqueezepy import ssq_cwt

skw = dict(wavelet=('gmw', {'gamma': 1, 'beta': 1}), scales='log')
# drop some rows to ~match number of octaves with scattering's
Tx0 = ssq_cwt(g0, **skw)[0][:-75]
Tx1 = ssq_cwt(g1, **skw)[0][:-75]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
pkw = dict(abs=1, ticks=0, show=0)
imshow(Tx0, **pkw, ax=axes[0], title=("|ssq_cwt(x_tau)|", {'fontsize': 19}))
imshow(Tx1, **pkw, ax=axes[1], title=("|ssq_cwt(y_tau)|", {'fontsize': 19}))

plt.subplots_adjust(wspace=.02)
savepath = os.path.join(savedir, 'ssq_xy.png')
fig.savefig(savepath)
plt.close(fig)
