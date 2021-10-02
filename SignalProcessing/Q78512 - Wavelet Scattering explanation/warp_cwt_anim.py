# -*- coding: utf-8 -*-
"""Sinusoid and echirp warping CWT GIF."""
# https://dsp.stackexchange.com/q/78512/50076 ################################
# https://dsp.stackexchange.com/q/78514/50076 ################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kymatio.torch import Scattering1D
from kymatio.visuals import imshow
from kymatio.toolkit import _echirp_fn

def tau(t, K=.08):
    return np.cos(2*np.pi * t**2) * K

def adtau(t, K=.08):
    return np.abs(np.sin(2*np.pi * t**2) * t * 4*np.pi*K)

import torch
USE_GPU = bool('cuda' if torch.cuda.is_available() else 'cpu')

#%% Create scattering object #################################################
N, f = 2048, 64
J, Q = 8, 8
T = 512
# increase freq width to improve temporal localization
# to not confuse ill low-freq localization with warp effects, for echirp
ts = Scattering1D(shape=N, J=J, Q=Q, average=False, oversampling=999,
                  T=T, out_type='list', r_psi=.82)
if USE_GPU:
    ts.cuda()

#%%# configure time-warps ####################################################
K_init = .01
n_pts = 64
t = np.linspace(0, 1, N, 0)

p = ts.psi1_f[0]
QF = p['xi'] / p['sigma']

adtau_init_max = adtau(t, K=K_init).max()
K_min = (1/QF) / 10
K_max = (1 / adtau_init_max) * K_init

K_all = np.logspace(np.log10(K_min), np.log10(K_max), n_pts - 1, 1)
K_all = np.hstack([0, K_all])
tau_all = np.vstack([tau(t, K=k) for k in K_all])
adtau_max_all = np.vstack([adtau(t, K=k).max() for k in K_all])

assert adtau_max_all.max() <= 1, adtau_max_all.max()

#%% Animate #################################################################
class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, plot_frames, imshow_frames, t, freqs, adtau_max_all):
        self.plot_frames = plot_frames
        self.imshow_frames = imshow_frames
        self.t = t
        self.freqs = freqs
        self.adtau_max_all = adtau_max_all

        self.N = len(t)
        self.n_rows = len(imshow_frames[0])
        self.n_frames = len(imshow_frames)

        self.title_kw = dict(weight='bold', fontsize=15, loc='left')
        self.label_kw = dict(weight='bold', fontsize=14, labelpad=3)
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left")

        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7))

        # plots ##############################################################
        ax = axes[0]
        self.lines0 = []

        ax.plot(plot_frames[0])
        self.lines0.append(ax.lines[-1])

        ax.set_xticks(np.linspace(0, self.N, 6, endpoint=True))
        xticklabels = np.linspace(t.min(), t.max(), 6, endpoint=True)
        ax.set_xticklabels(["%.1f" % xt for xt in xticklabels])

        self.txt0 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        # imshows ############################################################
        ax = axes[1]
        im = ax.imshow(self.imshow_frames[0], cmap='turbo', animated=True,
                       aspect='auto')
        self.ims1 = [im]

        ax.set_xticks(np.linspace(0, self.N, 6, endpoint=True))
        xticklabels = np.linspace(t.min(), t.max(), 6, endpoint=True)
        ax.set_xticklabels(["%.1f" % xt for xt in xticklabels])
        yticks = np.linspace(0, self.n_rows - 1, 6, endpoint=True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["%.2f" % (self.freqs[int(yt)] / self.N)
                            for yt in yticks])

        ax.set_title("|CWT(x(t))|", **self.title_kw)

        # finalize #######################################################
        fig.subplots_adjust(left=.05, right=.99, bottom=.06, top=.94,
                            wspace=.15)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        # plots ##############################################################
        self.lines0[0].set_ydata(self.plot_frames[frame_idx])

        if frame_idx == 0:
            txt = r"$x(t) \ ... \ |\tau'(t)| = 0$"
        else:
            txt = r"$x(t) \ ... \ |\tau'(t)| < %.3f$" % self.adtau_max_all[
                frame_idx]
        self.txt0.set_text(txt)

        # imshows ############################################################
        self.ims1[0].set_array(self.imshow_frames[frame_idx])

        # finalize ###########################################################
        self._drawn_artists = [*self.lines0, self.txt0, *self.ims1]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

#%%# Run on each signal ######################################################
def extend(x):
    return np.array(list(x) + 6*[x[-1]])

def run(x_all, adtau_max_all, ts, name):
    meta = ts.meta()
    freqs = N * meta['xi'][meta['order'] == 1][:, 0]

    Scx_all0 = [ts(x) for x in x_all]
    Scx_all = np.vstack([(np.vstack([c['coef'].cpu().numpy() for c in Scx]
                                    )[meta['order'] == 1])[None]
                         for Scx in Scx_all0])

    # animate
    x_all, Scx_all, adtau_max_all = [extend(h) for h in
                                     (x_all, Scx_all, adtau_max_all)]
    plot_frames, imshow_frames = x_all, Scx_all

    ani = PlotImshowAnimation(plot_frames, imshow_frames, t, freqs,
                              adtau_max_all)
    ani.save(f'warp_cwt_{name}.mp4', fps=10)
    plt.show()

#%% echirp
_t = (t - tau_all)
a0 = np.cos(2*np.pi * 3.7 * _t)
x_all = np.cos(_echirp_fn(fmin=40, fmax=N/8)(_t)) * a0
#%%
run(x_all, adtau_max_all, ts, "echirp")
#%% visualize warps for sine case with lower freq
def _tt(title):
    return (title, {'fontsize': 20})

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
x_all_viz = x_all#np.cos(2*np.pi * 64 * (t - tau_all))
imshow(x_all_viz, title=_tt("x(t - tau(t)) for all tau"), show=0, ax=axes[0],
       xlabel="time", ylabel="max(|tau'|)", yticks=adtau_max_all)
imshow(tau_all, title=_tt("all tau(t)"), show=0, ax=axes[1],
       xlabel="time", yticks=0)

for ax in axes:
    ax.set_xticks(np.linspace(0, len(t), 6))
    ax.set_xticklabels([0, .2, .4, .6, .8, 1])

plt.subplots_adjust(wspace=.03)
fig.savefig("sine_warp.png", bbox_inches='tight')
plt.show()
plt.close(fig)

#%% sine
x_all = np.cos(2*np.pi * f * (t - tau_all))
run(x_all, adtau_max_all, ts, "sine")
