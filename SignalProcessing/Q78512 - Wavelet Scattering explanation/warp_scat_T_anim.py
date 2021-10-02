# -*- coding: utf-8 -*-
"""Scattering unwarped vs warped echirp GIF, sweeping `T`."""
# https://dsp.stackexchange.com/q/78512/50076 ################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kymatio.torch import Scattering1D
from kymatio.visuals import plot
from kymatio.toolkit import _echirp_fn, l2

def tau(t, K=.08):
    return np.cos(2*np.pi * t**2) * K

def adtau(t, K=.08):
    return np.abs(np.sin(2*np.pi * t**2) * t * 4*np.pi*K)

import torch
USE_GPU = bool('cuda' if torch.cuda.is_available() else 'cpu')

#%%###########################################################################
N = 2048
J, Q = 8, 8
T_all = np.arange(1, N/2 + 1, 8)
T_all = np.logspace(np.log2(1), np.log2(N - 1), N//8, base=2, endpoint=True)

kw = dict(shape=N, J=J, Q=Q, average=True, oversampling=999, out_type='array',
          max_order=1, max_pad_factor=None, r_psi=.82)
ts_all = [Scattering1D(**kw, T=T) for T in T_all]
meta_all = [ts.meta() for ts in ts_all]

if USE_GPU:
    for ts in ts_all:
        ts.cuda()

#%%###########################################################################
meta = meta_all[0]
freqs = N * meta['xi'][meta['order'] == 1][:, 0]
#%%
t = np.linspace(0, 1, N, 0)
_tau = tau(t, K=.012)
a0 = np.cos(2*np.pi * 3.7 * (t - _tau))
x0 = np.cos(_echirp_fn(fmin=40, fmax=N/8)(t)) * a0
x1 = np.cos(_echirp_fn(fmin=40, fmax=N/8)(t - _tau)) * a0
#%%
Scx_all0 = np.array([ts(x0)[1:].cpu().numpy() for ts in ts_all])
Scx_all1 = np.array([ts(x1)[1:].cpu().numpy() for ts in ts_all])
#%%
dists = np.array([l2(S0, S1) for S0, S1 in zip(Scx_all0, Scx_all1)]).squeeze()
plot(T_all, dists, ylims=(0, None), show=1)

#%% extend frames
def extend(x):
    return np.array(6*[x[0]] + list(x) + 6*[x[-1]])

Scx_all0, Scx_all1, dists, T_all = [extend(h) for h in
                                    (Scx_all0, Scx_all1, dists, T_all)]

#%% Animate #################################################################
class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, imshow_frames0, imshow_frames1, plot_frame, T_all, vmax):
        self.imshow_frames0 = imshow_frames0
        self.imshow_frames1 = imshow_frames1
        self.plot_frame = plot_frame
        self.T_all = T_all
        self.vmax = vmax

        self.N = imshow_frames0[0].shape[-1]
        self.T_all_sec_log = np.log2(self.T_all / N)
        self.n_frames = len(imshow_frames0)

        self.label_kw = dict(weight='bold', fontsize=15, labelpad=3)
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left")

        fig = plt.figure(figsize=(18/1.5, 15/1.8))
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(3, 2, (3, 4))
        axes = [ax0, ax1, ax2]

        # imshow1 ############################################################
        ax = axes[0]
        imshow_kw = dict(cmap='turbo', aspect='auto', animated=True,
                         vmin=0, vmax=self.vmax)
        im = ax.imshow(self.imshow_frames0[0], **imshow_kw)
        self.ims0 = [im]

        self.txt0 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])

        # imshow2 ############################################################
        ax = axes[1]
        im = ax.imshow(self.imshow_frames1[0], **imshow_kw)
        self.ims1 = [im]

        self.txt1 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        ax.set_xticks([]); ax.set_yticks([])

        # plot ###############################################################
        ax = axes[2]
        line = ax.plot(self.T_all_sec_log, self.plot_frame)[0]
        line.set_data(self.T_all_sec_log[0], self.plot_frame[0])
        self.lines0 = [line]

        ax.set_ylim(0, np.max(self.plot_frame)*1.04)
        ax.set_xlabel('log2(T) [sec]', **self.label_kw)
        ax.set_ylabel('reldist', **self.label_kw)

        # finalize #######################################################
        fig.subplots_adjust(left=.05, right=.99, bottom=-.49, top=.96,
                            wspace=.02, hspace=.7)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        def _txt(txt_idx, T_sec):
            fill = (r"x(t)" if txt_idx == 0 else
                    r"x(t - \tau(t)")
            if frame_idx == 0:
                txt = r"$S_1(%s)$ | unaveraged" % fill
            else:
                txt = r"$S_1(%s)\ |\ T=%.3f$ sec " % (fill, T_sec)
            return txt

        T_sec = self.T_all[frame_idx] / N
        # imshow0 ############################################################
        self.ims0[0].set_array(self.imshow_frames0[frame_idx])
        self.txt0.set_text(_txt(0, T_sec))

        # imshow1 ############################################################
        self.ims1[0].set_array(self.imshow_frames1[frame_idx])
        self.txt1.set_text(_txt(1, T_sec))

        # plot ###############################################################
        self.lines0[0].set_data(self.T_all_sec_log[:frame_idx + 1],
                                self.plot_frame[:frame_idx + 1])

        # finalize ###########################################################
        self._drawn_artists = [*self.ims0, self.txt0, *self.ims1, self.txt1,
                               *self.lines0]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

imshow_frames0, imshow_frames1 = Scx_all0, Scx_all1
plot_frame = dists
vmax = max(Scx_all0.max(), Scx_all1.max())*.98

ani = PlotImshowAnimation(imshow_frames0, imshow_frames1, plot_frame, T_all,
                          vmax)
ani.save('warp_scat_T.mp4', fps=15, savefig_kwargs=dict(pad_inches=0))
plt.show()
