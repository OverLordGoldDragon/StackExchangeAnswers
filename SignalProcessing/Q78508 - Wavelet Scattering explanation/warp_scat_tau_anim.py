# -*- coding: utf-8 -*-
"""Scattering unwarped vs warped echirp GIF, sweeping `tau`."""
# https://dsp.stackexchange.com/q/78508/50076 ################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from kymatio.torch import Scattering1D
from kymatio.visuals import plot, plotscat, imshow
from kymatio.toolkit import _echirp_fn, l2

def tau(t, K=.08):
    return np.cos(2*np.pi * t**2) * K

def adtau(t, _tau):
    return np.abs(np.diff(_tau) / np.diff(t))

import torch
USE_GPU = bool('cuda' if torch.cuda.is_available() else 'cpu')

#%%###########################################################################
N = 2048
J, Q = 8, 8
T = 256

kw = dict(shape=N, J=J, Q=Q, T=T, oversampling=999, out_type='list',
          r_psi=.82, max_pad_factor=None)
ts0 = Scattering1D(**kw, max_order=1, average=False)
ts1 = Scattering1D(**kw, average=True)
if USE_GPU:
    ts0.cuda(); ts1.cuda()

# compute quality factor
p = ts0.psi1_f[0]
QF = p['xi'] / p['sigma']

#%%# helper methods ##########################################################
def make_tau(N, n_pts=64, warps=1, shifts=1, K_init=.01):
    t = np.linspace(0, 1, N, 0)

    adtau_init_max = adtau(t, tau(t, K_init)).max()
    K_min = (1 / adtau_init_max) * K_init * (1/10 * (1/QF))
    K_max = (1 / adtau_init_max) * K_init #* (10   * (1/QF))

    K_all = np.logspace(np.log10(K_min), np.log10(K_max), n_pts - 1, 1)
    K_all = np.hstack([0, K_all])
    tau_all = np.vstack([tau(t, K=k) for k in K_all])
    if not warps:
        tau_all *= 0

    if shifts:
        tau_all += np.logspace(np.log10(1/N), np.log10(.25), len(tau_all)
                               )[:, None]
    adtau_all = adtau(t, tau_all)
    adtau_max_all = adtau_all.max(axis=-1)

    assert adtau_max_all.max() <= 1, adtau_max_all.max()
    return t, tau_all, adtau_all, adtau_max_all

def get_dists(Scx_all0, Scx_all1, adtau_max_all, viz=0):
    adtau_max_all_log = np.log10(adtau_max_all[1:])
    dists, dists_log = {}, {}
    refs = {'CWT': Scx_all0[0], 'Sx': Scx_all1[0]}

    for k, S_all in zip(list(refs), (Scx_all0, Scx_all1)):
        ref = refs[k]
        dists[k] = np.array([l2(ref, S) for S in S_all]).squeeze()
        dists_log[k] = np.log10(dists[k][1:])
        if viz:
            plotscat(adtau_max_all_log, dists_log[k], show=0, auto_xlims=0)
    if viz:
        plot([], show=1)
    return dists, dists_log, adtau_max_all_log

def scatter(x_all, viz=0):
    def sc(x, ts):
        return np.array([c['coef'].cpu().numpy() for c in ts(x)])

    Scx_all0 = np.array([sc(x, ts0) for x in x_all])[:, 1:]
    Scx_all1 = np.array([sc(x, ts1) for x in x_all])
    if viz:
        imshow(Scx_all0[0],  abs=1)
        imshow(Scx_all0[40], abs=1)
    return Scx_all0, Scx_all1

#%% Animate #################################################################
class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, imshow_frames0, imshow_frames1, plot_frame0, plot_frame1,
                 adtau_max_all, adtau_max_all_log, T, vmax0, vmax1, uses_shift):
        self.imshow_frames0 = imshow_frames0
        self.imshow_frames1 = imshow_frames1
        self.plot_frame0 = plot_frame0
        self.plot_frame1 = plot_frame1
        self.adtau_max_all = adtau_max_all
        self.adtau_max_all_log = adtau_max_all_log
        self.T = T
        self.vmax0 = vmax0
        self.vmax1 = vmax1
        self.uses_shift = uses_shift

        self.N = imshow_frames0[0].shape[-1]
        self.T_sec = T / self.N
        self.adtau_max_all_sec = self.adtau_max_all / self.N
        self.adtau_max_all_sec_log = np.log10(10**adtau_max_all_log / self.N)

        self.n_repeats_first_total = n_repeats * repeat_first
        self.n_repeats_last_total = n_repeats * repeat_last
        self.n_repeats_total = (self.n_repeats_first_total +
                                self.n_repeats_last_total)
        self.n_data_pts = max(len(imshow_frames0), len(imshow_frames1))
        self.n_frames = self.n_data_pts + self.n_repeats_total

        self.label_kw = dict(weight='bold', fontsize=17, labelpad=3)
        self.txt_kw = dict(x=0, y=1.017, s="", ha="left")

        fig = plt.figure(figsize=(18/1.5, 15/1.9))
        ax0 = fig.add_subplot(2, 2, 1)
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(3, 2, 3)
        ax3 = fig.add_subplot(3, 2, 4)
        axes = np.array([[ax0, ax1], [ax2, ax3]])

        # imshow1 ############################################################
        ax = axes[0, 0]
        imshow_kw = dict(cmap='turbo', aspect='auto', animated=True, vmin=0)
        im = ax.imshow(self.imshow_frames0[0], **imshow_kw, vmax=self.vmax0)
        self.ims0 = [im]

        txt0 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        txt0.set_text(r"CWT$x(t - \tau(t))$")
        ax.set_xticks([]); ax.set_yticks([])

        # imshow2 ############################################################
        ax = axes[0, 1]
        im = ax.imshow(self.imshow_frames1[0], **imshow_kw, vmax=self.vmax1)
        self.ims1 = [im]

        txt1 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        txt1.set_text(r"$S_1x(t - \tau(t)) |\ T=%.3f$ sec" % self.T_sec)
        ax.set_xticks([]); ax.set_yticks([])

        # plot1 ###############################################################
        def add_plot(xdata, ydata, ax, name):
            line = ax.plot(xdata, ydata)[0]
            line.set_data(xdata[0], ydata[0])
            if not hasattr(self, name):
                setattr(self, name, [])
            getattr(self, name).append(line)

            # xlims
            at = np.abs(xdata)
            diff = at.max() - at.min()
            ax.set_xlim(xdata.min() - .04*diff, xdata.max() + .04*diff)

        ax = axes[1, 0]
        xdata = self.adtau_max_all_sec_log
        ydata0 = self.plot_frame0['CWT']
        ydata1 = self.plot_frame0['Sx']
        add_plot(xdata, ydata0, ax, 'lines0')
        add_plot(xdata, ydata1, ax, 'lines0')

        # text
        nm = (r"\tau(t)" if self.uses_shift else
              r"\max|\tau'(t)|")
        ax.set_xlabel(rf"$\log_{{10}}{nm}$", **self.label_kw)
        self.txt2 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        self.txt2.set_text(rf"$\log_{{10}}(reldist)\ |\ "
                           rf"\log_{{10}}{nm} = -\infty$")

        # plot2 ###############################################################
        ax = axes[1, 1]
        xdata = self.adtau_max_all_sec
        ydata0 = self.plot_frame1['CWT']
        ydata1 = self.plot_frame1['Sx']
        add_plot(xdata, ydata0, ax, 'lines1')
        add_plot(xdata, ydata1, ax, 'lines1')

        # text
        ax.set_xlabel(rf"${nm}$", **self.label_kw)
        self.txt3 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=18)
        self.txt3.set_text(rf"$reldist\ |\ {nm} = 0.000$")

        # finalize #######################################################
        fig.subplots_adjust(left=.05, right=.99, bottom=-.49, top=.96,
                            wspace=.08, hspace=.8)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        # handle repeats #####################################################
        n_repeats_offset = (0 if frame_idx < self.n_repeats_first_total else
                            self.n_frames - self.n_repeats_last_total)
        if frame_idx == 0:
            self.data_idx = 0
            self.prev_data_idx = 0
        elif frame_idx % (n_repeats + n_repeats_offset) == 0 or (
                self.n_repeats_first_total < frame_idx <
                (self.n_frames - self.n_repeats_last_total)):
            self.data_idx += 1

        if (frame_idx == self.n_frames - 1 and
            self.data_idx + 1 < self.n_data_pts):
            print("Could not finish animation: %s / %s data frames animated" % (
                self.data_idx + 1, len(self.imshow_frames0)))

        if self.data_idx == self.prev_data_idx:
            return
        self.prev_data_idx = self.data_idx
        frame_idx = self.data_idx  # adjusted

        # imshow1 ############################################################
        self.ims0[0].set_array(self.imshow_frames0[frame_idx])

        # imshow2 ############################################################
        self.ims1[0].set_array(self.imshow_frames1[frame_idx])

        # plot1 ###############################################################
        xdata = self.adtau_max_all_sec_log[:frame_idx+1]
        ydata0 = self.plot_frame0['CWT']  [:frame_idx+1]
        ydata1 = self.plot_frame0['Sx']   [:frame_idx+1]
        self.lines0[0].set_data(xdata, ydata0)
        self.lines0[1].set_data(xdata, ydata1)

        nm = (r"\tau(t)" if self.uses_shift else
              r"\max|\tau'(t)|")
        self.txt2.set_text((r"$\log_{{10}}(reldist)\ |\ \log_{{10}}{} = {:.3f}$"
                            ).format(nm, xdata[-1]))

        # plot2 ###############################################################
        xdata = self.adtau_max_all_sec  [:frame_idx+1]
        ydata0 = self.plot_frame1['CWT'][:frame_idx+1]
        ydata1 = self.plot_frame1['Sx'] [:frame_idx+1]
        self.lines1[0].set_data(xdata, ydata0)
        self.lines1[1].set_data(xdata, ydata1)
        self.txt3.set_text((r"$reldist\ |\ {} = {:.3f}$"
                            ).format(nm, xdata[-1]))

        # finalize ###########################################################
        self._drawn_artists = [*self.ims0, *self.ims1, *self.lines0, self.txt2,
                               *self.lines1, self.txt3]

    def new_frame_seq(self):
        # return iter(range(2))
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

repeat_first = 0
repeat_last = 1
n_repeats = 8

#%%
def run(Scx_all0, Scx_all1, dists, dists_log, adtau_max_all, adtau_max_all_log,
        name):
    imshow_frames0, imshow_frames1 = Scx_all0, Scx_all1[:, 1:Scx_all0.shape[1]]
    plot_frame0, plot_frame1 = dists_log, dists
    vmax0 = Scx_all0.max() * .99
    vmax1 = Scx_all1.max() * .99

    ani = PlotImshowAnimation(imshow_frames0, imshow_frames1, plot_frame0,
                              plot_frame1, adtau_max_all, adtau_max_all_log,
                              T, vmax0, vmax1, uses_shift=(name=='pulse'))
    ani.save(f'warp_scat_tau_{name}.mp4', fps=8,
             savefig_kwargs=dict(pad_inches=0))
    plt.show()
    return ani

#%% echirp
t, tau_all, adtau_all, adtau_max_all = make_tau(N, shifts=0)
_t = (t - tau_all)

a0 = np.cos(2*np.pi * 3.7 * _t)
x_all = np.cos(_echirp_fn(fmin=40, fmax=N/8)(_t)) * a0
x_all /= np.linalg.norm(x_all, axis=-1)[:, None]

#%%
Scx_all0, Scx_all1 = scatter(x_all, viz=1)
dists, dists_log, adtau_max_all_log = get_dists(Scx_all0, Scx_all1,
                                                adtau_max_all, viz=1)
#%%
run(Scx_all0, Scx_all1, dists, dists_log, adtau_max_all, adtau_max_all_log,
    name='echirp')

#%% Nyquist pulse ############################################################
t, tau_all, adtau_all, adtau_max_all = make_tau(N, shifts=1, warps=0)
x = np.exp(-(t - .5)**2 * 2**20) * np.cos(2*np.pi*t*N/2)
shifts = np.unique(np.ceil(tau_all[:, 0] * N)).astype(int)
x_all = np.array([np.roll(x, s) for s in shifts])
x_all /= np.linalg.norm(x_all, axis=-1)[:, None]

#%%
Scx_all0, Scx_all1 = scatter(x_all, viz=1)
dists, dists_log, adtau_max_all_log = get_dists(Scx_all0, Scx_all1, shifts, viz=1)
#%%
run(Scx_all0, Scx_all1, dists, dists_log, shifts, adtau_max_all_log,
    name='pulse')
