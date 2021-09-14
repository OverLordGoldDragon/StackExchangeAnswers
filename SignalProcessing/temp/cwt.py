# -*- coding: utf-8 -*-
"""Continuous Wavelet Transform GIF."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fft, ifft
from kymatio.numpy import Scattering1D
from kymatio.toolkit import echirp

#%%## Set params & create scattering object ##################################
N = 512
J, Q = 4, 3
# make CWT
average, oversampling = False, 999
ts = Scattering1D(shape=N, J=J, Q=Q, average=average, oversampling=oversampling,
                  out_type='list', max_order=1)
ts.psi1_f.pop(-1)  # drop last for niceness
meta = ts.meta()

#%%# Create signal & warp it #################################################
x = echirp(N, fmin=16, fmax=N/2.0)
t = np.linspace(0, 1, N, 1)

Scx0 = ts(x)
Scx = np.vstack([c['coef'] for c in Scx0])[1:]
freqs = N * meta['xi'][meta['order'] == 1][:, 0]

#%% Animate ##################################################################
class CWTAnimation(animation.TimedAnimation):
    @staticmethod
    def _compute_bandwidth(p, criterion_amplitude=1e-3):
        # move peak to origin
        mx_idx = np.argmax(p)
        p = np.roll(p, -mx_idx)

        # index of first point to drop below `criterion_amplitude`
        decay_idx = np.where(p < p.max() * criterion_amplitude)[0][0]
        return 2 * decay_idx / len(p)

    def __init__(self, ts, x, t, freqs, stride=1, alpha='6f'):
        self.ts = ts
        self.x = x
        self.t = t
        self.freqs = freqs
        self.stride = stride
        self.N = len(x)
        assert self.N % 2 == 0

        # unpack filters
        self.psi_fs = [p[0] for p in ts.psi1_f]
        self.phi_f = ts.phi_f
        # conjugate since conv == cross-correlation with conjugate
        psi_ts = np.array([np.conj(ifft(p)) for p in self.psi_fs])
        self.psi_ts = psi_ts / psi_ts.real.max()

        # compute filter params
        self.js = [p['j'] for p in ts.psi1_f]
        self.lens = [int(1.2*p['support'][0]) for p in ts.psi1_f]
        self.bandwidths = [self._compute_bandwidth(p) for p in self.psi_fs]
        self.n_psis = len(self.psi_fs)
        self.Np = len(self.psi_fs[0])
        self.trim = self.Np//4
        # padded x to do CWT with
        self.xpf = fft(np.pad(x, [ts.pad_left, ts.pad_right], mode='reflect'))

        # take CWT without modulus
        self.Wx = np.array([ifft(p * self.xpf)[ts.ind_start[0]:ts.ind_end[0]]
                            for p in self.psi_fs])
        self.Ux = np.abs(self.Wx)

        self.alpha = '6f'
        self.title_kw = dict(weight='bold', fontsize=18, loc='left')
        self.label_kw = dict(weight='bold', fontsize=15, labelpad=3)
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left")
        self._prev_psi_idx = -1
        self._edge_frames = 0
        self._edge_done = False

        fig, axes = plt.subplots(2, 2, figsize=(18/1.5, 15/1.5))

        # convolution ########################################################
        ax = axes[0, 0]
        self.lines00 = []

        ax.plot(t, x, color='#000000' + self.alpha)
        self.lines00.append(ax.lines[-1])

        psi = self.psi_ts[0][self.trim:-self.trim]
        ax.plot(self.t, psi.real, color='tab:blue')
        self.lines00.append(ax.lines[-1])
        ax.plot(self.t, psi.imag, color='tab:orange')
        self.lines00.append(ax.lines[-1])

        dT = t.max() - t.min()
        xmin, xmax = t.min() - dT/50, t.max() + dT/50
        ax.set_xlim(xmin, xmax)
        self.txt00 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=17)

        # conv output ########################################################
        ax = axes[0, 1]
        self.lines01 = []

        mx = self.Ux.max()
        _range = np.linspace(-mx, mx, self.N)
        ax.plot(self.t, _range, color='tab:blue')
        self.lines01.append(ax.lines[-1])
        ax.plot(self.t, _range, color='tab:orange')
        self.lines01.append(ax.lines[-1])
        ax.plot(self.t, _range, color='black', linestyle='--')
        self.lines01.append(ax.lines[-1])
        for l in self.lines01:
            l.set_data([], [])

        ax.set_xlim(xmin, xmax)
        ax.set_ylabel(r"correlation", **self.label_kw)
        self.txt01 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=17)

        # filterbank #########################################################
        ax = axes[1, 0]
        zoom = -1
        # define colors & linestyles
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2']
        self.colors = [self.colors[0]]*10

        # determine plot parameters ######################################
        # vertical lines (octave bounds)
        # x-axis zoom
        Nmax = self.Np
        if zoom == -1:
            xlims = (-.02 * Nmax, 1.02 * Nmax)
        else:
            xlims = (-.01 * Nmax / 2**zoom, .55 * Nmax / 2**zoom)

        # plot filterbank ################################################
        self.lines10 = []
        for i, p in enumerate(self.psi_fs):
            j = self.js[i]
            ax.plot(p, color=self.colors[j] + self.alpha)
            self.lines10.append(ax.lines[-1])

        # axes styling
        ax.set_xlim(xlims)
        ax.axvline(Nmax/2, color='k', linewidth=1)
        xticks = np.linspace(0, Nmax, 9, endpoint=True)[:-1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, .125, .25, .375, .5, -.375, -.25, -.125])

        self.txt10 = ax.text(transform=ax.transAxes, **self.txt_kw,
                             fontsize=15, weight='bold')
        ax.set_xlabel("frequency (frac of fs)", **self.label_kw)

        # |CWT| heatmap ##################################################
        ax = axes[1, 1]
        im = ax.imshow(self.Ux, cmap='turbo', animated=True, aspect='auto',
                       interpolation='none')
        self.Ux_now = 0 * self.Ux
        im.set_array(self.Ux_now)
        self.ims11 = [im]

        ax.set_xticks(np.linspace(0, self.N, 6, endpoint=True))
        xticklabels = np.linspace(t.min(), t.max(), 6, endpoint=True)
        ax.set_xticklabels(["%.1f" % xt for xt in xticklabels])
        yticks = np.linspace(0, self.n_psis - 1, 6, endpoint=True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["%.2f" % (self.freqs[int(yt)] / self.N)
                            for yt in yticks])

        ax.set_title(r"$|\psi \star x|$", **self.title_kw)
        ax.set_xlabel("time [sec]", **self.label_kw)
        ax.set_ylabel("frequency", **self.label_kw)

        # finalize #######################################################
        fig.subplots_adjust(left=.05, right=.99, bottom=.05, top=.97,
                            wspace=.15, hspace=.13)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        frame_idx_adj = frame_idx - self._edge_frames
        step = self.stride * frame_idx_adj % self.N
        # at right bound
        if self.stride != 1 and (step < self.stride - 1 and frame_idx > 1
                                 and not self._edge_done):
            step = self.N
            self._edge_frames += 1
            self._edge_done = True
        else:
            self._edge_done = False
        frame_idx_adj = frame_idx - self._edge_frames
        total_step = self.stride * frame_idx_adj
        psi_idx = int(total_step / self.N)

        # convolution ########################################################
        offset = self.trim
        psi = np.roll(self.psi_ts[psi_idx], offset + step)
        psi = psi[self.trim:-self.trim]

        T = self.lens[psi_idx]
        start, end = max(0, step - T//2), min(self.N, step + T//2)
        t = self.t[start:end]
        psi = psi[start:end]

        self.lines00[1].set_data(t, psi.real)
        self.lines00[2].set_data(t, psi.imag)

        tau = "%.2f" % ((step / self.N) * (self.t.max() - self.t.min()))
        self.txt00.set_text(r"$\psi_{{{}}}(t - {}),\ x(t)$".format(psi_idx, tau))

        # conv output ########################################################
        out = self.Wx[psi_idx][:step]
        self.lines01[0].set_data(self.t[:step], out.real)
        self.lines01[1].set_data(self.t[:step], out.imag)
        self.lines01[2].set_data(self.t[:step], np.abs(out))

        self.txt01.set_text((r"$x \star \psi_{{{0}}},\ "
                             r"|x \star \psi_{{{0}}}|$").format(psi_idx))

        # filterbank #########################################################
        # highlight active filter
        if psi_idx != self._prev_psi_idx:
            j = self.js[psi_idx]
            self.lines10[psi_idx].set_color(self.colors[j])
            self.lines10[psi_idx].set_linewidth(2)
            self._prev_psi_idx = psi_idx

            # revert previous active filter to inactive
            if psi_idx != 0:
                j_prev = self.js[psi_idx - 1]
                self.lines10[psi_idx - 1].set_color(self.colors[j_prev] +
                                                    self.alpha)
                self.lines10[psi_idx - 1].set_linewidth(1)

        self.txt10.set_text(("psi_{} | center freq: {:.2f}, bandwidth: {:.2f}"
                             ).format(psi_idx, self.freqs[psi_idx] / self.N,
                                      self.bandwidths[psi_idx]))

        # |CWT| ##############################################################
        self.Ux_now[psi_idx, start:step] = self.Ux[psi_idx, start:step]
        self.ims11[0].set_array(self.Ux_now)

        # finalize ###########################################################
        self._prev_psi_idx = psi_idx
        self._drawn_artists = [*self.lines00, self.txt00, *self.lines01,
                               *self.lines10, *self.ims11]

    def new_frame_seq(self):
        return iter(range(self.n_psis * int(self.N // self.stride + 1)))

    def _init_draw(self):
        pass


ani = CWTAnimation(ts, x, t, freqs, stride=4)
ani.save('cwt.mp4', fps=60)
plt.show()
