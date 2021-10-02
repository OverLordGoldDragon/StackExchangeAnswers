# -*- coding: utf-8 -*-
"""Continuous Wavelet Transform w/ lowpassing (first-order scattering) GIF."""
# https://dsp.stackexchange.com/q/78512/50076 ################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fft, ifft, fftshift, ifftshift
from kymatio.numpy import Scattering1D
from kymatio.toolkit import echirp

#%%## Set params & create scattering object ##################################
N = 512
J, Q = 4, 3
T = 2**(J + 1)
# make CWT
average, oversampling = False, 999
ts = Scattering1D(shape=N, J=J, Q=Q, average=average, oversampling=oversampling,
                  out_type='list', max_order=1, T=T, max_pad_factor=None)
ts.psi1_f.pop(-1)  # drop last for niceness

#%%# Create signal & warp it #################################################
x = echirp(N, fmin=16, fmax=N/2.0)
t = np.linspace(0, 1, N, 1)

meta = ts.meta()
freqs = N * meta['xi'][meta['order'] == 1][:, 0]

#%%# Animate #################################################################
class CWTAnimation(animation.TimedAnimation):
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
        self.phi_f = ts.phi_f[0]
        self.phi_t = ifft(self.phi_f).real
        # conjugate since conv == cross-correlation with conjugate
        psi_ts = np.array([np.conj(ifft(p)) for p in self.psi_fs])
        self.psi_ts = psi_ts / psi_ts.real.max()

        # compute filter params
        self.js = [p['j'] for p in ts.psi1_f]
        self.lens = [int(1.2*p['support'][0]) for p in ts.psi1_f]
        self.n_psis = len(self.psi_fs)
        self.Np = len(self.psi_fs[0])
        self.trim = ts.ind_start[0]
        self.phi_t_trim = fftshift(ifftshift(self.phi_t)[self.trim:-self.trim])
        # padded x to do CWT with
        self.xpf = fft(np.pad(x, [ts.pad_left, ts.pad_right], mode='reflect'))

        # take CWT without modulus
        self.Wx = np.array([ifft(p * self.xpf) for p in self.psi_fs])
        self.Ux = np.abs(self.Wx)
        self.Sx = ifft(fft(self.Ux) * self.phi_f).real
        start, end = ts.ind_start[0], ts.ind_end[0]
        self.Sx = self.Sx[:, start:end]
        self.Ux = self.Ux[:, start:end]
        # suppress boundary effects for visuals
        mx = self.Sx.max()
        mx_avg = self.Sx[5].max()
        self.Sx[0]  *= (mx_avg / mx) * 1.3
        self.Sx[-1] *= (mx_avg / mx) * 1.3
        # spike causes graphical glitch in converting to gif -- flatten
        mx = self.Ux.max() * 1.02
        self.Ux[0] *= .58 / .61
        self.Ux[0, -1] = mx

        self.alpha = '6f'
        self.title_kw = dict(weight='bold', fontsize=16, loc='left')
        self.label_kw = dict(weight='bold', fontsize=14, labelpad=3)
        self.txt_kw = dict(x=0, y=1.015, s="", ha="left")
        self._prev_psi_idx = -1

        fig, axes = plt.subplots(1, 2, figsize=(18/1.7, 8/1.7))

        # |CWT|*phi_f ####################################################
        ax = axes[0]
        self.lines0 = []

        ax.imshow(self.Ux, cmap='turbo', aspect='auto', interpolation='none')
        ax.set_xticks(np.linspace(0, self.N, 6, endpoint=True))
        xticklabels = np.linspace(t.min(), t.max(), 6, endpoint=True)
        ax.set_xticklabels(["%.1f" % xt for xt in xticklabels])
        yticks = np.linspace(0, self.n_psis - 1, 6, endpoint=True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["%.2f" % (self.freqs[int(yt)] / self.N)
                            for yt in yticks])

        phi_t = 6 - 600*self.phi_t_trim
        ax.plot(np.arange(self.N), phi_t, color='white', linewidth=2)
        self.lines0.append(ax.lines[-1])

        self.txt0 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        # output #########################################################
        ax = axes[1]
        im = ax.imshow(self.Sx, cmap='turbo', animated=True, aspect='auto',
                       interpolation='none')
        self.Sx_now = 0 * self.Sx
        im.set_array(self.Sx_now)
        self.ims1 = [im]

        ax.set_xticks(np.linspace(0, self.N, 6, endpoint=True))
        xticklabels = np.linspace(t.min(), t.max(), 6, endpoint=True)
        ax.set_xticklabels(["%.1f" % xt for xt in xticklabels])
        yticks = np.linspace(0, self.n_psis - 1, 6, endpoint=True)
        ax.set_yticks(yticks)
        ax.set_yticklabels(["%.2f" % (self.freqs[int(yt)] / self.N)
                            for yt in yticks])
        ax.set_title(r"$|\psi \star x| \star \phi$", **self.title_kw)

        # finalize #######################################################
        fig.subplots_adjust(left=.05, right=.99, bottom=.06, top=.94,
                            wspace=.15)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        step = self.stride * frame_idx % self.N
        total_step = self.stride * frame_idx
        psi_idx = int(total_step / self.N)
        # at right bound
        if self.stride != 1 and (step < self.stride - 1 and frame_idx > 1):
            step = self.N
            psi_idx -= 1

        # |CWT|*phi_f ####################################################
        T = self.ts.phi_f['support']
        start, end = max(0, step - T//2), min(self.N, step + T//2)
        phi_t = 6 - 600*self.phi_t_trim
        phi_t = np.roll(phi_t, step)[start:end]
        self.lines0[0].set_data(np.arange(start, end), phi_t)

        tau = "%.2f" % ((step / self.N) * (self.t.max() - self.t.min()))
        self.txt0.set_text(r"$\phi(t - {}),\ |\psi \star x|$".format(tau))

        # output #########################################################
        self.Sx_now[:, start:step] = self.Sx[:, start:step]
        self.ims1[0].set_array(self.Sx_now)

        # finalize ###########################################################
        self._prev_psi_idx = psi_idx
        self._drawn_artists = [*self.lines0, self.txt0, *self.ims1]

    def new_frame_seq(self):
        return iter(range(1 * int(self.N // self.stride + 1)))

    def _init_draw(self):
        pass


ani = CWTAnimation(ts, x, t, freqs, stride=2)
ani.save('cwt_phi.mp4', fps=60)
plt.show()
