# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87355/50076
# --- WORK IN PROGRESS ---

# USER CONFIGS
EXAMPLE_INDEX = 0
TOLERANCE = 0.1

import numpy as np
from numpy.fft import fft, ifft, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

from ssqueezepy import ssq_cwt, cwt, stft, Wavelet
from ssqueezepy.visuals import plot, plotscat, imshow
from ssqueezepy.experimental import scale_to_freq

from utils87355 import (
    wav_cfgs, sparse_mean, cc_2d1d,
    pad_input_make_t, make_unpad_shorthand,
    make_impulse_response, handle_wavelet,
    derivative_along_freq,
    find_audio_change_timestamps,
    data_names, data_dir, data_labels, load_data, vline_cfg,
)

#%%############################################################################
# Load data
# ---------
xo, fs, labels = load_data(EXAMPLE_INDEX)
x, t, pad_left, pad_right = pad_input_make_t(xo, fs)

N = len(x)

u = make_unpad_shorthand(pad_left, pad_right)

#%%############################################################################
# Quick inspection: CWT vs STFT
# -----------------------------

# Transform
wavelet = Wavelet('gmw')
Wx, scales = cwt(x, wavelet, padtype=None)
Sx = stft(x)[::-1]

freq_Wx = scale_to_freq(scales, wavelet, N, fs=fs)
freq_Sx = np.linspace(0, fs/2, len(Sx))[::-1]

# TODO escaling isnt actually square, account for exp of cwt

#%% Visualize
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ikw = dict(abs=1, fig=fig, show=0, xlabel="Time [sec]", xticks=u(t)[::10])
imshow(u(Wx)[:, ::10], **ikw, title="|CWT(x)|",  ax=axes[0],
       yticks=freq_Wx, ylabel="Frequency [Hz]")
imshow(u(Sx)[:, ::10], **ikw, title="|STFT(x)|", ax=axes[1], yticks=freq_Sx)
fig.subplots_adjust(wspace=.15)
plt.show()

#%%############################################################################
# Inspect region of interest
# --------------------------
reusables = handle_wavelet(
     wavelet='balanced', M=len(x), ssq_precfg=dict(scales='log', padtype=None),
     fmax_idx_frac=200/471, silence_interval_samples=int(fs*.2),
     escaling=(1, 1, 1),
 )
(wavelet, ssq_cfg, ir2df, pwidth, wsummerf, wsilencef, escale, fmax_idx
 ) = reusables

#%%
Txo, Wxo, *_ = ssq_cwt(x, **ssq_cfg)
Txoo = Txo.copy()

#%%
Txo = Txoo.copy()

#%%
def get_slc(t, center, interval):
    start, end = center - interval/2, center + interval/2
    t_start_idx = np.argmin(abs(t - start))
    t_end_idx = np.argmin(abs(t - end))
    slc = slice(t_start_idx, t_end_idx)
    return slc


def get_slc_predictions(t_slc, g, center, pwidth):
    peak = np.argmax(g)
    g = g.copy()
    g[peak - pwidth:peak + pwidth] = 0
    second_peak = np.argmax(g)

    one_peak_pred = peak
    two_peaks_pred = (peak + second_peak) // 2
    return one_peak_pred, two_peaks_pred


def viz_zoomed(fig, ax, subtitle, Tx, t, center, interval, cmap_scaling=.8,
               pwidth=None):
    slc = get_slc(t, center, interval)
    t_slc = t[slc]

    if Tx.ndim == 2:
        Tx_slc = Tx[:, slc]
        title = "{} -- zoomed around t={} sec".format(subtitle, center)
        imshow(Tx_slc, abs=1, xticks=t_slc, title=title, show=0, yticks=0,
               norm_scaling=cmap_scaling, fig=fig, ax=ax)
    else:
        assert pwidth is not None
        Tx_slc = Tx[slc]
        pred1, pred2 = get_slc_predictions(t_slc, Tx_slc, center, pwidth)

        center_in_slice = np.argmin(abs(t_slc - center))
        title = "Errors: {:.2f}%, {:.2f}% (one-peak, two-peak)".format(
            100 * abs(pred1 - center_in_slice) / t.size,
            100 * abs(pred2 - center_in_slice) / t.size
            )

        plot(t_slc, Tx_slc, show=0, fig=fig, ax=ax, yticks=0, title=title)
        ax.axvline(center, **vline_cfg)
    # # plant vertical line
    # vline_idx = np.argmin(abs(t - center)) - t_start_idx
    # vline_cfg = {'color': 'red', 'linewidth': 1, 'linestyle': '--'}
    # plot([], vlines=(vline_idx, vline_cfg), show=1)

def make_subplots():
    return plt.subplots(1, 2, figsize=(16, 6))

def run_viz(Tx, t, centers, intervals, subtitle="", cmap_scaling=.8, pwidth=None):
    cms = cmap_scaling
    pw = pwidth
    if not isinstance(intervals, tuple):
        intervals = (intervals,)

    for interval in intervals:
        fig, axes = make_subplots()
        viz_zoomed(fig, axes[0], subtitle, Tx, t, centers[0], interval, cms, pw)
        viz_zoomed(fig, axes[1], subtitle, Tx, t, centers[1], interval, cms, pw)
        fig.subplots_adjust(wspace=.07)
        plt.show()

# also drop uninteresting frequencies (determined manually here)
Tx_slc = abs(Txo[:fmax_idx])
escale = (np.linspace(1, 0, len(Tx_slc))**2)[:, None]
# Tx_slc *= escale


# Tx_slc = g_carvedo.copy()
# Tx_slc = abs(Txo2[:fmax_idx])
# Tx_slc = abs(Wxo2[:fmax_idx])
# Tx_slc = Tsx.copy()
# Tx_slc = Sx2.copy()

# take derivative
Tx_slc_d = derivative_along_freq(Tx_slc)
# TODO deriv along tm for stft, `<` thresholding

centers = labels
# centers = (3.5, 4.5)#, 4.50)#, 4.5)
# centers = (0.5, 1.5)
intervals = (1.5, 0.5)
run_viz(Tx_slc, t, centers, intervals, "|SSQ_CWT|")

#%%
run_viz(Tx_slc_d, t, centers, intervals[1], "diff(|SSQ_CWT|)")

#%%
OPTION = 0

# use shorthand
g = Tx_slc.copy()
gd = Tx_slc_d

# do carving
carve_th_scaling = 1/10
th = sparse_mean(gd) * carve_th_scaling
g[gd > th] = 0
# g *= escale
# g = ifft(fft(g, axis=-1) * ir2df[:len(g)]).real

# store for debug convenience
g_carved = g.copy()
# compute for inspection
Txc = Txo[:fmax_idx]#.copy()
Txc[gd > th] = 0
# xc = Txc.sum(axis=0).real
# Txc = abs(ssq_cwt(xc, **ssq_cfg)[0])
# g = Txc

run_viz(g, t, centers, intervals[1], "Tx_carved", cmap_scaling=.1)

# there was other play code here to auto compute `pwidth`, I forgot
pwidth = 1400#
w = np.zeros(g.shape[-1])
w[:pwidth//2] = 1
w[-(pwidth//2 - 1):] = 1
wsummerf = fft(w)

if OPTION == 0:
    g = g**2
    g = g.sum(axis=0)
else:
    if OPTION == 2:
        g *= escale
    g = cc_2d1d(g, ir2df)[0]
    g = g**2
g = ifft(wsummerf * fft(g)).real

run_viz(g, t, centers, intervals[1], pwidth=pwidth)
plot(u(t), u(g), vlines=(list(centers), vline_cfg),
     title="sliding_sum(sum(Tx_carved**2, axis=0)) -- The Feature Vector")
