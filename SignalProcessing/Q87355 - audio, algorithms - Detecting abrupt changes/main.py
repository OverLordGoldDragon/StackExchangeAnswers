# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87355/50076
# Will not reproduce the stackexchange post completely, lots to code

# USER CONFIGS
EXAMPLE_INDEX = 0
TOLERANCE = 0.1

import numpy as np
from numpy.fft import fft, ifft, ifftshift, fft2, ifft2
import matplotlib.pyplot as plt

from ssqueezepy import ssq_cwt, cwt, stft, Wavelet, ssq_stft
from ssqueezepy.visuals import plot, plotscat, imshow
from ssqueezepy.experimental import scale_to_freq

from utils87355 import (
    wav_cfgs, sparse_mean, cc2d1d,
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

M = len(x)

u = make_unpad_shorthand(pad_left, pad_right)

#%%############################################################################
# Quick inspection: CWT vs STFT
# -----------------------------

# Transform
wavelet = Wavelet('gmw')
Wx, scales = cwt(x, wavelet, padtype=None)
Sx = stft(x)[::-1]

freq_Wx = scale_to_freq(scales, wavelet, M, fs=fs)
freq_Sx = np.linspace(0, fs/2, len(Sx))[::-1]

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
reusables_ckw = dict(
    wavelet='balanced',
    fs=fs,
    ssq_precfg=dict(scales='log', padtype=None),
    silence_interval_samples=int(fs*.2),
    # to reproduce some of the post, in early "amplify discriminant" sections,
    # this should be set to `(0, 1, 1)`
    escaling=(1, 1, 1),
)
reusables = handle_wavelet(M=M, fmax_idx_frac=200/471, **reusables_ckw)
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
               pwidth=None, yticks=None):
    slc = get_slc(t, center, interval)
    t_slc = t[slc]

    if Tx.ndim == 2:
        if yticks is None:
            yticks = 0
            ylabel = None
        else:
            ylabel = "Frequencies [Hz]"
        Tx_slc = Tx[:, slc]
        title = "{} -- zoomed around t={} sec".format(subtitle, center)
        imshow(Tx_slc, abs=1, xticks=t_slc, title=title, show=0, yticks=yticks,
               norm_scaling=cmap_scaling, fig=fig, ax=ax, ylabel=ylabel)
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

def run_viz(Tx, t, centers, intervals, subtitle="", cmap_scaling=.8, pwidth=None,
            yticks=None):
    cms = (cmap_scaling if isinstance(cmap_scaling, (list, tuple)) else
           (cmap_scaling, cmap_scaling))
    pw = pwidth
    if not isinstance(intervals, tuple):
        intervals = (intervals,)

    for interval in intervals:
        fig, axes = make_subplots()
        viz_zoomed(fig, axes[0], subtitle, Tx, t, centers[0], interval,
                   cms[0], pw, yticks)
        viz_zoomed(fig, axes[1], subtitle, Tx, t, centers[1], interval,
                   cms[1], pw)
        fig.subplots_adjust(wspace=.07)
        plt.show()

# also drop uninteresting frequencies (determined manually here)
Tx_slc = abs(Txo[:fmax_idx])

# take derivative
Tx_slc_d = derivative_along_freq(Tx_slc)

centers = labels
# centers = (0.5, 1.5)
intervals = (1.5, 0.5)
run_viz(Tx_slc, t, centers, intervals, "|SSQ_CWT|")

#%%
run_viz(Tx_slc_d, t, centers, intervals[1], "diff(|SSQ_CWT|)")

#%%
OPTION = 1

# use shorthand
g = Tx_slc.copy()
gd = Tx_slc_d

# do carving
carve_th_scaling = 1/3
th = sparse_mean(gd) * carve_th_scaling
g[gd > th] = 0
# g *= escale
# g = ifft(fft(g, axis=-1) * ir2df[:len(g)]).real

# store for debug convenience
g_carved = g.copy()
# compute for inspection
Txc = Txo[:fmax_idx]#.copy()
Txc[gd > th] = 0

run_viz(g, t, centers, intervals[1], "Tx_carved", cmap_scaling=.1)

# there was other play code here to auto compute `pwidth`, I forgot
pwidth = 1400
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
    g = cc2d1d(g, ir2df)
    g = g**2
g = ifft(wsummerf * fft(g)).real

run_viz(g, t, centers, intervals[1], pwidth=pwidth)
plot(u(t), u(g), vlines=(list(centers), vline_cfg),
     title="sliding_sum(sum(Tx_carved**2, axis=0)) -- The Feature Vector")

#%%###########################################################################
# Amplify discriminative features (pt. 3)
# ---------------------------------------
Txo, Wxo, *_ = ssq_cwt(x, **ssq_cfg)
freqs = scale_to_freq(ssq_cfg['scales'], ssq_cfg['wavelet'], len(x), fs=fs,
                      padtype=None)
Tx, Wx = Txo[:400], Wxo[:400]
escale_ = np.logspace(np.log10(1), np.log10(np.sqrt(.1)), len(Tx))[:, None]**2

#%%
# escale_ = freqs[:, None]**2
to_viz = [(Wx, Tx), (Wx*escale_, Tx*escale_)]
norm_scalings = [(.9, .6), (.4, .3)]
for [(a, b), ns] in zip(to_viz, norm_scalings):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    kw = dict(abs=1, fig=fig, show=0, xticks=u(t)[::10], xlabel="Time [sec]")
    imshow(u(a)[:,::10], **kw, ax=axes[0], norm_scaling=ns[0],
           yticks=freqs[:400], ylabel="Frequency [Hz]")
    imshow(u(b)[:,::10], **kw, ax=axes[1], norm_scaling=ns[1], yticks=0)
    plt.show()

#%%###########################################################################
# Attenuate invariants (2): eliminate template confounding
# --------------------------------------------------------
x, fs, labels = load_data(3)
xp, tp, pad_left, pad_right = pad_input_make_t(x, fs=fs)
u = make_unpad_shorthand(pad_left, pad_right)
reusables = handle_wavelet(
    M=len(xp),
    fmax_idx_frac=400/471,
    **reusables_ckw,
)
(wavelet, ssq_cfg, ir2df, pwidth, wsummerf, wsilencef, escale, fmax_idx
 ) = reusables

#%%
Txo, Wxo, *_ = ssq_cwt(xp, **ssq_cfg)
Tx_slco = Txo[:fmax_idx]
freqs_cwt = scale_to_freq(ssq_cfg['scales'], ssq_cfg['wavelet'], len(xp),
                          fs=fs, padtype=None)

Tx_slc = Tx_slco.copy()
for _ in range(2):
    Tx_slc_d = derivative_along_freq(Tx_slc)
    th = sparse_mean(gd) * carve_th_scaling
    Tx_slc[Tx_slc_d > th] = 0
Tx_carved = Tx_slc

#%%
centers = (0.7, 1.42)
escale_ = np.logspace(np.log10(1), np.log10(np.sqrt(.1)), len(Tx_slco)
                      )[:, None]**2
ikw = dict(centers=centers, intervals=intervals[1], t=tp)
run_viz(Tx_slco*escale_, **ikw, subtitle="|SSQ_CWT|", cmap_scaling=.2,
        yticks=freqs_cwt[:fmax_idx])
run_viz(Tx_carved*escale_, **ikw, subtitle="Tx_carved", cmap_scaling=.2)

#%%
Tsx = ssq_stft(xp, 'hamm', flipud=True)[0]
escale_lin = np.linspace(1, 0, len(Tsx))[:, None]
freqs = np.linspace(1, 0, len(Tsx)) * fs/2

slc0 = get_slc(tp, centers[0], intervals[1])
slc1 = get_slc(tp, centers[1], intervals[1])
title0 = "|SSQ_STFT(x)| -- zoomed around t={:.2g} sec".format(centers[0])
title1 = "|SSQ_STFT(x)| -- zoomed around t={:.2g} sec".format(centers[1])

#%%
run_viz(Tsx*escale_lin, **ikw, subtitle="|SSQ_STFT|", cmap_scaling=(.4, .2),
        yticks=freqs)
#%%
aTsx = abs(Tsx * escale_lin)**2
aTsx = aTsx[:int(.75*len(Tsx))]

stft_carve_th = sparse_mean(aTsx)
Tsx[:len(aTsx)][aTsx > stft_carve_th] = 0
xfilt = Tsx.sum(axis=0).real

xp = xfilt

#%%
Txo, Wxo, *_ = ssq_cwt(xp, **ssq_cfg)
Tx_slco = Txo[:fmax_idx]

Tx_slc = Tx_slco.copy()
for _ in range(2):
    Tx_slc_d = derivative_along_freq(Tx_slc)
    th = sparse_mean(gd) * carve_th_scaling
    Tx_slc[Tx_slc_d > th] = 0
Tx_carved = Tx_slc

run_viz(Tx_slco*escale_, **ikw, subtitle="|SSQ_CWT|", cmap_scaling=.2,
        yticks=freqs_cwt[:fmax_idx])
# set cmap_scaling such that impulse feature brightness compares with previous
run_viz(Tx_carved*escale_, **ikw, subtitle="Tx_carved", cmap_scaling=.35)

#%%
# plot feature vector straight from `utils`, too much code to reproduce here
