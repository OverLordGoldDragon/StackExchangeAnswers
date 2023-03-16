# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/11333/50076
# Some code copied from my answer to https://dsp.stackexchange.com/q/85745/50076
import numpy as np
from scipy.signal.windows import dpss
from numpy.fft import fft, ifft, fftshift, ifftshift
from ssqueezepy import stft
from ssqueezepy.visuals import plot, plotscat

#%%############################################################################
# Helpers
# -------
def _pad_window(w, padded_len):
    pleft = (padded_len - len(w)) // 2
    pright = padded_len - pleft - len(w)
    return np.pad(w, [pleft, pright])

def get_adj_windows(window, n_fft, N):
    padded_len_conv = N + n_fft - 1
    # window_adj = np.pad(window, (padded_len_conv - len(window))//2)
    # window_adj_f = np.pad(window, (n_fft - len(window))//2)
    window_adj = _pad_window(window, padded_len_conv)
    window_adj_f = _pad_window(window, n_fft)

    # shortcut for later examples to spare code; ensure ifftshift center at idx 0
    def _center(w):
        w = ifftshift(w)
        w = fftshift(np.roll(w, np.argmax(w)))
        return w

    window_adj = _center(window_adj)
    window_adj_f = _center(window_adj_f)
    return window_adj, window_adj_f

#%%############################################################################
# Row-wise STFT implementation
# ----------------------------
def cisoid(N, f):
    t = np.linspace(0, 1, N, endpoint=False)
    return (np.cos(2*np.pi * f * t) +
            np.sin(2*np.pi * f * t) * 1j)


def stft_rowwise(x, window, n_fft):
    assert len(window) == n_fft and n_fft <= len(x)

    # compute some params
    xp = x
    N = len(x)
    padded_len = N

    # pad such that `armgax(window)` for `Sx[:, 0]` is at `Sx[:, 0]`
    # (DFT-center the convolution kernel)
    # note, due to how scipy generates windows and standard stft handles padding,
    # this still won't yield zero-phase for majority of signals, but it's both
    # fixable and can be very close; ideally just pass in `len(window)==len(x)`.
    _wpad_right = (padded_len - len(window)) / 2
    wpad_right = (int(np.floor(_wpad_right)) if n_fft % 2 == 1 else
                  int(np.ceil(_wpad_right)))
    wpad_left = padded_len - len(window) - wpad_right

    # generate filterbank
    cisoids = np.array([cisoid(n_fft, f) for f in range(n_fft//2 + 1)])
    fbank = ifftshift(window) * cisoids
    fbank = np.pad(fftshift(fbank), [[0, 0], [wpad_left, wpad_right]])
    fbank = ifftshift(fbank)
    fbank_f = fft(fbank, axis=-1).conj()

    # circular convolution
    prod = fbank_f * fft(xp)[None]
    Sx = ifft(prod, axis=-1)
    return Sx, fbank_f

#%%############################################################################
# Current answer -- messy code
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from ssqueezepy import ssq_cwt, TestSignals, Wavelet
from ssqueezepy.visuals import imshow

N = 256
n_fft = N
t = np.linspace(0, 1, N, 0)
x = np.cos(2*np.pi * 125 * t)
window = gaussian(N, 6, 0)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)

Sx, fbank_f = stft_rowwise(x, window_adj_f, n_fft)

plot(fbank_f[::4].T, abs=1, color='tab:blue', h=.7, w=1.1,
     vlines=(128, {'linewidth': 3, 'color': 'tab:red'}),
     title="STFT filterbank, frequency-domain | Gaussian", show=1)
#%%
for case in (0, 1):
    if case == 0:
        a = fbank_f[-8].copy()
    else:
        a = fbank_f[-1].copy()
    title = "Filter peak freq = {}".format(np.argmax(np.abs(a)))

    fig, axes = plt.subplots(4, 1, figsize=(8, 24))

    b = fft(x)
    a /= max(abs(a))
    b /= max(abs(b))
    pkw = dict(ylims=(-.01, 1.03), abs=1, fig=fig)
    plotscat(b, abs=1, color='k', ax=axes[0])
    plot(a, **pkw, title=title, ax=axes[0])
    plotscat(a * b, **pkw, ax=axes[1], title="fft(filter) * fft(x)")

    x_filt = ifft(a * b)
    plot(x_filt, complex=1, ax=axes[2], fig=fig,)
    plot(x_filt, abs=1, color='k', linestyle='--', ax=axes[2], fig=fig,
         title="x_filt = ifft(fft(filter) * fft(x))")

    plotscat(fft(abs(x_filt)), abs=1, ax=axes[3], fig=fig,
             title="fft(envelope) | envelope = abs(x_filt)")
    fig.subplots_adjust(hspace=.13)

#%%
ikw = dict(w=1.2, h=.35, yticks=np.arange(len(Sx))[::-1])
imshow(Sx[::-1], abs=1, **ikw, title="|STFT|")
imshow(Sx.real[::-1], **ikw, title="STFT.real")
imshow(Sx.imag[::-1], **ikw, title="STFT.imag")

#%%
ts = TestSignals(2048)
x = ts.hchirp(fmin=.11)[0]
t = np.linspace(0, 1, len(x), 0)
x += x[::-1]

for analytic in (0, 1):
    wavelet = Wavelet(('gmw', {'gamma': 1, 'beta': 1}))
    Tx, Wx, _, scales, *_ = ssq_cwt(x, wavelet, scales='log')

    if not analytic:
        psiup_t = ifft(wavelet(N=len(x)*4, scale=scales), axis=-1)
        wavelet._Psih = fft(psiup_t[:, ::2])
        Tx, Wx, *_ = ssq_cwt(x, wavelet, scales=scales)

    title = "analytic" if analytic else "non-analytic"
    for i, g in enumerate([Tx, Wx]):
        scl = .5 if i == 0 else .5
        scl = 240 if i == 0 else 3
        imshow(g[:], abs=1, w=.6, h=.45, norm=(0, np.abs(g).mean()*scl),
               yticks=0,
               title=title + (" |SSQ_CWT|" if i == 0 else " |CWT|"))

#%%
for case in (0, 1):
    beta = 10 if case == 1 else 60
    wavelet = Wavelet(('gmw', {'gamma': 3, 'beta': beta}))
    _ = ssq_cwt(np.arange(8192), wavelet)

    pf = wavelet._Psih[20].copy()
    pf = np.roll(pf, -np.argmax(pf) + len(pf)//2)
    if case == 1:
        pf[len(pf)//2 + 1:] = 0
        pf[len(pf)//2] /= 2
    pt = ifftshift(ifft(pf))

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    plot(pf, abs=1, ticks=0, fig=fig, ax=axes[0], title="freq-domain")
    plot(pt, complex=1, fig=fig, ax=axes[1])
    plot(pt, abs=1, linestyle='--', color='k', ticks=0,
         fig=fig, ax=axes[1], title="time-domain")

    ctr = len(pt)//2
    zm = 40
    slc = pt[ctr-zm:ctr+zm+1]
    plot(slc, complex=1, fig=fig, ax=axes[2])
    plot(slc, abs=1, linestyle='--', color='k', show=0, ticks=0,
         fig=fig, ax=axes[2], title="time-domain, zoomed")
    fig.subplots_adjust(hspace=.1)
    plt.show()
