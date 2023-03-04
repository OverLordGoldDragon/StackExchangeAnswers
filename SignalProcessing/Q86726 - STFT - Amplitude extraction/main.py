# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/86726/50076
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, dpss
from numpy.fft import fft, fftshift, ifftshift
from ssqueezepy import stft, ssq_stft
from ssqueezepy.visuals import plotscat, imshow

#%%############################################################################
# Main example
# ------------

def viz(Sx, Sxs):
    slcs = Sxs[:, 0]
    imshow(Sxs, abs=1, w=.7, h=.55, interpolation='none')
    plotscat(slcs, complex=1, w=.6, h=.82, show=1)
    print("sum(Sx[:, 0]) =", slcs.sum(), flush=True)
    print("max(abs(Sx[:, 0])) / sum(window) =", abs(slcs).max() / window.sum(),
          flush=True)

# gen signal
N = 256
t = np.linspace(0, 1, N, 0)
x = np.cos(2*np.pi * 60 * t)

# gen window
window = dpss(N//2, N//8 - 1, sym=False)
# this is always done under the hood, `len(window) == n_fft`
window = np.pad(window, (N - len(window)) // 2)

# match input length so `x` is seen as perfect sinusoid by DFT;
# can also be integer fraction of `N` but then `x` can't be of all int freqs
n_fft = N
# in general should be 'reflect' but slightly complicates example here, simplify
padtype = 'wrap'

Sx = stft(x, window, n_fft=n_fft, hop_len=1, padtype=padtype, dtype='float64')
Sxs = Sx.copy()
Sxs[1:-1] *= 2
viz(Sx, Sxs)

#%%
Tx = ssq_stft(x, window, n_fft=len(window), padtype=padtype)[0]
# ... *almost* all the work
Txs = Tx.copy()
Txs[1:-1] *= 2
viz(Txs, Txs)

# ignore ssqueezepy warning (there is as of writing, false positive)

#%%############################################################################
# Criteria demos
# --------------

# helper
def viz2(S, w, wf, sig_title, tm_idx=128):
    slc = S[:, tm_idx]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    imshow(S, abs=1, ax=axes[0], fig=fig, show=0,
           title="abs(Sx_adj) | " + sig_title, interpolation='none')
    plotscat(slc, abs=1, ax=axes[1], fig=fig,
             title=f"abs(Sx_adj[:, {tm_idx}])")
    fig.subplots_adjust(wspace=.1)
    plt.show()

    print(
        ("max(abs(x)) = {1:.3g}\n"
         "max(abs(Sx_adj[:, {0}])) = {2:.3g}\n"
         "sum(abs(Sx_adj[:, {0}])) = {3:.3g}\n"
         "max(abs(Sx_adj[:, {0}])) / sum(window_adj) = {4:.3g} -- time norm\n"
         "sum(abs(Sx_adj[:, {0}])) / sum(abs(fft(window_adj_f))) = {5:.3g} "
         "-- freq norm\n"
         ).format(
             tm_idx,
             abs(x).max(),
             sum(abs(S[:, tm_idx])),
             max(abs(S[:, tm_idx])),
             max(abs(S[:, tm_idx])) / sum(w),
             sum(abs(S[:, tm_idx])) / sum(abs(fft(wf))),
             )
    )


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

N = 256
t = np.linspace(0, 1, N, 0)

#%%############################################################################
# Everything's right
# ^^^^^^^^^^^^^^^^^^
n_fft = N
window = dpss(N//8, N//8//2 - 1, sym=False)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)

f = 60
x = np.cos(2*np.pi * f * t)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype)
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2

viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{f}*t)")

#%%############################################################################
# Nonstationary case
# ^^^^^^^^^^^^^^^^^^
f = 60
x = np.cos(2*np.pi * f * t**2)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype)
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2

viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{f}*t**2)")

#%%############################################################################
# Too close to Nyquist
# ^^^^^^^^^^^^^^^^^^^^
f = N//2 - 10
x = np.cos(2*np.pi * f * t)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype)
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2

viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{f}*t)", 32)

#%%############################################################################
# Too close to DC
# ^^^^^^^^^^^^^^^
f = 10
x = np.cos(2*np.pi * f * t)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype, dtype='float64')
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2

viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{f}*t)", 32)

#%%############################################################################
# Multi-component intersection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
f, fm = 60, 50
x = np.cos(2*np.pi * f * t) + np.cos(2*np.pi * fm * t**2)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype)
Sxs = Sx.copy()
Sxs[1:-1] *= 2

viz2(Sxs, window_adj, window_adj_f, f"\ncos(2*pi*{f}*t) + cos(2*pi*{fm}*t**2)")

#%%############################################################################
# Insufficient `n_fft`
# ^^^^^^^^^^^^^^^^^^^^
f = 60
x = np.cos(2*np.pi * f * t)
n_fft = 36
window = hann(n_fft, sym=False)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)

Sx = stft(x, window, n_fft=n_fft, padtype=padtype, dtype='float64')
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2

viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{f}*t)")

#%%############################################################################
# Excessive `hop_size`
# ^^^^^^^^^^^^^^^^^^^^
fc, fa = 60, 2
x = np.sin(2*np.pi * fa * t) * np.cos(2*np.pi * fc * t)
n_fft = len(x)
window = dpss(n_fft, n_fft//2 - 1, sym=False)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)

Sx = stft(x, window, n_fft=n_fft, padtype=padtype, dtype='float64',
          hop_len=64)
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2
viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{fc}*t) * sin(2*pi*{fa}*t)",
     2)

#%%############################################################################
# Minimal `hop_size`
# ^^^^^^^^^^^^^^^^^^
Sx = stft(x, window, n_fft=n_fft, padtype=padtype, dtype='float64',
          hop_len=1)
Sx_adj = Sx.copy()
Sx_adj[1:-1] *= 2
viz2(Sx_adj, window_adj, window_adj_f, f"cos(2*pi*{fc}*t) * sin(2*pi*{fa}*t)",
     96)

#%%############################################################################
# Non-localized window
# ^^^^^^^^^^^^^^^^^^^^
np.random.seed(0)
n_fft = N
window = np.abs(np.random.randn(n_fft//4) + .01)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)

f = 60
x = np.cos(2*np.pi * f * t)
Sx = stft(x, window, n_fft=n_fft, padtype=padtype)
Sxs = Sx.copy()
Sxs[1:-1] *= 2

viz2(Sxs, window_adj, window_adj_f, f"cos(2*pi*{f}*t)")
