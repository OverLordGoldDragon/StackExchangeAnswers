# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/85745/50076
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
    """
      - no hop or pad support
      - equivalent to `'wrap'` (periodic) pad if `len(window) < len(x)//2`
      - real-valued `x` only
      - returns `Sx`, `fbank_f`
    """
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


#%% first, validate correctness
for N in (128, 129):
    x = np.random.randn(N)
    for n_fft in (55, 56):
        window = np.abs(np.random.randn(n_fft) + .01)

        Sx0 = stft(x, window, n_fft=n_fft, modulated=True,
                   padtype='wrap', hop_len=1, dtype='float64')
        Sx1, _ = stft_rowwise(x, window, n_fft)

        assert np.allclose(Sx0, Sx1)

#%% Demo filterbanks
N = 256
x = np.random.randn(N)
pkw = dict(color='tab:blue', w=1, h=.6)
window = dpss(N//6, N//6//4 - 1, sym=False)

plotscat(window, title="scipy.signal.windows.dpss(42, 9.5, sym=False)", show=1)

# full spectrum
n_fft = N
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)
_, fbank_f0 = stft_rowwise(x, window_adj_f, n_fft)
plot(fbank_f0.T.real, title="fbank_f0 | n_fft=N", **pkw, show=1)

# integer subset
n_fft = N//4
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)
_, fbank_f1 = stft_rowwise(x, window_adj_f, n_fft)
assert np.allclose(fbank_f1, fbank_f0[::4])
plot(fbank_f1.T.real, title="fbank_f1 = fbank_f0[::4] | n_fft=N/4", **pkw,
     show=1)

# fractional subset
n_fft = int(N/5.5)
window_adj, window_adj_f = get_adj_windows(window, n_fft, N)
_, fbank_f2 = stft_rowwise(x, window_adj_f, n_fft)
plot(fbank_f2.T.real,
     title="fbank_f2, fbank_f1 | n_fft=floor(N/5.5), xlims=(100, 150)", **pkw)
plot(fbank_f1.T.real, show=1, color='tab:orange', xlims=(100, 150))

# time-domain examples
def plot_complex(pt, f):
    plot(pt, complex=1)
    plot(pt, abs=1, linestyle='--', color='k', w=.5, h=.8,
         title=f"STFT filter | f={f}", show=1, ylims=(-1.03, 1.03))

f0, f1 = 20, 40
pt0 = ifftshift(ifft(fbank_f0[f0]))
pt1 = ifftshift(ifft(fbank_f0[f1]))
plot_complex(pt0, f0)
plot_complex(pt1, f1)
