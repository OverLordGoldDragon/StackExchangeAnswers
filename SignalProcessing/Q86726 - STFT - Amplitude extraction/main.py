# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/85745/50076
import numpy as np
import scipy.signal
from ssqueezepy import stft, ssq_stft
from ssqueezepy.visuals import plotscat, imshow


def viz(S):
    slc = S[:, 0]
    imshow(S, abs=1, w=.7, h=.55)
    plotscat(slc, complex=1, w=.6, h=.82, show=1)
    print(slc.sum(), flush=True)


N = 256
t = np.linspace(0, 1, N, 1)
x = np.cos(2*np.pi * 64 * t)
window = scipy.signal.windows.dpss(N, N//4, sym=False)

Sx = stft(x, window, n_fft=len(window), hop_len=1)
Sx[1:-1] *= 2
Sx /= N

viz(Sx)

Tx = ssq_stft(x, window, n_fft=len(window))[0]
# ... *almost* all the work
Tx[1:-1] *= 2
viz(Tx)

# ignore ssqueezepy warning (there is as of writing, false positive)
