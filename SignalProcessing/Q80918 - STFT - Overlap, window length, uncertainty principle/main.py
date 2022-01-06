# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/a/80920/50076 ################################
import numpy as np
from ssqueezepy import ssq_stft
from ssqueezepy.visuals import imshow
from scipy.signal.windows import dpss

# signal
N = 4096
f = 32
t = np.linspace(0, 1, N, 0)[::-1]
fm = np.cos(2*np.pi * f * t) / (100*f)
x =  np.cos(2*np.pi * f*8 * (t + fm))

# window
n_fft = 512
window = dpss(n_fft, n_fft//2 - 1)

# STFT & SSQ
Tx0, Sx0, *_ = ssq_stft(x, window, n_fft=n_fft, flipud=0, hop_len=1)
Tx1, Sx1, *_ = ssq_stft(x, window, n_fft=n_fft, flipud=0, hop_len=64)

# visualize ##################################################################
# cheat & drop boundary effects for clarity
Sx1, Tx1 = Sx1[:, 1:], Tx1[:, 1:]
kw = dict(abs=1, interpolation='none', w=.8, h=.6)
imshow(Sx0, **kw, title=("|STFT|     -- hop_len=1",  {'fontsize': 18}))
imshow(Tx0, **kw, title=("|SSQ_STFT| -- hop_len=1",  {'fontsize': 18}))
imshow(Sx1, **kw, title=("|STFT|     -- hop_len=64", {'fontsize': 18}))
imshow(Tx1, **kw, title=("|SSQ_STFT| -- hop_len=64", {'fontsize': 18}))
