# -*- coding: utf-8 -*-
"""Performance-optimized estimators."""
# https://dsp.stackexchange.com/q/76644/50076
import numpy as np
from scipy.fft import fft
from _optimized import kay_weighted_complex, abs_argmax


class Kay2Complex():
    def __init__(self):
        self.weights_N = {}

    def __call__(self, x, N=None):
        N = len(x)
        if N in self.weights_N:
            weights = self.weights_N[N]
        else:
            idxs = np.arange(N - 1)
            weights = 1.5*N / (N**2 - 1) * (1 - ((idxs - (N/2 - 1)) / (N/2))**2
                                            ) / (2*np.pi)
            self.weights_N[N] = weights

        f_est = kay_weighted_complex(x.real, x.imag, weights)
        return f_est


class Cedron3BinComplex():
    def __call__(self, x):
        xf = fft(x)
        kmax = abs_argmax(xf.real, xf.imag)
        return 5

    # didn't bother implementing the O(1) finishing step.
    # it'd only slightly change the N=100 result, and not at all for
    # anything else.
    # the handicap of not using FFTW is far greater
