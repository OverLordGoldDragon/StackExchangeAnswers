# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87781/50076
import numpy as np
import scipy.signal
from cc2d import cross_correlate_2d

#%% Testing ##################################################################
np.random.seed(0)

rand = lambda M, N, real: (np.random.randn(M, N) if real else
                           (np.random.randn(M, N) + 1j*np.random.randn(M, N)))

lengths = (1, 2, 3, 4, 5, 6, 7, 15, 50)

for real in (True, False):
    for mode in ('full', 'same', 'valid'):
        for M0 in lengths:
            for N0 in lengths:
                x = rand(M0, N0, real)
                for M1 in lengths:
                    for N1 in lengths:
                        h = rand(M1, N1, real)

                        fn0 = lambda: cross_correlate_2d(
                            x.copy(), h.copy(), mode, real=real)
                        fn1 = lambda: scipy.signal.correlate2d(
                            x.copy(), h.copy(), mode=mode)

                        # compute
                        try:
                            out0 = fn0()
                        except ValueError:
                            try:
                                fn1()
                            except ValueError:
                                continue
                            except:
                                raise AssertionError
                        out1 = fn1()

                        # assert equality
                        cfg = (real, mode, M0, N0, M1, N1)
                        assert out0.shape == out1.shape, cfg
                        assert np.allclose(out0, out1), cfg


# check reusables
out0, reusables = cross_correlate_2d(x, h,  get_reusables=True)
out1 = cross_correlate_2d(x, reusables)
assert np.allclose(out0, out1)
