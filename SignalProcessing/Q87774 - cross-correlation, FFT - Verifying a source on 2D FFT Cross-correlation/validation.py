# -*- coding: utf-8 -*-
def cross_correlate_2d(x, h):
    h = ifftshift(ifftshift(h, axes=0), axes=1)
    return ifft2(fft2(x) * np.conj(fft2(h)))

##############################################################################
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

def crand(*s):
    return np.random.randn(*s) + 1j*np.random.randn(*s)

for M in (64, 65, 99):
    for N in (64, 65, 99):
        # case 1 -------------------------------------------------------------
        x = crand(M, N)

        o = cross_correlate_2d(x, x)
        hmax, wmax = np.where(abs(o) == abs(o).max())
        assert hmax == M//2, (hmax, M//2)
        assert wmax == N//2, (wmax, N//2)

        # case 2 -------------------------------------------------------------
        x = np.zeros((M, N), dtype='complex128')
        h = np.zeros((M, N), dtype='complex128')

        hctr, wctr = M//8, N//8
        hsize, wsize = hctr*2, wctr*2
        target_loc = slice(0, hsize), slice(0, wsize)
        false_positive_loc = slice(M//2, M//2 + hsize), slice(-wsize, None)
        target = crand(hsize, wsize)

        x[target_loc] = target
        x[false_positive_loc] = target[::-1, ::-1]
        h[M//2 - hctr:M//2 + hctr, N//2 - wctr:N//2 + wctr] = target

        o = cross_correlate_2d(x, h)
        hmax, wmax = np.where(abs(o) == abs(o).max())
        assert hmax == hctr, (hmax, hctr)
        assert wmax == wctr, (wmax, wctr)

        # case 3 -------------------------------------------------------------
        x[false_positive_loc] = np.conj(target)

        o = cross_correlate_2d(x, h)
        hmax, wmax = np.where(abs(o) == abs(o).max())
        assert hmax == hctr, (hmax, hctr)
        assert wmax == wctr, (wmax, wctr)
