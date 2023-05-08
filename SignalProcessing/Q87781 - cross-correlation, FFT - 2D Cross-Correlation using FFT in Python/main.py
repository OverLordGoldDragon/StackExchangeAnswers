# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87781/50076
import numpy as np
import scipy.signal
from scipy.fft import next_fast_len, fft2, ifft2


#%% Implementation ###########################################################
def cross_correlate_2d(x, h, mode='same', real=True, get_reusables=False):
    """2D cross-correlation, replicating `scipy.signal.correlate2d`.

    `reusables` are passed in as `h`.
    Set `get_reusables=True` to return `out, reusables`.
    """
    # check if `h` is reusables
    if not isinstance(h, tuple):
        # fetch shapes, check inputs
        xs, hs = x.shape, h.shape
        h_not_smaller = all(hs[i] >= xs[i] for i in (0, 1))
        x_not_smaller = all(xs[i] >= hs[i] for i in (1, 0))
        if mode == 'valid' and not (h_not_smaller or x_not_smaller):
            raise ValueError(
                "For `mode='valid'`, every axis in `x` must be at least "
                "as long as in `h`, or vice versa. Got x:{}, h:{}".format(
                                 str(xs), str(hs)))

        # swap if needed
        swap = bool(mode == 'valid' and not x_not_smaller)
        if swap:
            xadj, hadj = h, x
        else:
            xadj, hadj = x, h
        xs, hs = xadj.shape, hadj.shape

        # compute pad quantities
        full_len_h = xs[0] + hs[0] - 1
        full_len_w = xs[1] + hs[1] - 1
        padded_len_h = next_fast_len(full_len_h)
        padded_len_w = next_fast_len(full_len_w)
        padded_shape = (padded_len_h, padded_len_w)

        # compute unpad indices
        if mode == 'full':
            offset_h, offset_w = 0, 0
            len_h, len_w = full_len_h, full_len_w
        elif mode == 'same':
            len_h, len_w = xs
            offset_h, offset_w = [g//2 for g in hs]
        elif mode == 'valid':
            ax_pairs = ((xs[0], hs[0]), (xs[1], hs[1]))
            len_h, len_w = [max(g) - min(g) + 1 for g in ax_pairs]
            offset_h, offset_w = [min(g) - 1 for g in ax_pairs]
        unpad_h = slice(offset_h, offset_h + len_h)
        unpad_w = slice(offset_w, offset_w + len_w)

        # handle filter / template
        if real:
            hadj = hadj[::-1, ::-1]
        else:
            hadj = np.conj(hadj)[::-1, ::-1]
        hf = fft2(hadj, padded_shape)
    else:
        reusables = h
        (hf, swap, padded_shape, unpad_h, unpad_w) = reusables
        if swap:
            xadj, hadj = h, x
        else:
            xadj, hadj = x, h

    # FFT convolution
    out = ifft2(fft2(xadj, padded_shape) * hf)
    if real:
        out = out.real

    # unpad, unswap
    out = out[unpad_h, unpad_w]
    if swap:
        out = out[::-1, ::-1]

    # pack reusables
    if get_reusables:
        reusables = (hf, swap, padded_shape, unpad_h, unpad_w)

    # return
    return ((out, reusables) if get_reusables else
            out)

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

                        fn0 = lambda: cross_correlate_2d(x, h, mode, real=real)
                        fn1 = lambda: scipy.signal.correlate2d(x, h, mode=mode)

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
