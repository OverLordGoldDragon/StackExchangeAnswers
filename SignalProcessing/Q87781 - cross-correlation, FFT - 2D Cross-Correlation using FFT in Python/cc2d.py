# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87781/50076
import numpy as np
from scipy.fft import next_fast_len, fft2, ifft2


def cross_correlate_2d(x, h, mode='same', real=True, get_reusables=False,
                       inplace=True, workers=-1):
    """2D cross-correlation, replicating `scipy.signal.correlate2d`.

    Parameters
    ----------
    x : np.ndarray, 2D
        Input.

    h : np.ndarray, 2D
        Filter/template.

    mode : str
        `'full'`, `'same'`, or `'valid'` (see scipy docs).

    real : bool (default True)
        Whether to assume `x` and `h` are real-valued, which is faster.

    get_reusables : bool (default False)
        Whether to return `out, reusables`.

        If `h` is same and `x` has same shape, pass in `reusables` as `h` for
        speedup,

    inplace : bool (default True)
        If True, is faster but may alter `x` and/or `h` that are passed in
        (unless via `reusables`).

    workers : int
        Number of CPU cores to use with FFT. Defaults to `-1`, which is all.
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
            if inplace:
                np.conj(hadj[::-1, ::-1], out=hadj)
            else:
                hadj = np.conj(hadj[::-1, ::-1])
        hf = fft2(hadj, padded_shape, workers=workers)
    else:
        reusables = h
        (hf, swap, padded_shape, unpad_h, unpad_w) = reusables
        if swap:
            xadj, hadj = h, x
        else:
            xadj, hadj = x, h

    # FFT convolution
    xf = fft2(xadj, padded_shape, workers=workers)
    if inplace:
        np.multiply(xf, hf, out=xf)
    else:
        xf = xf * hf
    out = ifft2(xf, workers=workers)
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
