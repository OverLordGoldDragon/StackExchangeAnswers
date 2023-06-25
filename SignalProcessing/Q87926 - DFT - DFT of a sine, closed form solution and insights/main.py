# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87926/50076
"""Tests:

    1. Sine DFT solution, against `fft(x)` (solution by Cedron Dawg)
       [1] https://www.dsprelated.com/showarticle/771.php
       [2] # TODO url
    2. Modulus extension of (1), against `abs(fft(x))`
       [3] # TODO url
    3. N/4 symmetry
       [4] # TODO url
    4. Unwindowed STFT solution, against `scipy.signal.stft`
       [5] # TODO url
    5. Shift-dependence, via an example (with visuals)
       [6] Same as [4]

The more numerically precise version isn't tested, as I found it late.

In this less precise version, the following are observed:

    - All of 1-4 are validated within numeric precision, float64 (`np.allclose`),
      for vast majority of sine parameters
    - 3, 4 are validated for all parameters
    - Poor performance of 1 & 2 for near-integer `f` (e.g. `5.0000001`),
      worse if `N` is higher; that's expected ([2], Appendix A)
    - Poor performance of 2 for the DC bin (unclear why)
    - 5 can also be validated within float precision, I just didn't
"""
import numpy as np
from numpy.fft import fft, rfft
from solutions import sine_dft, sine_dft_modulus, sine_stft

import warnings
try:
    import matplotlib.pyplot as plt
except:
    plt = None
    warnings.warn("`matplotlib` not installed, skipping visual example")

try:
    import scipy, scipy.signal
except:
    scipy = None
    warnings.warn("Couldn't import `scipy.signal`, skipping STFT test")

#%%############################################################################
# Configure testing
# -----------------

# high-level ------------------------------------------------------------------
# whether to print progress
VERBOSE = 1

# parameter sweeps ------------------------------------------------------------
# N==1 also works but takes more coding
N_all = (2, 3, 4, 14, 15, 16, 17, 1002, 10009, 123456)
# phases
phi_all = (0, 0.13, 1, -0.145)
# f = f_frac * N;
# log spacing more uniformly explores functional space;
# include positives and negatives, and in the testing loop, make integer cases
# from fractional ones but don't repeat
f_frac_all = np.logspace(np.log10(1e-5), np.log10(0.5), 64, endpoint=False)
f_frac_all = np.hstack([-f_frac_all[::-1], 0, f_frac_all])[::-1]
df_all = (0.01, 0.13, 0.51, 1.1, -0.145)
# note, `f_frac_all` deliberately excludes `f` very close to integer,
# as advertised

# STFT
# testing takes long, trim to minimum
N_all_stft = (2, 3, 4, 14, 1002, 10009)
phi_all_stft = (0, 1.13, -0.145)
_ff = np.array([-.5, -.44, -.31, -.25, -0.0001])
f_frac_all_stft = np.hstack([_ff, 0, _ff*.99])
# tests less hops and other configs past this `N`
high_N_stft = 1000
# note, `f_frac_all_stft` deliberately excludes `f` very close to integer, so
# it's not actually "for all parameters", but it is for all STFT parameters,
# hence "all" follows from the numerically precise version

# testing ---------------------------------------------------------------------
# numpy defaults, float64
rtol_default = 1e-5
atol_default = 1e-8
# high `N` runs into numerical precision limitations
rtol_high_N = rtol_default * 1
atol_high_N = atol_default * 2
# there's something particularly imprecise about the DC bin of `sine_dft_modulus`
rtol_high_N_axf_dc = rtol_default * 100
atol_high_N_axf_dc = atol_default * 1
# define "high N"
high_N = int(1e5)
high_N_axf_dc = int(1e4)

_get_tols = lambda N: (
    dict(rtol=rtol_default, atol=atol_default) if N < high_N else
    dict(rtol=rtol_high_N,  atol=atol_high_N))
_get_tols_axf_dc = lambda N: (
    dict(rtol=rtol_default, atol=atol_default) if N < high_N_axf_dc else
    dict(rtol=rtol_high_N_axf_dc,  atol=atol_high_N_axf_dc))

# helpers ---------------------------------------------------------------------
if VERBOSE:
    _print = lambda *a, **k: print(*a, **k)
else:
    _print = lambda *a, **k: None

def assert_allclose(a, b, tols, info, var_names):
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    ae = abs(a - b)
    argmax_ae = np.argmax(ae)
    max_ae = ae[argmax_ae]

    info = var_names + info
    info = info + "\nAbsolute Error: (max, index of max) = ({:.3g}, {})".format(
        max_ae, argmax_ae)

    assert np.allclose(a, b, **tols), info

#%%############################################################################
# Testing: raw formulae
# ---------------------
_print("\nTEST: raw formulae")

for N in N_all:
    t = np.arange(N) / N
    f_ints_done = []
    for phi in phi_all:
        for f_frac in f_frac_all:
            for do_int in (False, True):
                # make signal
                f = f_frac * N
                if do_int:
                    if f in f_ints_done:
                        continue
                    f = int(f)
                x = np.cos(2*np.pi * f * t + phi)

                # get DFTs
                xf0 = fft(x)
                xf1 = sine_dft(N, f, phi)

                # get |DFT|s
                axf0 = abs(xf0)
                axf1 = sine_dft_modulus(N, f, phi)

                # for the integer case, the formulae are more accurate than `fft`,
                # and assertions fail for large `N` since "float zero" of `fft` is
                # bigger for bigger input - so we handle it here by normalizing by
                # the previous power of 10 (can change test instead but more code)
                if do_int:
                    xf0, xf1, axf0, axf1 = [g / 10**int(np.log10(N)) for g in
                                            (xf0, xf1, axf0, axf1)]

                # prepare testing
                tols = _get_tols(N)
                tols_axf_dc = _get_tols_axf_dc(N)
                ckw = dict(N=N, phi=phi, f=f)
                info = "\n  " + "\n  ".join(
                    f"{k}={v}" for k, v in dict(**ckw, **tols).items())
                info_axf_dc = "\n  " + "\n  ".join(
                    f"{k}={v}" for k, v in dict(**ckw, **tols_axf_dc).items())

                # test
                assert_allclose(xf0,  xf1,  tols, info,
                                "xf0, xf1")
                assert_allclose(axf0[1:], axf1[1:], tols, info,
                                "axf0[1:], axf1[1:]")
                assert_allclose(axf0[0],  axf1[0],  tols_axf_dc, info_axf_dc,
                                "axf0[0], axf1[0]")

                # post-loop steps
                if do_int:
                    f_ints_done.append(f)
    _print(f"N={N} done")

#%%############################################################################
# Testing: N/4-symmetry, even N; N/4-asymmetry, odd N
# ---------------------------------------------------
_print("\nTEST: N/4-symmetry, even N; N/4-asymmetry, odd N")

# N==1 also works but takes more coding
for N in N_all:
    # set outcome
    cond = bool(N % 2 == 0)

    t = np.arange(N)/N
    for df in df_all:
        for phi in phi_all:
            # make signal
            f0 = N//2/2 + df
            f1 = N//2/2 - df

            phi0, phi1 = phi, -phi
            x0 = np.cos(2*np.pi * f0 * t + phi0)
            x1 = np.cos(2*np.pi * f1 * t + phi1)

            # DFT - use rDFT for cleaner coding
            xf0 = rfft(x0)
            xf1 = rfft(x1)

            # modulus
            axf0 = abs(xf0)
            axf1 = abs(xf1)

            # prepare testing
            info = "\n  " + "\n  ".join(
                f"{k}={v}" for k, v in dict(N=N, df=df, phi=phi).items())

            # test
            assert np.allclose(xf0,  xf1.conj()[::-1]) is cond, info
            assert np.allclose(axf0, axf1[::-1])       is cond, info
    _print(f"N={N} done")

#%%############################################################################
# Reproduce the shift-dependence plots
# ------------------------------------
if plt is not None:
    # configure, make sine ---------------------------------------------------
    N = 501
    phi = 1.3
    f = 23.5
    n_hops = 129
    seg_len = 123

    t = np.arange(N)/N
    x = np.cos(2*np.pi * f * t + phi)

    # compute ----------------------------------------------------------------
    # this is identical to shifting the underlying function, and in this case
    # there's no cheating by coincidence with integer `tau`
    re = np.zeros((n_hops, seg_len//2 + 1))
    im = np.zeros((n_hops, seg_len//2 + 1))
    for i in range(n_hops):
        seg = x[i:i + seg_len]
        segf = rfft(seg)
        re[i] = segf.real
        im[i] = segf.imag

    r_re, r_im = [np.zeros_like(re) for _ in range(2)]
    r_re = re / re[0, :][None]
    r_im = im / im[0, :][None]
    r_re[np.isnan(r_re) + np.isinf(r_re)] = 0
    r_im[np.isnan(r_im) + np.isinf(r_im)] = 0

    # plot -------------------------------------------------------------------
    amx0, amx1 = np.max(abs(r_re)), np.max(abs(r_im))
    plt.imshow(r_re.T, cmap='bwr', aspect=2, vmin=-amx0, vmax=amx0)
    plt.show()
    plt.imshow(r_im.T, cmap='bwr', aspect=2, vmin=-amx1, vmax=amx1)
    plt.show()

    a = r_im[:, 1]
    b = x[:len(r_im[:, 1])]
    a, b = a / a.max(), b / b.max()
    plt.plot(a)
    plt.plot(b)
    plt.show()
    plt.plot(a)
    plt.plot(np.roll(b, 29))
    plt.show()

#%%############################################################################
# Testing: unwindowed STFT of sine
# --------------------------------
_print("\nTEST: unwindowed STFT of sine")

# `stft` kwargs
scipy_ckw = dict(return_onesided=False, padded=False, boundary=None,
                 detrend=False, fs=1)

# signal loops ----------------------------------------------------------------
for N in N_all_stft:
    t = np.arange(N) / N
    f_ints_done = []
    for phi in phi_all:
        for f_frac in f_frac_all_stft:
            for do_int in (False, True):
                f = f_frac * N
                if do_int:
                    if f in f_ints_done:
                        continue
                    f = int(f)
                x = np.cos(2*np.pi * f * t + phi)

                # STFT loops -------------------------------------------------
                if N <= high_N_stft:
                    M_all = range(1, N)
                    H_all = range(1, N + 1)
                else:
                    M_all = (N//2 - 1, N//2, N//4)
                    H_all = (127, 256)

                for M in M_all:
                    for H in H_all:
                        out0 = sine_stft(N, M, H, f, phi)
                        out1 = scipy.signal.stft(
                            x, window=np.ones(M), nperseg=M, noverlap=M-H, nfft=M,
                            **scipy_ckw)[-1]
                        # scipy lacks `scaling = None` and divides by
                        # `sqrt(sum(window)**2)`
                        out1 *= M

                        info = "\n  " + "\n  ".join(
                            f"{k}={v}" for k, v in dict(
                                N=N, phi=phi, f=f, do_int=do_int, M=M, H=H,
                                out0_shape=out0.shape, out1_shape=out1.shape,
                                ).items())
                        assert out0.shape == out1.shape, info
                        assert np.allclose(out0, out1), info
    _print(f"N={N} done")
