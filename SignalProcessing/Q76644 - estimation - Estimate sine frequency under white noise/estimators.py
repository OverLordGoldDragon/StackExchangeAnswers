# -*- coding: utf-8 -*-
"""Estimators for sine frequency under White Gaussian Noise.

Functions aren't performance-optimized, and have redundant arguments.
"""
# https://dsp.stackexchange.com/q/76644/50076
import numpy as np
from numpy.fft import rfft, fft
from scipy.signal import hilbert


def est_freq(x, names, real=True):
    if not isinstance(names, (list, tuple)):
        names = [names]

    N = len(x)
    if real:
        X = rfft(x)
    else:
        X = fft(x)

    f_ests = [[] for _ in range(len(names))]
    for i, name in enumerate(names):
        fn = estimator_fns[name]
        if 'cedron' in name:
            # `1:-1` avoids bin duplication: happens with peak at DC/Nyquist
            # (Hermitian symmetry); it does degrade performance. Non-applicable to
            # complex or two-bin case. Actual DC/Nyquist peaks can be handled,
            # not done here (note, they have double the value of any other bin
            # for same sine amplitude)
            # (heuristic by John Muradeli)
            if real:
                kmax = np.argmax(abs(X[1:-1])) + 1
                Z = X[kmax-1:kmax+2]
            else:
                kmax = np.argmax(abs(X))
                if kmax == 0:
                    Z = np.array([X[-1], X[0], X[1]])
                elif kmax == N - 1:
                    Z = np.array([X[-2], X[-1], X[0]])
                else:
                    Z = X[kmax-1:kmax+2]
            f_est = fn(Z, kmax, N)

        elif 'kay' in name:
            if real:
                x_analytic = hilbert(x)
            else:
                x_analytic = x
            f_est = fn(x_analytic, N)

        elif name == 'dft_quadratic':
            if not real:
                raise NotImplementedError
            Npad = 2048
            Xpa = abs(rfft(x, n=Npad))
            kmax = np.argmax(Xpa[1:-1]) + 1
            Z = Xpa[kmax-1:kmax+2]
            f_est = fn(Z, kmax, Npad)# * (N / Npad)

        else:
            f_est = fn(x)

        f_ests[i].append(f_est)

    f_ests = _postprocess_f_ests(f_ests)
    return f_ests


def est_freq_multi(x, names=None, n_tones=4):
    if not isinstance(names, (list, tuple)):
        names = [names]
    for name in names:
        assert name in ('cedron_3bin', 'dft_quadratic', 'dft_argmax'), name

    N = len(x)
    X = rfft(x)
    Xa = abs(X)
    f_ests = [[] for _ in range(len(names))]

    for _ in range(n_tones):
        kmax = np.argmax(Xa[1:-1]) + 1
        for i, name in enumerate(names):
            fn = estimator_fns[name]
            if name == 'cedron_3bin':
                Z = X[kmax-1:kmax+2]
                f_est = fn(Z, kmax, N)
            elif name == 'dft_quadratic':
                Z = Xa[kmax-1:kmax+2]
                f_est = fn(Z, kmax, N)
            elif name == 'dft_argmax':
                f_est = kmax/N
            f_ests[i].append(f_est)

        # for low A, the remaining peak points of higher A can dominate;
        # clear a bit more
        Xa[kmax-3:kmax+4] = 0

    f_ests = _postprocess_f_ests(f_ests)
    return f_ests


def est_f_cedron(Z, k, N):
    """
    "Exact Frequency Formula for a Pure Real Tone in a DFT", Cedron Dawg, Eq 20
    https://www.dsprelated.com/showarticle/773.php
    """
    r = np.exp(-1j*2*np.pi/N)
    cosb = np.cos(2*np.pi/N * np.array([k-1, k, k+1]))

    num = -Z[0]*cosb[0] + Z[1]*(1 + r)*cosb[1] - Z[2]*r*cosb[2]
    den = -Z[0]         + Z[1]*(1 + r)         - Z[2]*r
    f = np.real(np.arccos(num / den)) / (2*np.pi)
    return f


def est_f_cedron_complex(Z, k, N):
    """
    "Three Bin Exact Frequency Formulas for a Pure Complex Tone in a DFT",
    Cedron Dawg,
    https://www.dsprelated.com/showarticle/1043.php
    """
    R1 = np.exp(-1j*2*np.pi/N)
    num = -R1 * Z[0] + (1 + R1) * Z[1] - Z[2]
    den = -Z[0] + (1+R1)*Z[1] - R1*Z[2]
    alpha = np.real(np.log(num / den) / 1j)

    if k > N//2:
        k = -(N - k)
    f = k/N + alpha / (2 * np.pi)
    return f


def _get_cedron_basics(Z, k, N):
    re = Z.real
    im = Z.imag

    rt2 = np.sqrt(2)
    betas = np.array([k - 1, k, k + 1]) * 2*np.pi/N
    cosb = np.cos(betas)
    sinb = np.sin(betas)
    return re, im, cosb, sinb, rt2

def _cedron_bin_finish(A, B, C):
    P = C / np.linalg.norm(C)
    D = A + B
    K = D - (D @ P)*P

    num = K @ B  # dot product
    den = K @ A
    ratio = max(min(num/den, 1), -1)  # handles float issues
    f = np.arccos(ratio) / (2*np.pi)
    return f


def est_f_cedron_3bin_complex(Z, k, N):
    """
    "Three Bin Exact Frequency Formulas for a Pure Complex Tone in a DFT",
    Cedron Dawg, Eq 19 (via Eqs 35, 31, 20, 16)
    https://www.dsprelated.com/showarticle/1043.php
    """
    R1 = np.exp(-1j*1*2*np.pi/N)
    DZ = Z * np.array([1/R1, 1, R1])
    G = np.conj(Z + DZ)
    K = G - np.mean(G)

    num = K @ Z
    den = K @ DZ
    ratio = num / den
    if k > N//2:
        k = -(N - k)
    f = k/N + np.arctan2(ratio.imag, ratio.real) / (2*np.pi)
    return f


def est_f_cedron_3bin(Z, k, N):
    """
    "Improved Three Bin Exact Frequency Formula for a Pure Real Tone in a DFT",
    Cedron Dawg, Eq 9
    https://www.dsprelated.com/showarticle/1108.php
    """
    re, im, cosb, sinb, rt2 = _get_cedron_basics(Z, k, N)

    A = np.array([(re[1] - re[0])/rt2,
                  (re[1] - re[2])/rt2,
                  im[0],
                  im[1],
                  im[2]])
    B = np.array([(cosb[1]*re[1] - cosb[0]*re[0])/rt2,
                  (cosb[1]*re[1] - cosb[2]*re[2])/rt2,
                  cosb[0]*im[0],
                  cosb[1]*im[1],
                  cosb[2]*im[2]])
    C = np.array([(cosb[1] - cosb[0])/rt2,
                  (cosb[1] - cosb[2])/rt2,
                  sinb[0],
                  sinb[1],
                  sinb[2]])

    f = _cedron_bin_finish(A, B, C)
    return f


def est_f_cedron_2bin(Z, k, N):
    """
    "A Two Bin Solution", Cedron Dawg, Eq 14
    https://www.dsprelated.com/showarticle/1284.php
    """
    re, im, cosb, sinb, rt2 = _get_cedron_basics(Z, k, N)

    A = np.array([(re[1] - re[0])/rt2,
                  im[1],
                  im[0]])
    B = np.array([(cosb[1]*re[1] - cosb[0]*re[0])/rt2,
                  cosb[1]*im[1],
                  cosb[0]*im[0]])
    C = np.array([(cosb[1] - cosb[0])/rt2,
                  sinb[1],
                  sinb[0]])

    f = _cedron_bin_finish(A, B, C)
    return f


def est_f_cedron_jacobsen(Z, k, N):
    """
    "Candan's Tweaks of Jacobsen's Frequency Approximation", Cedron Dawg, Eq 28
    https://www.dsprelated.com/showarticle/1481.php
    """
    num = Z[0] - Z[2]
    den = -Z[0] + 2*Z[1] - Z[2]

    den2 = den.real**2 + den.imag**2

    QJ = (num.real*den.real + num.imag*den.imag) / den2
    f = (np.arctan(QJ * np.tan(np.pi/N))  / (np.pi/N) + k) / N
    return f


def est_f_kay_1(x, N):
    """
    "A Fast and Accurate Single Frequency Estimator", Steven Kay, Eq 17
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.320.7222&rep=rep1&type=pdf
    """
    f = np.sum(np.angle(np.conj(x[:-1]) * x[1:])) / (2*np.pi*(N-1))
    return f


def est_f_kay_2(x, N):
    """
    "A Fast and Accurate Single Frequency Estimator", Steven Kay, Eq 16
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.320.7222&rep=rep1&type=pdf
    """
    idxs = np.arange(N - 1)
    weights = 1.5*N / (N**2 - 1) * (1 - ((idxs - (N/2 - 1)) / (N/2))**2)
    f = np.sum(weights*np.angle(np.conj(x[:-1]) * x[1:])) / (2*np.pi)
    return f


def est_f_dft_argmax(x):
    return np.argmax(abs(rfft(x)))


def est_f_dft_quadratic(Z, kmax, N):
    p1, p2, p3 = abs(Z)
    f = (kmax + ((p1 - p3) / (2*(p1 + p3)))) / N
    return f


def _postprocess_f_ests(f_ests):
    for i in range(len(f_ests)):
        f_ests[i] = np.array(f_ests[i])
    if len(f_ests) == 1:
        f_ests = float(f_ests[0])
    return f_ests


estimator_fns = {
    'cedron': est_f_cedron,
    'cedron_complex': est_f_cedron_complex,
    'cedron_3bin': est_f_cedron_3bin,
    'cedron_2bin': est_f_cedron_2bin,
    'cedron_3bin_complex': est_f_cedron_3bin_complex,
    'cedron_jacobsen': est_f_cedron_jacobsen,
    'kay_1': est_f_kay_1,
    'kay_2': est_f_kay_2,
    'dft_argmax': est_f_dft_argmax,
    'dft_quadratic': est_f_dft_quadratic,
}
