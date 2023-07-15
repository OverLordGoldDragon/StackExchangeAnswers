# -*- coding: utf-8 -*-
"""Estimators for sine frequency under White Gaussian Noise.

Functions aren't performance-optimized, and have redundant arguments.
"""
# https://dsp.stackexchange.com/q/76644/50076
import numpy as np
from numpy.fft import rfft
from scipy.signal import hilbert


def est_freq(x, name):
    fn = estimator_fns[name]
    N = len(x)

    if 'cedron' in name:
        X = rfft(x)
        # `1:-1` avoids bin duplication: happens with peak at DC/Nyquist
        # (Hermitian symmetry); it does degrade performance. Non-applicable to
        # complex or two-bin case. Actual DC/Nyquist peaks can be handled,
        # not done here (note, they have double the value of any other bin
        # for same sine amplitude)
        # (heuristic by John Muradeli)
        kmax = np.argmax(abs(X[1:-1])) + 1

        Z = X[kmax-1:kmax+2]
        f = fn(Z, kmax, N)
    elif 'kay' in name:
        x_analytic = hilbert(x)
        f = fn(x_analytic, N)
    else:
        f = fn(x)
    return f


def est_freq_multi(x, n_tones=4):
    N = len(x)
    X = rfft(x)
    Xa = abs(X)
    f_ests0, f_ests1 = [], []

    for _ in range(n_tones):
        kmax = np.argmax(Xa[1:-1]) + 1
        Z = X[kmax-1:kmax+2]
        f_est0 = est_f_cedron_3bin(Z, kmax, N)
        # for low A, the remaining peak points of higher A can dominate;
        # clear a bit more
        Xa[kmax-3:kmax+4] = 0

        # DFT peak
        f_est1 = kmax/N

        f_ests0.append(f_est0)
        f_ests1.append(f_est1)

    return np.array(f_ests0), np.array(f_ests1)


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
    "Exact Frequency Formula for a Pure Real Tone in a DFT", Cedron Dawg
    https://www.dsprelated.com/showarticle/773.php
    """
    R1 = np.exp(-1j*2*np.pi/N)
    num = -R1 * Z[0] + (1 + R1) * Z[1] - Z[2]
    den = -Z[0] + (1+R1)*Z[1] - R1*Z[2]
    alpha = np.real(np.log(num / den) / 1j)
    f = (alpha / (2 * np.pi)) + k / N
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


estimator_fns = {
    'cedron': est_f_cedron,
    'cedron_complex': est_f_cedron_complex,
    'cedron_3bin': est_f_cedron_3bin,
    'cedron_2bin': est_f_cedron_2bin,
    'cedron_jacobsen': est_f_cedron_jacobsen,
    'kay_1': est_f_kay_1,
    'kay_2': est_f_kay_2,
    'dft_argmax': est_f_dft_argmax,
}
