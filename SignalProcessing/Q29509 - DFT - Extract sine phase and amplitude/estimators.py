# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/29509/50076
import numpy as np
from numpy.fft import rfft


def est_amp_phase_cedron_2bin(x):
    """Amplitude & phase.
    "A Two Bin Solution", Cedron Dawg, Eqs 25 & 26
    https://www.dsprelated.com/showarticle/1284.php
    """
    # not performance-optimized

    # get DFT & compute freq
    N = len(x)
    xf = rfft(x)

    if N == 3:
        kmax = 1
        Z = xf
        f_N = _est_f_cedron_2bin(Z, kmax, N)
    else:
        kmax = np.argmax(abs(xf[1:-1])) + 1  # see other referenced answer
        Z = xf[kmax-1:kmax+2]
        f_N = _est_f_cedron_3bin(Z, kmax, N)

    # run two bin calculations
    alpha = f_N * (2*np.pi)
    alphaN = alpha * N

    kj = kmax - 1
    kk = kmax
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    cosbj, cosbk = np.cos(2*np.pi/N * np.array([kj, kk]))
    sinbj, sinbk = np.sin(2*np.pi/N * np.array([kj, kk]))

    UA = np.cos(alphaN) - 1
    VA = np.cos(alphaN - alpha) - cosa

    UB = np.sin(alphaN)
    VB = np.sin(alphaN - alpha) + sina

    fj = 1 / (2 * N * (cosa - cosbj))
    fk = 1 / (2 * N * (cosa - cosbk))

    A = np.array([
        fj * (UA * cosbj - VA),
        fj * (UA * sinbj),
        fk * (UA * cosbk - VA),
        fk * (UA * sinbk),
    ])
    B = np.array([
        fj * (UB * cosbj - VB),
        fj * (UB * sinbj),
        fk * (UB * cosbk - VB),
        fk * (UB * sinbk),
    ])

    # "unfurl"
    Z = np.array([Z[0].real, Z[0].imag, Z[1].real, Z[1].imag]) / N

    AB = A@B
    AZ = A@Z
    BZ = B@Z
    AA = A@A
    BB = B@B

    d = AA*BB - AB*AB
    ca = (BB*AZ - AB*BZ) / d
    cb = (AA*BZ - AB*AZ) / d

    amplitude = np.sqrt(ca**2 + cb**2)
    phase = np.arctan2(-cb, ca)
    return amplitude, phase


### Frequency case ###########################################################
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


def est_f_cedron_3bin(x):
    N = len(x)
    xf = rfft(x)
    kmax = np.argmax(abs(xf[1:-1])) + 1
    Z = xf[kmax-1:kmax+2]

    f = _est_f_cedron_3bin(Z, kmax, N)
    return f


def _est_f_cedron_3bin(Z, k, N):
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


def est_f_cedron_2bin(x):
    N = len(x)
    xf = rfft(x)

    if N == 3:
        # not necessarily optimal scheme
        kmax = 1
        Z = xf
    else:
        kmax = np.argmax(np.abs(xf))
        Z = xf[kmax-1:kmax+1]

    f = _est_f_cedron_2bin(Z, kmax, N)
    return f


def _est_f_cedron_2bin(Z, k, N):
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
