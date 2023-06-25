# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87926/50076
import numpy as np

def _get_UVk(N, f, phi):
    assert not float(f).is_integer(), f
    k = np.arange(N)
    U = np.cos(2*np.pi*f + phi) - np.cos(phi)
    V = np.cos(2*np.pi*f + phi - 2*np.pi*f/N) - np.cos(phi - 2*np.pi*f/N)
    return U, V, k


def sine_dft(N, f, phi):
    """Solution by Cedron Dawg https://www.dsprelated.com/showarticle/771.php
    """
    if float(f).is_integer():
        return _sine_dft_int(N, f, phi)
    return _sine_dft_frac(N, f, phi)

def sine_dft_modulus(N, f, phi):
    """Modulus analysis & insights by John Muradeli # TODO URL
    """
    if float(f).is_integer():
        return _sine_dft_modulus_int(N, f, phi)
    return _sine_dft_modulus_frac(N, f, phi)

def _sine_dft_frac(N, f, phi):
    U, V, k = _get_UVk(N, f, phi)

    n = U*np.exp(1j*2*np.pi*k/N) - V
    d = np.cos(2*np.pi*f/N) - np.cos(2*np.pi*k/N)
    out = 1/2 * n / d
    return out

def _sine_dft_modulus_frac(N, f, phi, get_params=False):
    U, V, k = _get_UVk(N, f, phi)

    n = np.sqrt(U**2 + V**2 - 2*U*V*np.cos(2*np.pi*k/N))
    d = abs(np.cos(2*np.pi*f/N) - np.cos(2*np.pi*k/N))
    out = 1/2 * n / d

    if get_params:
        return out, (U, V)
    return out

def _delta(x):
    """Kronecker Delta (discrete unit impulse). Handles numeric precision."""
    x = np.asarray(x)
    eps = (0 if x.dtype.name.startswith('int') else
           np.finfo(x.dtype).eps)
    return (abs(x) <= eps).astype(float)

def _sine_dft_int(N, f, phi):
    k = np.arange(N)
    a = np.exp(1j*phi ) * _delta(np.mod(k - f, N))
    b = np.exp(-1j*phi) * _delta(np.mod(k + f, N))
    return (N/2) * (a + b)

def _sine_dft_modulus_int(N, f, phi):
    k = np.arange(N)
    dright = _delta(np.mod(k - f, N))
    dleft  = _delta(np.mod(k + f, N))

    return (N/2) * np.sqrt(dleft + dright + 2*np.cos(2*phi)*dleft*dright)


def sine_stft(N, M, H, f, phi):
    assert M <= N and 1 <= H <= N, (N, M, H)
    n_hops = (N - M)//H + 1
    out = np.zeros((M, n_hops), dtype='complex128')

    for tau in range(n_hops):
        phi_tau = phi + 2*np.pi*f*tau*H/N
        out[:, tau] = sine_dft(M, f*M/N, phi_tau)
    return out
