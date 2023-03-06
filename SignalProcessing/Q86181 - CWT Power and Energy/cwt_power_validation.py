# -*- coding: utf-8 -*-
# Answer to https://dsp.stackexchange.com/q/86181/50076
import numpy as np
from numpy.fft import fft, ifft

def E(x):
    return np.sum(np.abs(x)**2)

#%% Generate filterbank ######################################################
# show that it works with any filters
# note ES still works despite violating condition 3, but condition 3
# is still needed for valid interpretation (sum won't invert to signal)
psi_fs = np.random.randn(51, 256) + 1j*np.random.randn(51, 256)
# wavelets are zero-mean
psi_fs[:, 0] = 0
# "any" but still analytic here, need more code for "any any"
psi_fs[:, 129:] = 0
# nyquist imaginary part must be zero else inverse can't be real-valued
psi_fs[:, 128].imag = 0


#%% Compute "transfer functions" #############################################
ET_tfn = np.sum(np.abs(psi_fs)**2, axis=0)
ES_tfn = np.abs(np.sum(psi_fs, axis=0))**2

#%% Run tests ################################################################
for case in ('real', 'complex'):
    # generate signal
    M = 256
    x = np.random.randn(M)
    if case == 'complex':
        x = x + 1j * np.random.randn(M)
    xf = fft(x)

    # compute CWT
    out = ifft(xf * psi_fs)

    # invert
    x_inv = out.sum(axis=0)
    if case == 'real':
        x_inv = x_inv.real

    # compute energies via transfer fns
    xfe = np.abs(xf)**2
    ET = xfe * ET_tfn / M
    ES = xfe * ES_tfn / M
    if case == 'real':
        ES[1:M//2] /= 2
    ET, ES = ET.sum(), ES.sum()

    # assert agreement
    assert np.allclose(ET, E(out))
    assert np.allclose(ES, E(x_inv))
