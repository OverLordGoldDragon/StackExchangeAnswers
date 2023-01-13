# -*- coding: utf-8 -*-
# Answer to https://dsp.stackexchange.com/q/86181/50076
import numpy as np
from ssqueezepy import cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq

def E(x):
    return np.sum(np.abs(x)**2)

#%% Configure ################################################################
fs = 400             # (in Hz)  anything works
duration = 5         # (in sec) anything works
padtype = 'reflect'  # anything (supported) works

# get power between these frequencies
freq_min = 50   # (in Hz)
freq_max = 150  # (in Hz)

#%% Obtain transform #########################################################
# make signal & wavelet
# assume this is Amperes passing through 1 Ohm resistor; P = I^2*R, E = P * T
# check actual physical units for your specific application and adjust accordingly
np.random.seed(0)
x = np.random.randn(fs * duration)
wavelet = Wavelet()

# transform, get frequencies
Wx, scales = cwt(x, wavelet, padtype=padtype)
freqs = scale_to_freq(scales, wavelet, len(x), fs=fs, padtype=padtype)

# fetch coefficients according to `freq_min` and `freq_max`
Wx_spec = Wx[(freq_min < freqs) * (freqs < freq_max)]

#%% Normalization ############################################################
# See "Normalization" in the answer

# fetch wavelets in freq domain, compute ET & ES transfer funcs, fetch maxima
psi_fs = wavelet._Psih  # fetch wavelets in freq domain
ET_tfn = np.sum(np.abs(psi_fs)**2, axis=0)
ES_tfn = np.abs(np.sum(psi_fs, axis=0))**2
# real-valued case adjustment:
#   - ET since we operate on half of spectrum (half as many coeffs)
#   - ES since `.real` halves spectrum, quartering energy on real side;
#     note, for this to work right, the filterbank must be halved at Nyquist
#     by design (which should also be done for sake of temporal decay)
ET_adj = ET_tfn.max() / 2
ES_adj = ES_tfn.max() / 4

#%% Compute energy & power ###################################################
# compute energy & power (discrete)
ET_disc = np.sum(np.abs(Wx_spec)**2) / ET_adj
ES_disc = np.sum(np.abs(np.sum(Wx_spec, axis=0).real)**2) / ES_adj
PT_disc = ET_disc / len(x)
PS_disc = ES_disc / len(x)

# compute energy & power (physical); estimate underlying continuous waveform via
# Riemann integration
sampling_period = 1 / fs
ET_phys = ET_disc * sampling_period * duration
ES_phys = ES_disc * sampling_period * duration
PT_phys = ET_phys / duration
PS_phys = ES_phys / duration

# repeat for original signal
Ex_disc = E(x)
Px_disc = Ex_disc / len(x)
Ex_phys = Ex_disc * sampling_period * duration
Px_phys = Ex_phys / duration

#%% Report ###################################################################
print(("Between {:d} and {:d} Hz, DISCRETE:\n"
       "{:<9.6g} -- energy of transform\n"
       "{:<9.6g} -- energy of signal\n"
       "{:<9.6g} -- power of transform\n"
       "{:<9.6g} -- power of signal\n"
       ).format(freq_min, freq_max, ET_disc, ES_disc, PT_disc, PS_disc))

print(("Between {:d} and {:d} Hz, PHYSICAL (via Riemann integration):\n"
       "{:<9.6g} Joules -- energy of transform\n"
       "{:<9.6g} Joules -- energy of signal\n"
       "{:<9.6g} Watts  -- power of transform\n"
       "{:<9.6g} Watts  -- power of signal\n"
       ).format(freq_min, freq_max, ET_phys, ES_phys, PT_phys, PS_phys))

print(("Original signal:\n"
       "{:<9.6g} -- energy (discrete)\n"
       "{:<9.6g} -- power  (discrete)\n"
       "{:<9.6g} Joules -- energy (physical)\n"
       "{:<9.6g} Watts  -- power  (physical)\n").format(
           Ex_disc, Px_disc, Ex_phys, Px_phys))
