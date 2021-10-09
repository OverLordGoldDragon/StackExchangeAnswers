# -*- coding: utf-8 -*-
"""Visualize JTFS of pure sine: as GIF of joint slices (2D)."""
# https://dsp.stackexchange.com/q/78622/50076 ################################
import numpy as np
from kymatio.numpy import TimeFrequencyScattering1D
from kymatio.toolkit import energy
from kymatio.visuals import plot, imshow
from kymatio import visuals

#%% Generate sine and create scattering object ###############################
# `pow2 + 1` with `endpoint==1` to pad to perfect sine
N = 4097
x = np.cos(2*np.pi * (N//16) * np.linspace(0, 1, N, 1))

# 9 temporal octaves
# largest scale is 2**9 [samples] / 4096 [samples / sec] == 125 ms
J = 9
# 8 bandpass wavelets per octave
# J*Q ~= 144 total temporal coefficients in first-order scattering
Q = 16
# scale of temporal invariance, 31.25 ms
T = 2**7
# 4 frequential octaves
J_fr = 4
# 2 bandpass wavelets per octave
Q_fr = 2
# scale of frequential invariance, F/Q == 0.5 cycle per octave
F = 8
# do frequential averaging to enable 4D concatenation
average_fr = True
# frequential padding; 'zero' avoids few discretization artefacts for this example
pad_mode_fr = 'zero'
# return packed as dict keyed by pair names for easy inspection
out_type = 'dict:array'

params = dict(J=J, Q=Q, T=T, J_fr=J_fr, Q_fr=Q_fr, F=F, average_fr=average_fr,
              out_type=out_type, pad_mode_fr=pad_mode_fr, max_pad_factor=4,
              max_pad_factor_fr=4, oversampling=999, oversampling_fr=999)
jtfs = TimeFrequencyScattering1D(shape=N, **params)

#%% Take JTFS, print pair names and shapes ###################################
Scx = jtfs(x)
print("JTFS pairs:")
for pair in Scx:
    print("{:<12} -- {}".format(str(Scx[pair].shape), pair))

E_up = energy(Scx['psi_t * psi_f_up'])
E_dn = energy(Scx['psi_t * psi_f_down'])
print("E_down / E_up = {:.1f}".format(E_dn / E_up))
#%% Show `x` and its (time-averaged) scalogram ###############################
plot(x, show=1, w=.7,
     xlabel="time [samples]",
     title="Pure sine | fmin=64, fmax=2048, 4096 samples")
#%%
freqs = jtfs.meta()['xi']['S1'][:, -1]
imshow(Scx['S1'].squeeze(), abs=1, w=.8, h=.67, yticks=freqs,
       xlabel="time [samples]",
       ylabel="frequency [frac of fs]",
       title="Scalogram, time-averaged (first-order scattering)")

#%% Create & save GIF ########################################################
# fetch meta (structural info)
jmeta = jtfs.meta()
# specify save folder
savedir = ''
# time between GIF frames (ms)
duration = 150
visuals.gif_jtfs_2d(Scx, jmeta, savedir=savedir, base_name='jtfs_sine',
                    save_images=0, overwrite=1, gif_kw={'duration': duration})
