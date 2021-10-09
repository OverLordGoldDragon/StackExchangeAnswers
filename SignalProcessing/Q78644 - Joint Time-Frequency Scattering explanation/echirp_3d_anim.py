# -*- coding: utf-8 -*-
"""Visualize JTFS of exponential chirp: as GIF of 3D slices unfolded over time."""
# https://dsp.stackexchange.com/q/78622/50076 ################################
from kymatio.numpy import TimeFrequencyScattering1D
from kymatio.toolkit import echirp, pack_coeffs_jtfs
from kymatio.visuals import gif_jtfs_3d

#%% Make scattering object ###################################################
N = 4096
jtfs = TimeFrequencyScattering1D(shape=N, J=8, Q=16, T=2**8, F=4, J_fr=4, Q_fr=2,
                                 max_pad_factor=None, max_pad_factor_fr=None,
                                 oversampling=1, out_type='dict:array',
                                 pad_mode_fr='zero', pad_mode='zero',
                                 average_fr=True, sampling_filters_fr='resample')
meta = jtfs.meta()

#%% Make echirp & scatter ####################################################
x = echirp(N, fmin=64, fmax=N/2)
Scx = jtfs(x)

#%% Make GIF with and without spinned ########################################
for separate_lowpass in (False, True):
    packed = pack_coeffs_jtfs(Scx, meta, structure=2, separate_lowpass=True,
                              sampling_psi_fr=jtfs.sampling_psi_fr)
    if separate_lowpass:
        packed = packed[0]
    packed = packed.transpose(-1, 0, 1, 2)  # time first

    nm = 'spinned' if separate_lowpass else 'full'
    base_name = f'jtfs3d_echirp_{nm}'
    gif_jtfs_3d(packed, base_name=base_name, overwrite=1, cmap_norm=.5)
