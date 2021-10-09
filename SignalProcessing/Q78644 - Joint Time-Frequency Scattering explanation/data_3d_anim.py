# -*- coding: utf-8 -*-
"""Visualize JTFS of real data: as GIF of 3D slices unfolded over time."""
# https://dsp.stackexchange.com/q/78622/50076 ################################
import numpy as np
import torch
import librosa
from timeit import default_timer as dtime

from kymatio import TimeFrequencyScattering1D
from kymatio.toolkit import pack_coeffs_jtfs, jtfs_to_numpy, normalize
from kymatio.visuals import gif_jtfs_3d, make_gif

#%% load data ################################################################
x, sr = librosa.load(librosa.ex('trumpet'))
x = x[:81920]

#%% create scattering object, move to GPU ####################################
N = len(x)
J = int(np.log2(N) - 3)
Q = (16, 1)
Q_fr = 2
J_fr = 4
T = 2**(J - 4)
F = 4

jtfs = TimeFrequencyScattering1D(shape=N, J=J, J_fr=J_fr, Q=Q, Q_fr=Q_fr,
                                 T=T, F=F, average_fr=True,
                                 max_pad_factor=None, max_pad_factor_fr=None,
                                 out_type='dict:array', frontend='torch')
jmeta = jtfs.meta()
for a in ('N_frs', 'J_pad_frs'):
    print(getattr(jtfs, a), '--', a)
jtfs.cuda()
torch.cuda.empty_cache()

#%% scatter ##################################################################
t0 = dtime()
Scxt = jtfs(x)
Scx = jtfs_to_numpy(Scxt)

#%% pack
packed = pack_coeffs_jtfs(Scx, jmeta, structure=2, separate_lowpass=True,
                          sampling_psi_fr=jtfs.sampling_psi_fr)
packed_spinned = packed[0]
packed_spinned = packed_spinned.transpose(-1, 0, 1, 2)

#%% normalize
s = packed_spinned.shape
packed_spinned = normalize(packed_spinned.reshape(1, s[0], -1)).reshape(*s)

#%% make smooth camera transition ############################################
packed_viz = packed_spinned
print(packed_viz.shape)
n_pts = len(packed_viz)
extend_edge = int(.35 * n_pts)

def gauss(n_pts, mn, mx, width=20):
    t = np.linspace(0, 1, n_pts)
    g = np.exp(-(t - .5)**2 * width)
    g *= (mx - mn)
    g += mn
    return g

x = np.logspace(np.log10(2.5), np.log10(8.5), n_pts, endpoint=1)
y = np.logspace(np.log10(0.3), np.log10(6.3), n_pts, endpoint=1)
z = np.logspace(np.log10(2.0), np.log10(2.0), n_pts, endpoint=1)

x, y, z = [gauss(n_pts, mn, mx) for (mn, mx)
           in [(2.5, 8.5), (0.3, 6.3), (2, 2)]]

eyes = np.vstack([x, y, z]).T
assert len(x) == len(packed_viz), (len(x), len(packed_viz))

#%% Make gif  ################################################################
t0 = dtime()
gif_jtfs_3d(packed_viz, base_name='jtfs3d_trumpet', images_ext='.png',
            overwrite=1, save_images=1, angles=eyes, gif_kw=dict(duration=50))
print(dtime() - t0)

#%% remake if needed
# make_gif('', 'etc.gif', duration=33, delimiter='seiz3d', ext='.jpg',
#          overwrite=1, HD=True)
