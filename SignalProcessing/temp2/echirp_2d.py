# -*- coding: utf-8 -*-
"""
Visualize JTFS of exponential chirp: as GIF of coefficients
superimposed with wavelets.
"""
import numpy as np
from kymatio.numpy import TimeFrequencyScattering1D
from kymatio.toolkit import echirp, pack_coeffs_jtfs
from kymatio.visuals import make_gif
from numpy.fft import ifft, ifftshift
import matplotlib.pyplot as plt

#%% Generate echirp and create scattering object #############################
N = 4096
# span low to Nyquist; assume duration of 1 second
x = echirp(N, fmin=64, fmax=N/2)

#%% Show joint wavelets on a smaller filterbank ##############################
o = (0, 999)[1]
rp = np.sqrt(.5)
jtfs = TimeFrequencyScattering1D(shape=N, J=5, Q=(16, 1), J_fr=3, Q_fr=1,
                                 sampling_filters_fr='resample',
                                 average=0, average_fr=0, F=4,
                                 r_psi=(rp, .9*rp, rp), out_type='dict:list',
                                 oversampling=o, oversampling_fr=o)

#%% scatter
Scx_orig = jtfs(x)
jmeta = jtfs.meta()

#%% pack
Scx = pack_coeffs_jtfs(Scx_orig, jmeta, structure=2)
cmx = Scx.max() * .5  # color max

#%%
n_n2s = sum(p['j'] > 0 for p in jtfs.psi2_f)
n_n1_frs = len(jtfs.psi1_f_fr_up)
# drop spin up
Scx = Scx[:, n_n1_frs:]
# drop spin down
# Scx = Scx[:, :n_n1_frs + 1]

#%%
psis_down = jtfs.psi1_f_fr_down
psi2s = [p for p in jtfs.psi2_f if p['j'] > 0]
# reverse ordering
psi2s = psi2s[::-1]

# reverse order of fr wavelets
psis_down = psis_down[::-1]

# for spin down
c_psi_dn = Scx[1:, 1:]
c_phi_t  = Scx[0, 1:]
c_phi_f  = Scx[1:, 0]
c_phis   = Scx[0, 0]

# for spin up
# c_psi_dn = Scx[1:, :-1]
# c_phi_t  = Scx[0, :-1]
# c_phi_f  = Scx[1:, -1]
# c_phis   = Scx[0, -1]

#%% Visualize ################################################################
def no_border(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

def to_time(p_f):
    if isinstance(p_f, dict):
        p_f = p_f[0]
    if isinstance(p_f, list):
        p_f = p_f[0]
    return ifftshift(ifft(p_f.squeeze()))

spin_up = False

imshow_kw0 = dict(aspect='auto', cmap='bwr')
imshow_kw1 = dict(aspect='auto', cmap='turbo')

n_rows = len(psis_down) + 1
n_cols = len(psi2s) + 1
w = 13
h = 13 * n_rows / n_cols

fig0, axes0 = plt.subplots(n_rows, n_cols, figsize=(w, h))
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(w, h))

# compute common params to zoom on wavelets based on largest wavelet
pf_f = psis_down[0]
pt_f = psi2s[0]
# centers
ct = len(pt_f[0]) // 2
cf = len(pf_f[0]) // 2
# supports
st = int(pt_f['support'][0] / 1.5)
sf = int(pf_f['support'][0] / 1.5)

# coeff max
cmx = max(c_phi_t.max(), c_phi_f.max(), c_psi_dn.max()) * .8

# psi_t * psi_f_down
for n2_idx, pt_f in enumerate(psi2s):
    for n1_fr_idx, pf_f in enumerate(psis_down):
        pt = to_time(pt_f)
        pf = to_time(pf_f)
        # trim to zoom on wavelet
        pt = pt[ct - st:ct + st + 1]
        pf = pf[cf - sf:cf + sf + 1]

        Psi = pf[:, None] * pt[None]

        a = n1_fr_idx if spin_up else n1_fr_idx + 1
        ax0 = axes0[a, n2_idx + 1]
        ax1 = axes1[a, n2_idx + 1]

        mx = np.abs(Psi).max()
        ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)
        no_border(ax0)

        # coeffs
        c = c_psi_dn[n2_idx, n1_fr_idx]
        ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)
        no_border(ax1)


# psi_t * phi_f
phif = to_time(jtfs.phi_f_fr)
phif = phif[cf - sf:cf + sf + 1]
for n2_idx, pt_f in enumerate(psi2s):
    pt = to_time(pt_f)
    pt = pt[ct - st:ct + st + 1]
    Psi = phif[:, None] * pt[None]

    a = -1 if spin_up else 0
    ax0 = axes0[a, n2_idx + 1]
    ax1 = axes1[a, n2_idx + 1]

    mx = np.abs(Psi).max()
    ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)
    no_border(ax0)

    # coeffs
    c = c_phi_f[n2_idx]
    ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)
    no_border(ax1)

# phi_t * psi_f
phit = to_time(jtfs.phi_f)
phit = phit[ct - st:ct + st + 1]
for n1_fr_idx, pf_f in enumerate(psis_down):
    pf = to_time(pf_f)
    pf = pf[cf - sf:cf + sf + 1]

    Psi = pf[:, None] * phit[None]

    a = n1_fr_idx if spin_up else n1_fr_idx + 1
    ax0 = axes0[a, 0]
    ax1 = axes1[a, 0]

    mx = np.abs(Psi).max()
    ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)
    no_border(ax0)

    # coeffs
    c = c_phi_t[n1_fr_idx]
    ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)
    no_border(ax1)

# phi_t * phi_f
a = -1 if spin_up else 0
ax0 = axes0[a, 0]
ax1 = axes1[a, 0]

Psi = phif[:, None] * phit[None]
mx = np.abs(Psi).max()
ax0.imshow(Psi.real, **imshow_kw0, vmin=-mx, vmax=mx)
no_border(ax0)

# coeffs
c = c_phis
ax1.imshow(c, **imshow_kw1, vmin=0, vmax=cmx)
no_border(ax1)

fig0.subplots_adjust(wspace=.02, hspace=.02)
fig1.subplots_adjust(wspace=.02, hspace=.02)

base_name = 'jtfs_echirp_wavelets'
fig0.savefig(f'{base_name}0.png', bbox_inches='tight')
fig1.savefig(f'{base_name}1.png', bbox_inches='tight')
make_gif('', f'{base_name}.gif', duration=2000, start_end_pause=0,
         delimiter=base_name, overwrite=1)
