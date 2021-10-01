# -*- coding: utf-8 -*-
"""JTFS of Frequency-Dependent Time Shifts."""
import numpy as np
import torch
from kymatio.torch import Scattering1D, TimeFrequencyScattering1D
from kymatio.toolkit import fdts, l2
from kymatio.visuals import make_gif
import matplotlib.pyplot as plt

#%% Generate echirp and create scattering object #############################
N = 4096
f0 = N // 24
n_partials = 5
total_shift = N//12
seg_len = N//6

x, xs = fdts(N, n_partials, total_shift, f0, seg_len, partials_f_sep=1.7)

#%% Make scattering objects ##################################################
J = int(np.log2(N))  # max scale / global averaging
Q = (8, 2)
kw = dict(J=J, Q=Q, shape=N, max_pad_factor=4, pad_mode='zero')
cwt = Scattering1D(**kw, average=False, out_type='list', oversampling=999,
                   max_order=1)
ts = Scattering1D(out_type='array', max_order=2, **kw)
jtfs = TimeFrequencyScattering1D(Q_fr=2, J_fr=5, out_type='array', **kw,
                                 out_exclude=('S1', 'phi_t * psi_f',
                                              'phi_t * phi_f', 'psi_t * phi_f'),
                                 max_pad_factor_fr=4, pad_mode_fr='zero',
                                 sampling_filters_fr=('resample', 'resample'))
cwt, ts, jtfs = [s.cuda() for s in (cwt, ts, jtfs)]

#%% Scatter ##################################################################
cwt_x  = torch.vstack([c['coef'] for c in cwt(x)]).cpu().numpy()[1:]
cwt_xs = torch.vstack([c['coef'] for c in cwt(xs)]).cpu().numpy()[1:]

ts_x  = ts(x).cpu().numpy()
ts_xs = ts(xs).cpu().numpy()

jtfs_x  = jtfs(x).cpu().numpy()[0]
jtfs_xs = jtfs(xs).cpu().numpy()[0]

l2_ts   = float(l2(ts_x, ts_xs))
l2_jtfs = float(l2(jtfs_x, jtfs_xs))

print(("\nFDTS sensitivity:\n"
       "JTFS/TS = {:.1f}\n"
       "TS      = {:.4f}\n"
       "JTFS    = {:.4f}\n").format(l2_jtfs / l2_ts, l2_ts, l2_jtfs))

#%%# Make GIF ################################################################
data = {'x':    [x, xs],
        'cwt':  [cwt_x, cwt_xs],
        'ts':   [ts_x, ts_xs],
        'jtfs': [jtfs_x, jtfs_xs]}
xmx = max(abs(x).max(), abs(xs).max())*1.05
cmx = max(np.abs(cwt_x).max(), np.abs(cwt_xs).max())
tmx = max(np.abs(ts_x).max(), np.abs(ts_xs).max())
jmx = max(np.abs(jtfs_x).max(), np.abs(jtfs_xs).max()) * .95

for i in (0, 1):
    _x, _cwt, _ts, _jtfs = [data[k][i] for k in data]

    fig = plt.figure()
    fig = plt.figure(figsize=(14, 14))
    ax0 = fig.add_subplot(12, 2, (1, 9))
    ax1 = fig.add_subplot(12, 2, (2, 10))
    ax2 = fig.add_subplot(14, 2, (13, 14))
    ax3 = fig.add_subplot(14, 2, (15, 16))
    axes = [ax0, ax1, ax2, ax3]

    imshow_kw = dict(cmap='turbo', aspect='auto', vmin=0)
    s, e = 1700, len(x) - 1200
    tks = np.arange(s, e)
    ax0.plot(tks, _x[s:e])
    ax1.imshow(_cwt,    **imshow_kw, vmax=cmx)
    ax2.imshow(_ts.T,   **imshow_kw, vmax=tmx)
    ax3.imshow(_jtfs.T, **imshow_kw, vmax=jmx)

    # remove ticks
    for ax in (ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])
    ax0.set_yticks([])
    ax1.set_yticks([])

    # set common limits
    ax0.set_ylim(-xmx, xmx)

    fig.subplots_adjust(hspace=.25, wspace=.02)

    # reduce y-spacing between bottom two subplots
    yspace = .05
    pos_ax3 = ax3.get_position()
    pos_ax2 = ax2.get_position()
    ywidth = pos_ax3.y1 - pos_ax3.y0
    pos_ax3.y0 = pos_ax2.y0 - ywidth * yspace
    pos_ax3.y1 = pos_ax2.y0 - ywidth * (1 + yspace)
    ax3.set_position(pos_ax3)

    fig.savefig(f'jtfs_ts{i}.png', bbox_inches='tight')
    plt.close(fig)

make_gif('', 'jtfs_ts.gif', duration=1000, start_end_pause=0,
         delimiter='jtfs_ts', overwrite=1, verbose=1)
