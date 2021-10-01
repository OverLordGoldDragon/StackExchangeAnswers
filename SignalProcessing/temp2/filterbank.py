# -*- coding: utf-8 -*-
"""Visualize joint JTFS filterbank (2D)."""
from kymatio.numpy import TimeFrequencyScattering1D
from kymatio.visuals import make_gif
from kymatio import visuals
import matplotlib.pyplot as plt

#%% Viz frequential filterbank (1D) ##########################################
jtfs = TimeFrequencyScattering1D(shape=512, J=7, Q=(16, 1), J_fr=5, Q_fr=2, F=8,
                                 normalize='l1')
visuals.filterbank_jtfs_1d(jtfs, zoom=-1, lp_sum=1)

#%% Show joint wavelets on a smaller filterbank ##############################
jtfs = TimeFrequencyScattering1D(shape=512, J=4, Q=(16, 1), J_fr=3, Q_fr=1, F=8)

#%%
# nearly invisible, omit (also unused in scattering per `j2 > 0`)
jtfs.psi2_f.pop(0)
# not nearly invisible but still makes plot too big
jtfs.psi2_f.pop(0)
jtfs.psi1_f_fr_up.pop(0)
jtfs.psi1_f_fr_down.pop(0)

#%% Make GIF ################################################################
base_name = 'filterbank'
for i, part in enumerate(('real', 'imag', 'complex')):
    fig, axes = visuals.filterbank_jtfs_2d(jtfs, part=part, labels=1,
                                        suptitle_y=1.03)
    # fig.show()
    # break
    fig.savefig(f'{base_name}{i}.png', bbox_inches='tight')
    plt.close(fig)

make_gif('', f'{base_name}.gif', duration=2000, start_end_pause=0,
          delimiter=base_name, overwrite=1)
