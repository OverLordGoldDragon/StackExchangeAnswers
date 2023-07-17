# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76644/50076
# All code is written for readability, not performance.
import numpy as np
import matplotlib.pyplot as plt

# ensure the files can be found
import sys
from pathlib import Path
_dir = Path(__file__).parent
assert _dir.is_file() or _dir.is_dir(), str(_dir)
if not any(str(_dir).lower() == p.lower() for p in sys.path):
    sys.path.insert(0, str(_dir))

from estimators import est_freq, estimator_fns
from utils76644 import (
    make_x, run_test, run_test_multitone, snrs_db_practical, snrs_db_wide,
    run_viz, run_viz2, run_viz_multitone)

print("Available estimators:\n  " + "\n  ".join(list(estimator_fns)))

#%% Configurations ###########################################################
# USER -----------------------------------------------------------------------
# prints test progress
VERBOSE = 1
# other configs in `utils` file

# set certain defaults
f_N_all_nonints_large_N = (0.05393, 0.10696, 0.25494, 0.46595)
f_N_all_nonints_small_N = (0.053,   0.106,   0.254,   0.465)
f_N_all_ints_small_N    = (0.05,    0.10,    0.25,    0.46)
f_N_all_ints_large_N    = f_N_all_nonints_small_N

#%% Manual testing ###########################################################
np.random.seed(0)
name = ('cedron_3bin', 'kay_2', 'dft_quadratic')[0]
# name = ('cedron_3bin_complex',)[0]
N = 10000
f = N*0.053123
phi = 1
x = np.cos(2*np.pi * f * np.arange(N)/N + phi)
# x = x + 1j*np.sin(2*np.pi * f * np.arange(N)/N)
x += np.random.randn(N) * .1

print(est_freq(x, name) / (f/N), sep='\n')

#%% Full testing #############################################################
# configure
seed = 0
N = 100
n_trials = 2000
real = True
sweep_mode = ('practical', 'wide')[0]
name0, name1 = 'cedron_3bin', 'kay_2'
# name0, name1 = 'cedron_3bin', 'dft_quadratic'
# name0, name1 = 'cedron_3bin_complex', 'kay_2'
f_N_all = (f_N_all_nonints_small_N, f_N_all_nonints_large_N,
           f_N_all_ints_small_N,    f_N_all_ints_large_N)[0]
# f_N_all = np.linspace(1/N, .5-1/N, 50)
# f_N_all = np.linspace(-.5+1/N, .5-1/N, 100)

errs0_all, errs1_all, snrs, crlbs = run_test(
    f_N_all, N, n_trials, name0, name1, real, seed, sweep_mode, verbose=VERBOSE)

#%% Visualize ################################################################
names = ("Cedron", "Kay_2")
# names = ("Cedron", "DFT_quadratic")

args = (errs0_all, errs1_all, snrs, crlbs, f_N_all, N, n_trials, names)
run_viz(*args)
# run_viz2(*args)

#%% Multi-tone Example #######################################################
seed = 0
N = 10000
n_trials = 2000
name0, name1 = 'cedron_3bin', 'dft_quadratic'
# include integer case
f_N_all = (0.05305, 0.10605, 0.254, 0.46505)
# f_N_all = (0.10601, 0.10644, 0.10696, 0.10747)
A_all = (0.5, 0.8, 1.2, 1.5)  # mean=1
# each `A` will have different SNR, so extend the range so we can plot all
# under a common snr
snrs_bounds = (snrs_db_practical[0], snrs_db_practical[-1])
snrs = np.linspace(snrs_bounds[0] - 10, snrs_bounds[1] + 15,
                   int(len(snrs_db_practical)*1.25))

errs0_all, errs1_all, snrs_f_N, crlbs = run_test_multitone(
    f_N_all, A_all, N, n_trials, name0, name1, snrs, seed, verbose=VERBOSE)

#%% Visualize ################################################################
names = ("Cedron", "DFT_quadratic")
run_viz_multitone(errs0_all, errs1_all, snrs_f_N, crlbs,
                  f_N_all, A_all, N, n_trials, snrs_bounds, names=names)

#%% "Extreme" example ########################################################
N = 100000
f_N = .053056
snr = -30
f = f_N * N

errs = []
for _ in range(2000):
    x = make_x(N, f, snr)
    f_est = est_freq(x, 'cedron_3bin')
    errs.append((f_N - f_est)**2)
mse = np.mean(errs)

print(mse)

#%%
x, xo = make_x(N, f, snr, get_xo=True)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), layout='constrained')
axes[0].plot(xo[:500])
axes[0].set_title("Original signal, zoomed | N=100000",
                  weight='bold', loc='left', fontsize=24)
axes[1].plot(x[:500])
axes[1].set_title("Noisy, zoomed | SNR=1/1000 (-30dB)",
                  weight='bold', loc='left', fontsize=24)
