# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/76644/50076
import os
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import numpy as np
import matplotlib.pyplot as plt

# ensure the files can be found
import sys
from pathlib import Path
_dir = Path(__file__).parent
assert _dir.is_file() or _dir.is_dir(), str(_dir)
if not any(str(_dir).lower() == p.lower() for p in sys.path):
    sys.path.insert(0, str(_dir))

from optimized import Kay2Complex, Cedron3BinComplex
from utils76644 import make_x, timeit

#%%
kay2_complex_optimized = Kay2Complex()
cedron_3bin_complex_optimized = Cedron3BinComplex()
fns = {
  'kay_2': kay2_complex_optimized,
  'cedron_3bin': cedron_3bin_complex_optimized,
}

#%% Manual testing ###########################################################
np.random.seed(0)
name = ('cedron_3bin', 'kay_2')[1]
N = 100
f = N*0.153123
snr = 100
x = make_x(N, f, snr, real=False)

fn = fns[name]
f_est = fn(x)

print(f_est / (f/N), sep='\n')

#%% Benchmark ################################################################
np.random.seed(0)
times = {'cedron_3bin': {}, 'kay_2': {}}
n_trials = 400
repeats = 20
Npow2 = True

N_all = 2**np.arange(7, 21)
if not Npow2:
    for i, N in enumerate(N_all):
        N_all[i] = round(N, -(len(str(N)) - 2))

for N in N_all:
    x = np.random.randn(N) + 1j*np.random.randn(N)
    for name in times:
        fn = fns[name]
        # warmup
        for _ in range(10):
            fn(x)
        # bench
        times[name][N] = timeit(fn, x, n_trials, repeats)
    print(end='.', flush=True)

print(times)

#%% Visualize ################################################################
data = [np.array(list(times['cedron_3bin'].values())),
        np.array(list(times['kay_2'].values()))]
N_all = np.array(list(times['cedron_3bin']))

fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(9.5, 8))

ax.plot(   N_all, data[1] / data[0])
ax.scatter(N_all, data[1] / data[0], s=40)

ax.set_xscale('log')

ax.set_xlabel("N", size=20)
ax.set_ylabel("t_ratio", size=20)
title = ("Time ratios (Kay2_complex/Cedron_complex)\n"
         "n_trials={}, n_repeats={}\n"
         "Npow2={}, FFTW=False"
         ).format(n_trials, repeats, Npow2)
ax.set_title(title, weight='bold', fontsize=24, loc='left')
ax.set_ylim(0, 3)
ax.axhline(1, color='tab:red')

plt.show()
