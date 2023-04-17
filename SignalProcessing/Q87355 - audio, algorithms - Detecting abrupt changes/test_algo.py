# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87355/50076
# USER CONFIGS
TOLERANCE = 0.05
VIZ_PREDS = 1
GPU = 0
FULL_GPU = 0
PRINT_TIMES = 1

import os
os.environ['SSQ_GPU'] = '1' if GPU else '0'
os.environ['FULL_GPU'] = '1' if FULL_GPU else '0'

import numpy as np
from utils87355 import (
    find_audio_change_timestamps, handle_wavelet, pad_input_make_t,
    load_data, data_labels, data_dir,
)
from timeit import default_timer as dtime

#%%############################################################################
# Configure
# ---------
cfg = dict(
    wavelet='freq',
    stft_prefilter_scaling=1,
    carve_th_scaling=1/3,
    fmax_idx_frac=400/471,
    silence_th_scaling=50,
    final_pred_n_peaks=1,
    escaling=(1, 1, 1),
)

#%%############################################################################
# Make reusables
# --------------
Nmax = max(len(load_data(i)[0]) for i in range(len(data_labels)))
fs = load_data(0)[1]  # assumes same for all
Mmax = len(pad_input_make_t(np.arange(Nmax), fs)[0])
N_ref = Nmax
reusables = handle_wavelet(
    wavelet=cfg['wavelet'],
    fs=fs,
    M=Mmax,
    ssq_precfg=dict(scales='log', padtype=None),
    fmax_idx_frac=cfg['fmax_idx_frac'],
    silence_interval_samples=int(.2*fs),
    escaling=cfg['escaling'],
)
other_cfg = dict(reusables=reusables, fs=fs, N_ref=N_ref)

#%%############################################################################
# Run
# ---
results_train = {'preds': []}
results_test = {'preds': []}

idxs_train = tuple(range(0, 4))
idxs_test = tuple(range(4, len(data_labels)))
idxs_all = idxs_train + idxs_test

# get predictions
for example_idx in idxs_all:
    # load data
    x, fs, labels = load_data(example_idx)

    # make predictions
    viz_labels = ((example_idx, labels) if VIZ_PREDS else
                  None)
    if PRINT_TIMES:
        t0 = dtime()
    preds = find_audio_change_timestamps(
        x, **other_cfg, **cfg, viz_labels=viz_labels)[0]
    if PRINT_TIMES:
        print("%.3g" % (dtime() - t0) + " sec", flush=True)
    else:
        print(end=".", flush=True)

    # append
    d = (results_train if example_idx in idxs_train else
         results_test)
    d['preds'].append(preds)

#%%############################################################################
# Calculate score
# ---------------
def is_success(preds, label, tolerance=TOLERANCE):
    if not isinstance(preds, (list, tuple)):
        preds = [preds]
    return any(abs(p - label) < tolerance for p in preds)

for d in (results_train, results_test):
    for k in ('scores', 'scores_flattened'):
        d[k] = []

tolerance = TOLERANCE

printed_line = False
for example_idx in idxs_all:
    d = (results_train if example_idx in idxs_train else
         results_test)
    pred_idx = (example_idx if example_idx in idxs_train else
                example_idx - len(idxs_train))
    preds = d['preds'][pred_idx]
    labels = load_data(example_idx)[-1]

    if len(labels) == 2:
        score0 = is_success(preds, labels[0], tolerance)
        score1 = is_success(preds, labels[1], tolerance)
        scores_packed = (score0, score1)
    else:
        score0 = is_success(preds[0], labels[0], tolerance)
        scores_packed = (score0,)

    # append
    d['scores'].append(scores_packed)
    d['scores_flattened'].extend(scores_packed)
    # print(end=".", flush=True)

    if example_idx not in idxs_train and not printed_line:
        print("="*80)
        printed_line = True

    print()
    preds = sorted([float("%.3g" % p) for p in preds])
    print(tuple(np.array(list(scores_packed)).astype(int)), '--', example_idx)
    print(tuple(preds))
    print(tuple(labels))

# finalize
accuracy_train = np.mean(results_train['scores_flattened'])
accuracy_test = np.mean(results_test['scores_flattened'])

print("Accuracy (train, test): {:.3f}, {:.3f}".format(
    accuracy_train, accuracy_test))

#%%
# from ? import make_gif
# from pathlib import Path
# make_gif(data_dir, str(Path(data_dir, "preds.gif")),
#          duration=1000, overwrite=True, delimiter="im", HD=True, verbose=True)

#%%############################################################################
# Last run's output
# -----------------
"""
(1, 1) -- 0
(1.56, 3.54)
(1.55, 3.5)

(1, 1) -- 1
(1.22, 1.74)
(1.21, 1.74)

(1, 1) -- 2
(0.958, 1.57)
(0.94, 1.57)

(1, 1) -- 3
(1.42, 1.87)
(1.42, 1.85)
================================================================================

(1,) -- 4
(0.783, 1.12)
(0.76,)

(1, 1) -- 5
(0.818, 1.69)
(0.79, 1.68)

(1, 1) -- 6
(2.89, 3.3)
(2.87, 3.28)

(0,) -- 7
(0.488, 0.738)
(0.75,)

(1,) -- 8
(0.623, 1.77)
(0.63,)

(1,) -- 9
(0.211, 0.468)
(0.46,)

(1,) -- 10
(4.69, 4.97)
(4.97,)

(1,) -- 11
(0.543, 0.956)
(0.94,)

(1, 1) -- 12
(2.35, 3.0)
(2.37, 2.99)

(1, 0) -- 13
(2.06, 2.44)
(2.03, 2.73)
Accuracy (train, test): 1.000, 0.857
"""
