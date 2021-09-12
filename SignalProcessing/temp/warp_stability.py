import numpy as np
import os
import matplotlib.pyplot as plt
from kymatio.torch import Scattering1D
from kymatio.visuals import imshow, plot, make_gif
from kymatio.toolkit import _echirp_fn, l2

savedir = r"C:\Desktop\School\Deep Learning\DL_Code\signals\warp\\"

#%%# Define helpers ##########################################################
def tau(t, K=.08):
    return np.cos(2*np.pi * t**2) * K

def adtau(t, K=.08):
    return np.abs(np.sin(2*np.pi * t**2) * t * 4*np.pi*K)

def viz_x(x_all):
    # unwarped and max warped
    kw = dict(xlabel="time [sec]", xticks=t, show=1)
    plot(x_all[0],  **kw, title="x(t), unwarped")
    plot(x_all[-1], **kw, title="x(t), max warp")

    # all warps
    imshow(x_all, **kw, title="x(t), all warps",
           ylabel="max(|tau'|)", yticks=adtau_max_all)

def scatter_x(x_all, ts):
    meta = ts.meta()
    Scx_all = [ts(x) for x in x_all]
    Scx_all0 = np.vstack([(np.vstack([c['coef'].cpu().numpy() for c in Scx]
                                     )[meta['order'] == 1])[None]
                          for Scx in Scx_all])
    freqs = N * meta['xi'][meta['order'] == 1][:, 0]
    return Scx_all0, freqs

def make_warp_gif(Scx_all, freqs, base_name="warp", images_ext=".jpg",
                  overwrite=True, duration=250):
    norm = (0, Scx_all.max() * .9)

    for i in range(n_pts):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        kw = dict(abs=1, show=0, norm=norm)

        title="|CWT(x(t))|"
        imshow(Scx_all[i], title="|CWT(x(t))|", ax=axes[0], **kw,
               yticks=freqs, xticks=t,
               xlabel="time [sec]", ylabel="frequency [Hz]")
        if i == 0:
            title = r"$x(t) \ ... \ |\tau'(t)| = 0$"
        else:
            title = r"$x(t) \ ... \ |\tau'(t)| < %.3f$" % adtau_max_all[i]
        plot(x_all[i], ax=axes[1], show=0, ticks=(1, 0), xticks=t, title=title)

        plt.subplots_adjust(wspace=.02)

        path = os.path.join(savedir, f'{base_name}{i}{images_ext}')
        if os.path.isfile(path) and overwrite:
            os.unlink(path)
        if not os.path.isfile(path):
            plt.savefig(path, bbox_inches='tight')
        plt.close()

    savepath = os.path.join(savedir, 'warp.gif')
    make_gif(loaddir=savedir, savepath=savepath, ext=images_ext,
             duration=duration, overwrite=overwrite, delimiter=base_name,
             verbose=1, start_end_pause=7)

#%%## Set params & create scattering object ##################################
N, f = 2048, 64
J, Q = 8, 8
# make CWT
average, oversampling = False, 999
ts = Scattering1D(shape=N, J=J, Q=Q, average=average, oversampling=oversampling,
                  out_type='list', max_order=1)
ts.cuda()
meta = ts.meta()

#%%# Create signal & warp it #################################################
# guess then adjust
K_init = .01
# number of warps
n_pts = 64

# compute actual quality factor
p = ts.psi1_f[0]
QF = p['xi'] / p['sigma']

t = np.linspace(0, 1, N, 0)
# initial guess
adtau_init_max = adtau(t, K=K_init).max()
# compute min as the safely stable deformation, max as `|tau'| < 1`
K_min = (1/QF) / 10
K_max = (1 / adtau_init_max) * K_init

# logspace uniformly samples orders of magnitude
K_all = np.logspace(np.log10(K_min), np.log10(K_max), n_pts - 1, 1)
# no warp for reference
K_all = np.hstack([0, K_all])
tau_all = np.vstack([tau(t, K=k) for k in K_all])
adtau_max_all = np.vstack([adtau(t, K=k).max() for k in K_all])
# ensure no `|tau'| >= 1`
assert adtau_max_all.max() < 1, adtau_max_all.max()

#%%# Scatter, visualize ######################################################
def warp_demo(x_all, ts, name):
    # visualize warps
    viz_x(x_all)

    # scatter all
    Scx_all, freqs = scatter_x(x_all, ts)

    # plot Euclidean distances between warped and unwarped
    dists = l2(Scx_all[1:], Scx_all[:1], axis=(1, 2))
    plot(dists, ylims=(0, None), show=1)

    # make gif
    make_warp_gif(Scx_all, freqs, duration=250, base_name=f"{name}_warp")
    make_warp_gif(Scx_all, freqs, duration=50,  base_name=f"{name}_warp_fast")

#%%# sinusoid
x_all = np.cos(2*np.pi * f * (t - tau_all))
warp_demo(x_all, ts, "sine")

#%% exponential chirp
x_all = np.cos(_echirp_fn(fmin=64, fmax=N/8)(t - tau_all))
warp_demo(x_all, ts, "echirp")
