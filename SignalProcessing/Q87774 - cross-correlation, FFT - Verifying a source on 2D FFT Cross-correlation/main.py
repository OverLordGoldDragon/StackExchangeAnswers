# -*- coding: utf-8 -*-
# https://dsp.stackexchange.com/q/87774/50076
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from numpy.fft import fft2, ifft2, ifftshift
from PIL import Image
from matplotlib.patches import Circle

def rand(*s, ival=True):
    return ((1j*np.random.randn(*s)) if ival else
            np.random.randn(*s))

def cross_correlate_2d(x, h):
    h = ifftshift(ifftshift(h, axes=0), axes=1)
    return ifft2(fft2(x) * np.conj(fft2(h)))

def load_image(path):
    img = np.array(Image.open(path).convert("L")) / 255.
    img[img==1] = 0
    return img

def imshow(x, title=None, show_argmax=False, fig=None, ax=None, show=True):
    xa = np.abs(x)
    if fig is None:
        fig, ax = plt.subplots()
    ax.imshow(xa)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, weight='bold', loc='left', fontsize=20)
    if show_argmax:
        hmax, wmax = np.where(xa == xa.max())
        size = len(x) // 7 // 2
        circ = Circle((wmax, hmax), size, fill=False, color='tab:red',
                      linewidth=2)
        ax.add_patch(circ)
    if show:
        plt.show()


def run_example(x, h, show_h=False):
    # compute
    out0 = scipy.signal.correlate2d(x, h)
    out1 = cross_correlate_2d(x, h)

    # plot
    if show_h:
        fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.7), layout='constrained')
        imshow(x, "|x|, |h|", fig=fig, ax=axes[0], show=False,)
        imshow(h, fig=fig, ax=axes[1])
    else:
        imshow(x, "|x|; x = image + iWGN/5")

    nm = "x, x" if not show_h else "x, h"
    imshow(out0, f"|scipy.signal.correlate2d({nm})|", show_argmax=True)
    imshow(out1, f"|cross_correlate_2d({nm})|", show_argmax=True)


#%% Example: self-cc #########################################################
# load image as greyscale
np.random.seed(0)
x = load_image("covid.png")

# subsample & add noise
x = x[::4, ::4]
x = x + rand(*x.shape) / 5

run_example(x, x)

#%% Example: flipped false positive ##########################################
np.random.seed(0)
x = load_image("covid_target.png")
h = load_image("covid_template.png")

x, h = x[::9, ::9], h[::9, ::9]
x = x + rand(*x.shape) / 10
h = h + rand(*h.shape) / 10

run_example(x, h, show_h=True)

#%% Example: conjugated false positive########################################
np.random.seed(0)
x = load_image("covid_target2.png")
h = load_image("covid_template.png")

x, h = 1j*x[::9, ::9], 1j*h[::9, ::9]
M, N = x.shape
x[:, N//2:] = np.conj(x[:, N//2:])

x = x + rand(*x.shape) / 10 * 1j
h = h + rand(*h.shape) / 10 * 1j

run_example(x, h, show_h=True)
