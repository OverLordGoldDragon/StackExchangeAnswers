# -*- coding: utf-8 -*-
"""Scalogram gradient-based reconstruction on exponential chirp GIF."""
# https://dsp.stackexchange.com/q/78512/50076 ################################
# https://dsp.stackexchange.com/q/78514/50076 ################################
import numpy as np
import torch
from kymatio import Scattering1D
from kymatio.toolkit import echirp, l2, l1
from kymatio.visuals import plot
from ssqueezepy import ssq_cwt, Wavelet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%# Make scattering object ##################################################
J = 6
Q = 8
N = 2048
ts = Scattering1D(J, N, Q, frontend='torch', out_type='list', oversampling=999,
                  average=False, max_pad_factor=2, max_order=1).to(device)

def sc(x):
    return torch.vstack([c['coef'] for c in ts(x)])

#%%# Configure training ######################################################
n_iters = 100
loss_switch_iter = 50
y = torch.from_numpy(echirp(N, fmin=1).astype('float32')).to(device)
Sy = sc(y)
div = Sy.max()
Sy /= div

torch.manual_seed(0)
x = torch.randn(N, device=device)
x /= torch.max(torch.abs(x))
x.requires_grad = True
optimizer = torch.optim.SGD([x], lr=500, momentum=.9, nesterov=True)
loss_fn = torch.nn.MSELoss()
dist_fn = l2

losses, losses_recon = [], []
x_recons = []
lrs = []
for i in range(n_iters):
    optimizer.zero_grad()
    Sx = sc(x)
    Sx /= div
    loss = loss_fn(Sx, Sy)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu().numpy()))
    xn, yn = x.detach().cpu().numpy(), y.detach().cpu().numpy()
    losses_recon.append(float(dist_fn(yn, xn)))
    x_recons.append(xn)

    if i > loss_switch_iter:
        loss_fn = torch.nn.L1Loss()
        dist_fn = l1
        if i == loss_switch_iter + 1 or i % 5 == 0:
            factor = 2 if i == 51 else 2
            for g in optimizer.param_groups:
                g['lr'] /= factor
    lrs.append(optimizer.param_groups[0]['lr'])

th, th_recon, th_end_ratio = 1e-5, 1.05, 50
end_ratio = losses[0] / losses[-1]


kw = dict(show=1)
plot(np.log10(losses), **kw)
plot(np.log10(losses_recon), **kw)
plot(xn, show=1)

print(("\nReconstruction (torch):\n(end_start_ratio, min_loss, "
       "min_loss_recon) = ({:.1f}, {:.2e}, {:.6f})").format(
           end_ratio, min(losses), min(losses_recon)))

#%% take SSQ of every reconstruction
wavelet = Wavelet(('gmw', {'gamma': 1, 'beta': 1}))
x_recons = np.array(x_recons)
ssq_x_recon = np.abs(ssq_cwt(x_recons, wavelet, scales='log')[0])

#%%# Animate #################################################################
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PlotImshowAnimation(animation.TimedAnimation):
    def __init__(self, imshow_frames, plot_frames0, plot_frames1, plot_frames2):
        self.imshow_frames = imshow_frames
        self.plot_frames0 = plot_frames0
        self.plot_frames1 = plot_frames1
        self.plot_frames2 = plot_frames2
        self.xticks = np.arange(len(plot_frames1))

        self.n_repeats_total = n_repeats * repeat_first
        self.n_frames = len(imshow_frames) + self.n_repeats_total - repeat_first

        self.title_kw = dict(weight='bold', fontsize=15, loc='left')
        self.label_kw = dict(weight='bold', fontsize=14, labelpad=3)
        self.txt_kw = dict(x=0, y=1.017, s="", ha="left", weight='bold')

        fig, axes = plt.subplots(2, 2, figsize=(18/1.5, 9))

        # plots ##############################################################
        ax = axes[0, 0]
        ax.plot(self.plot_frames0[0]*1.03)
        ax.set_xlim(-30, 2078)
        ax.set_title("x_reconstructed", **self.title_kw)
        ax.set_yticks([-1, -.5, 0, .5, 1])
        ax.set_yticklabels([r'$\endash 1.0$', r'$\endash 0.5$', '0',
                            '0.5', '1.0'])
        self.lines0 = [ax.lines[-1]]

        # imshows ############################################################
        ax = axes[0, 1]
        # color norm
        mx = np.max(imshow_frames) * .5
        im = ax.imshow(self.imshow_frames[0], cmap='turbo', animated=True,
                       aspect='auto', vmin=0, vmax=mx)
        self.ims1 = [im]
        ax.set_title("|ssq_cwt(x_reconstructed)|", **self.title_kw)
        ax.set_yticks([])

        # plots ##############################################################
        ax = axes[1, 0]
        ax.plot(self.xticks, self.plot_frames1)
        ax.set_xlabel("n_iters", **self.label_kw)
        ax.set_yticks([0, -1, -2, -3, -4, -5])
        ax.set_yticklabels(['0'] + [rf'$\endash {n}$' for n in (1, 2, 3, 4, 5)])
        ax.set_ylim([-5.75, 0])

        self.lines2 = [ax.lines[-1]]
        self.lines2[0].set_data(self.xticks[0], self.plot_frames1[0])
        self.txt2 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        ax = axes[1, 1]
        ax.plot(self.xticks, self.plot_frames2)
        ax.set_xlabel("n_iters", **self.label_kw)
        ax.set_yticks([])

        self.lines3 = [ax.lines[-1]]
        self.lines3[0].set_data(self.xticks[0], self.plot_frames2[0])
        self.txt3 = ax.text(transform=ax.transAxes, **self.txt_kw, fontsize=15)

        # finalize #######################################################
        fig.subplots_adjust(left=.035, right=.99, bottom=.058, top=.96,
                            wspace=.02, hspace=.15)
        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, frame_idx):
        if frame_idx == 0:
            self.loss_idx = 0
            self.prev_loss_idx = 0
        elif frame_idx % n_repeats == 0 or frame_idx > self.n_repeats_total:
            self.loss_idx += 1

        if self.loss_idx == self.prev_loss_idx:
            return
        self.prev_loss_idx = self.loss_idx
        frame_idx = self.loss_idx  # adjusted

        self.lines0[0].set_ydata(self.plot_frames0[frame_idx])
        self.ims1[0].set_array(self.imshow_frames[ frame_idx])
        self.lines2[0].set_data(self.xticks[:frame_idx],
                                self.plot_frames1[:frame_idx])
        self.lines3[0].set_data(self.xticks[:frame_idx],
                                self.plot_frames2[:frame_idx])

        loss = self.plot_frames1[frame_idx]
        loss_recon = self.plot_frames2[frame_idx]
        txt = "log10(loss_scalogram)={:.1f} ({}) | LR={:.1f}".format(
            loss, "L2->L1" if frame_idx > loss_switch_iter else "L2",
            lrs[frame_idx])
        self.txt2.set_text(txt)
        self.txt3.set_text("log10(loss_x_reconstructed)={:.1f}".format(
            loss_recon))

        # finalize ###########################################################
        self._drawn_artists = [*self.lines0, *self.ims1, *self.lines2,
                               *self.lines3, self.txt2, self.txt3]

    def new_frame_seq(self):
        return iter(range(self.n_frames))

    def _init_draw(self):
        pass

losses, losses_recon, lrs = [ls.copy() for ls in (losses, losses_recon, lrs)]
imshow_frames = ssq_x_recon
plot_frames0 = x_recons
plot_frames1 = np.log10(losses)
plot_frames2 = np.log10(losses_recon)

repeat_first = 1
n_repeats = 5

ani = PlotImshowAnimation(imshow_frames, plot_frames0, plot_frames1, plot_frames2)
ani.save('reconstruction.mp4', fps=10)
plt.show()
