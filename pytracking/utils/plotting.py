import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_tensor(a: torch.Tensor, fig_num = None, title = None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')
    plt.axis('equal')
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)


def plot_graph(a: torch.Tensor, fig_num = None, title = None):
    """Plot graph. Data is a 1D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    if a_np.ndim > 1:
        raise ValueError
    plt.figure(fig_num)
    # plt.tight_layout()
    plt.cla()
    plt.plot(a_np)
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)
