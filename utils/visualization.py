from os.path import join as opj

from matplotlib import pyplot as plt


def show_mat(mat, title: str, show: bool = False, return_fig: bool = False, save: bool = False, save_path: str = None,
             xlabel=None, ylabel=None, vmin=0.0, vmax=1.0, figsize=(10, 10), tight_layout=False,
             cmap: str = 'viridis', colorbar: bool = False):
    plt.figure(figsize=figsize)
    if tight_layout:
        plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.matshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)  # or plt.imshow ?
    plt.title(title)     
    if colorbar:
        pos = plt.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pos, shrink=0.3).minorticks_on()
    plt.axis('off')
    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show()
    if save and save_path is not None:
        plt.savefig(opj(save_path, title.lower().replace(" ", "_") + '.png'))
    if return_fig:
        return plt
    else:
        plt.close()
        plt.clf()
        return None
