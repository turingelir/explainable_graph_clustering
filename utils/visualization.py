from os.path import join as opj

from matplotlib import pyplot as plt


def show_mat(mat, title: str, show: bool = False, return_fig: bool = False, save: bool = False, save_path: str = None,
             xlabel=None, ylabel=None, vmin=0.0, vmax=1.0, figsize=(10, 10), tight_layout=False,
             cmap: str = 'viridis', colorbar: bool = False):
    if tight_layout:
        pos = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        pos = plt.matshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)  # or plt.imshow ?  
    # Set figsize
    plt.figure(pos.get_figure().number, figsize=figsize)
    plt.title(title)

    if colorbar:
        # pos = plt.pcolormesh(mat, cmap=cmap, vmin=vmin, vmax=vmax)
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

if __name__ == '__main__':
    # Test show_mat
    import numpy as np
    mat = np.random.rand(10, 10)
    show_mat(mat, "Random Matrix", show=True, colorbar=True, tight_layout=True)