"""
Functions to visualize MRIs.

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2018-2020
"""

# %% Import
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from relevancerelated.dataloader.prune_image import find_edges
from relevancerelated.utils import cprint

# %% ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<

planes = ["sagittal/longitudinal", "transverse/superior/horizontal", "coronal/posterior/frontal"]


# %% Plotting functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def prep_save_folder(save_folder: str | Path) -> Path:
    """Prepare save folder."""
    save_folder = Path(save_folder, "")  # adds '/' if not there

    if (not save_folder.exists()) and (len(str(save_folder)) > 0):
        print("Create save folder:", save_folder.absolute())
        save_folder.mkdir()

    return save_folder


def plot_slice(  # noqa: ANN201
    mri: np.ndarray, axis: int, idx_slice: int, edges: bool = True, c_range: str | None = None, **kwargs
):
    """Plot a 3D image slice."""
    # In general: axis-call could work like the following:
    # axis==0: a_slice = (idx_slice, slice(None), slice(None))  == (idx_slice, ...)
    # axis==1: a_slice = (..., idx_slice, slice(None))
    # axis==2: a_slice = (..., idx_slice)

    im, _edges = None, None  # init
    if edges:
        _edges = find_edges(mri if mri.shape[-1] > 4 else mri[..., -1])  # noqa: PLR2004
        # works for transparent (!) RGB image (x,y,z,4) and volume (x,y,z)

    # Set color range
    if c_range == "full":  # takes full possible spectrum
        i_max = 255 if np.max(mri) > 1 else 1.0
        i_min = 0 if np.min(mri) >= 0 else -1.0
    elif c_range == "single":  # takes min/max of given brain
        i_max = np.max(mri)
        i_min = np.min(mri)
    else:  # c_range=None
        if c_range is not None:
            msg = "c_range must be 'full', 'single' or None."
            raise ValueError(msg)
        i_max, i_min = None, None

    # Get kwargs (which are not for imshow)
    crosshairs = kwargs.pop("crosshairs", True)
    ticks = kwargs.pop("ticks", False)

    if axis == 0:  # sagittal
        im = plt.imshow(mri[idx_slice, :, :], vmin=i_min, vmax=i_max, **kwargs)
        if edges:
            plt.hlines(_edges[2] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)  # == max edges
            plt.hlines(_edges[3] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[4] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[5] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)

    elif axis == 1:  # transverse / superior
        im = plt.imshow(np.rot90(mri[:, idx_slice, :], axes=(0, 1)), vmin=i_min, vmax=i_max, **kwargs)
        if edges:
            plt.hlines(mri.shape[0] - _edges[5] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.hlines(mri.shape[0] - _edges[4] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[0] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[1] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)

    elif axis == 2:  # # coronal / posterior  # noqa: PLR2004
        im = plt.imshow(np.rot90(mri[:, :, idx_slice], axes=(1, 0)), vmin=i_min, vmax=i_max, **kwargs)
        if edges:
            plt.hlines(_edges[2] - 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.hlines(_edges[3] + 1, 2, mri.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(mri.shape[1] - _edges[1] - 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(mri.shape[1] - _edges[0] + 1, 2, mri.shape[1] - 2, colors="darkgrey", alpha=0.3)

    # Add mid-cross ('crosshairs')
    if crosshairs:
        plt.hlines(int(mri.shape[axis] / 2), 2, len(mri) - 2, colors="red", alpha=0.3)
        plt.vlines(int(mri.shape[axis] / 2), 2, len(mri) - 2, colors="red", alpha=0.3)

    if not ticks:
        plt.axis("off")

    return im


def plot_mid_slice(
    mri: np.ndarray,
    axis: int | None = None,
    figname: str | None = None,
    edges: bool = True,
    c_range: str | None = None,
    save: bool = False,
    save_folder: str = "./TEMP/",
    **kwargs,
):
    """
    Plot the mid-2d-slice of a given 3D-MRI.

    If no axis is given, plot for each axis its mid-slice.

    :param mri: 3D MRI
    :param axis: None: all three axes
    :param figname: name of the figure
    :param edges: if True, draw edges around the brain
    :param c_range: "full", "single", or None
    :param save: True/False
    :param save_folder: Where to save
    :param kwargs: arguments for plt.imshow(), & crosshairs: bool = True, ticks: bool = False
    """
    if save:
        save_folder = prep_save_folder(save_folder)

    # Get color bar kwargs (if any)
    cbar = kwargs.pop("cbar", False)
    cbar_range = kwargs.pop("cbar_range") if ("cbar_range" in kwargs and cbar) else None
    # only if cbar is active
    suptitle = kwargs.pop("suptitle", None)
    slice_idx = kwargs.pop("slice_idx", None)  # in case mid or other slice is given

    # Set (mid-)slice index
    sl = int(np.round(mri.shape[0] / 2)) if slice_idx is None else slice_idx

    if slice_idx is None:
        sl_str = "mid"
    elif isinstance(sl, (list, tuple)):
        if len(sl) != 3:  # noqa: PLR2004
            msg = "slice_idx must be tuple or list of length 3, is None or single int."
            raise ValueError(msg)
        sl_str = str(sl)
    else:
        sl_str = str(int(sl))

    # Begin plotting
    if axis is None:
        _fs = {"size": 10}  # define font size

        fig = plt.figure(num=f"{figname or ''} MRI {sl_str}-slice", figsize=(12, 4))
        if suptitle:
            fig.suptitle(suptitle, fontsize=14)

        # # Planes
        ims = []
        axs = []
        sls = [sl] * 3 if isinstance(sl, int) else sl  # is tuple pf length 3 or int
        for ip, plane in enumerate(planes):
            axs.append(fig.add_subplot(1, 3, ip + 1))
            ims.append(plot_slice(mri, axis=ip, idx_slice=sls[ip], edges=edges, c_range=c_range, **kwargs))
            plt.title(plane, fontdict=_fs)

            divider = make_axes_locatable(axs[ip])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.axis("off")
            if cbar and ip == len(planes) - 1:
                cax_bar = fig.colorbar(ims[-1], ax=cax, fraction=0.048, pad=0.04)  # shrink=.8, aspect=50)
                if cbar_range:
                    cax_bar.set_ticks(np.linspace(0, 1, 7), True)
                    cax_bar.ax.set_yticklabels(
                        labels=[
                            f"{tick:.2g}"
                            for tick in np.linspace(cbar_range[0], cbar_range[1], len(cax_bar.get_ticks()))
                        ]
                    )

        if save:
            fig.savefig(fname=save_folder / f"{figname or ''} MRI {sl_str}-slice.png")
            plt.close()
        else:
            plt.show()

    else:  # If specific axis to plot
        if axis not in range(3):
            msg = f"axis={axis} is not valid. Take 0, 1 or 2."
            raise ValueError(msg)

        axis_name = planes[axis].split("/")[0]

        fig = plt.figure(f"{figname or ''} {axis_name} {sl_str}-slice")
        im = plot_slice(mri, axis, idx_slice=sl, edges=edges, c_range=c_range, **kwargs)
        if cbar:
            cax_bar = fig.colorbar(im, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
            if cbar_range:
                cax_bar.set_ticks(np.linspace(0, 1, 7), True)
                cax_bar.ax.set_yticklabels(
                    labels=[
                        f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1], len(cax_bar.get_ticks()))
                    ]
                )

        plt.tight_layout()

        if save:
            fig.savefig(fname=save_folder / f"{figname or ''} {axis_name} {sl_str}-slice.png")
            plt.close()

        else:
            plt.show()


def multi_slice_viewer(mri, axis=0, **kwargs):
    """
    View image in multiple slices.

    Source: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
    """

    def remove_keymap_conflicts(new_keys_set):
        """Remove keymap conflicts."""
        for prop in plt.rcParams:
            if prop.startswith("keymap."):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def apply_rot(vol_slice):
        """Apply rotation."""
        if axis > 0:
            axes = (0, 1) if axis == 1 else (1, 0)
            vol_slice = np.rot90(vol_slice, axes=axes)
        return vol_slice

    def previous_slice(ax_):
        """Previous slice."""
        volume = ax_.volume
        ax_.index = (ax_.index - 1) % volume.shape[axis]  # wrap around using %
        slice_ax = [slice(None)] * 3
        slice_ax[axis] = slice(ax_.index, ax_.index + 1, None)
        ax_.images[0].set_array(apply_rot(volume[slice_ax].squeeze()))

    def next_slice(ax_):
        """Next slice."""
        volume = ax_.volume
        ax_.index = (ax_.index + 1) % volume.shape[axis]
        slice_ax = [slice(None)] * 3
        slice_ax[axis] = slice(ax_.index, ax_.index + 1, None)
        ax_.images[0].set_array(apply_rot(volume[slice_ax].squeeze()))

    def process_key(event):
        """Process key."""
        _fig = event.canvas.figure
        _ax = _fig.axes[0]
        if event.key == "j":
            previous_slice(_ax)
        elif event.key == "k":
            next_slice(_ax)
        print(f"At slice: {_ax.index}\r", end="")
        _ax.set_title(_ax.get_title().split("slice")[0] + f"slice {_ax.index}")
        _fig.canvas.draw()

    # Execute
    remove_keymap_conflicts({"j", "k"})
    cprint("\nPress 'j' and 'k' to slide through the volume.", "y")

    # Prepare plot
    fig, ax = plt.subplots()

    # Unpack kwargs
    window_title = kwargs.pop("window_title", "MRI Slice Slider")
    fig.canvas.set_window_title(window_title)  # fig.canvas.get_window_title()

    cbar = kwargs.pop("cbar", False)
    cbar_range = kwargs.pop("cbar_range") if ("cbar_range" in kwargs and cbar) else None
    # cbar_range: only if cbar is active

    ax.volume = mri
    ax.index = mri.shape[axis] // 2
    ax.set_title(f"{planes[axis]} | slice {ax.index}")

    # Plot
    im = plot_slice(mri=mri, axis=axis, idx_slice=ax.index, **kwargs)

    if cbar:
        ax_bar = fig.colorbar(im)
        if cbar_range:
            ax_bar.ax.set_yticklabels(
                labels=[f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1], len(ax_bar.get_ticks()))]
            )
    fig.canvas.mpl_connect("key_press_event", process_key)


# >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
