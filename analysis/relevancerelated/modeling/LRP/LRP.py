"""
Functions for XAI-method LRP applied on MRInet.

# iNNvestigate
* https://github.com/albermax/innvestigate
* Most efficient/pragmatic implementation, however tf model must be re-implemented in Keras
* see also implementation of quantitative evaluation (Samek et al., 2018)
* Contrastive Layer-wise Relevance Propagation or CLRP: https://github.com/albermax/CLRP

# LRP toolbox
Sebastian Lapuschkin (FH HHI): https://github.com/sebastian-lapuschkin/lrp_toolbox

# LRP wrappers for tensorflow
* Vignesh Srinivasan (FH HHI): https://github.com/VigneshSrinivasan10/interprettensor
* Niels Warncke (?): https://github.com/nielsrolf/tensorflow-lrp

Author: Simon M. Hofmann | 2021
"""  # noqa: N999

# %% Import
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import innvestigate
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

from relevancerelated.configs import params, paths
from relevancerelated.dataloader.prune_image import get_global_max_axis, prune_mri, reverse_pruning
from relevancerelated.modeling.LRP.apply_heatmap import apply_colormap, create_cmap, gregoire_black_firered
from relevancerelated.modeling.MRInet.trained import crop_model_name, load_trained_model
from relevancerelated.utils import check_system, load_obj, save_obj
from relevancerelated.visualizer.visualize_mri import plot_mid_slice, prep_save_folder

if TYPE_CHECKING:
    import tensorflow.keras.models


# %% General functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def analyze_model(
    mri: np.ndarray,
    analyzer_type: str,
    model_: tensorflow.keras.models.Sequential,
    norm: bool,
    neuron_selection=None,  # noqa: ANN001
) -> np.ndarray:
    """Analyze model."""
    # Create analyzer
    analyzer = innvestigate.create_analyzer(
        analyzer_type,
        model_,
        disable_model_checks=True,
        neuron_selection_mode="index" if isinstance(neuron_selection, int) else "max_activation",
    )

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(mri, neuron_selection=neuron_selection)

    # Aggregate along color channels
    a = a[0, ..., 0]  # -> (x, y, z)

    if norm:
        # Normalize to [-1, 1]
        a /= np.max(np.abs(a))

    return a


def non_zero_analyzer_and_clim(sub_nr: int, a: np.ndarray, t: float, t_y: float, analy_type: str, save_folder: str):  # noqa: ANN201
    """
    Check where the distribution of the analyzer object is non-zero.

    :param sub_nr: subject number
    :param a: analyzer object
    :param t: true target value
    :param t_y: predicted target value
    :param analy_type: analyzer type
    :param save_folder: path of folder for saving the histogram plot
    :return:
    """
    bin_cnt = np.histogram(a.flatten(), 100)

    a_no_zero = a[
        np.where(
            np.logical_or(
                bin_cnt[1][np.max([np.argmax(bin_cnt[0]) - 1, 0])] > a, a > bin_cnt[1][np.argmax(bin_cnt[0]) + 1]
            )
        )
    ]  # flattened

    clim = (np.mean(a_no_zero) - np.std(a_no_zero) * 1.5, np.mean(a_no_zero) + np.std(a_no_zero) * 1.5)
    # Could calculate global distribution of relevance values

    hist_fig = plt.figure(analy_type + " Hist", figsize=(10, 5))
    plt.subplot(1, 2, 1)
    bin_cnt_no_zero = plt.hist(a_no_zero, bins=100, log=False)
    plt.vlines(x=clim[1], ymin=0, ymax=np.max(bin_cnt_no_zero[0]))
    plt.vlines(x=clim[0], ymin=0, ymax=np.max(bin_cnt_no_zero[0]))
    plt.title("Leave peak out Histogram")
    plt.subplot(1, 2, 2)
    _ = plt.hist(a.flatten(), bins=100, log=True)  # bin_cnt
    plt.vlines(x=clim[1], ymin=0, ymax=np.max(bin_cnt_no_zero[0]))
    plt.vlines(x=clim[0], ymin=0, ymax=np.max(bin_cnt_no_zero[0]))
    plt.title("Full Histogram")
    plt.tight_layout()

    plt.savefig(
        f"{save_folder}{analy_type}_S{sub_nr}_groundtruth={t}_"
        f"pred={t_y:{'.2f' if isinstance(t_y, float) else ''}}_heatmap_hist.png"
    )
    plt.close(hist_fig)

    return clim


def plot_heatmap(
    sub_nr: int,
    t: float,
    t_y: float,
    ipt: np.ndarray,
    analyzer_obj: np.ndarray,
    analyzer_type: str = "lrp.sequential_preset_a",
    fix_clim: float | None = None,
    fn_suffix: str = "",
    rendermode: str = "alpha",
    save_folder: str | Path = "",
    save_plot: bool = True,
    mid_slices: bool = True,
    **kwargs,
) -> None:
    """
    Plot analyzer heatmap.

    :param sub_nr: subject number
    :param t: true target value
    :param t_y: predicted target value
    :param ipt: model input (MR) image
    :param analyzer_obj: analyzer object
    :param analyzer_type: type of analyzer
    :param fix_clim: fix color range
    :param fn_suffix: filename suffix
    :param rendermode: 'overlay': Plots heatmap just over input image;
                       'alpha' [default]: the magnitude of the analyzer object determines the alpha values of
                                          the underlying input image.
    :param save_folder: path to folder where plots are to be saved
    :param save_plot: whether to save plot
    :param mid_slices: whether to use mid-slices of each image axis
    :param kwargs: additional kwargs
    :return: None
    """
    a = analyzer_obj.copy()
    if save_plot:
        save_folder = prep_save_folder(Path(paths.keras.INTERPRETATION, save_folder))

    # # Plot heatmap over T1 image
    # Figure name
    sub_nr = f"S{sub_nr}" if str(sub_nr).isnumeric() else sub_nr
    figname = (
        f"{analyzer_type}_{sub_nr}_groundtruth={t:{'.2f' if isinstance(t, float) else ''}}_"
        f"pred={t_y:{'.2f' if isinstance(t_y, (float, np.floating)) else ''}}{fn_suffix}"
    )

    # Extract kwargs
    cintensifier = kwargs.pop("cintensifier", 1.0)
    clipq = kwargs.pop("clipq", 1e-2)
    min_sym_clip = kwargs.pop("min_sym_clip", True)
    true_scale = kwargs.pop("true_scale", False)
    wbg = kwargs.pop("wbg", False)  # white background, quick&dirty implementation (remove if no benefit)
    plot_empty = kwargs.pop("plot_empty", False)

    # Render image
    if rendermode.lower() == "overlay":
        assert fix_clim is None or isinstance(fix_clim, float), "fix_clim must be None or float!"  # noqa: S101
        # e.g. 0.02

        # Define color-lims for input-image & heatmap
        ipt_lim = non_zero_analyzer_and_clim(
            sub_nr=sub_nr, a=ipt, t=t, t_y=t_y, analy_type="input", save_folder=save_folder
        )
        clim = non_zero_analyzer_and_clim(
            sub_nr=sub_nr, a=a, t=t, t_y=t_y, analy_type=analyzer_type, save_folder=save_folder
        )
        # Center colour map
        clim = list(clim)
        min_max_lim = fix_clim or np.min(np.abs(clim))  # np.max for more fine-grained col-map
        clim[0], clim[1] = -1 * min_max_lim, min_max_lim
        clim[0], clim[1] = -1 * min_max_lim, min_max_lim
        # clim[0], clim[1] = -1*.01, .01  # noqa: ERA001
        clim = tuple(clim)
        # Plot T1-image
        plot_mid_slice(mri=ipt, figname=figname, cmap="binary", clim=ipt_lim, alpha=0.3, save=False)
        # Plot heatmap over T1
        plot_mid_slice(
            mri=a,
            figname=figname,
            cmap="seismic",
            clim=clim,
            c_range=None,
            alpha=0.8,
            cbar=True,
            save=save_plot,
            save_folder=save_folder,
            kwargs=kwargs,
        )

    elif rendermode.lower() == "alpha":
        if wbg:  # make the background white
            # Mirror values
            ipt = -1 * ipt + 1

        colored_a = apply_colormap(
            R=a,
            inputimage=ipt,
            cmapname="black-firered",
            cintensifier=cintensifier,
            clipq=clipq,
            min_sym_clip=min_sym_clip,
            gamma=0.2,
            true_scale=true_scale,
        )

        cbar_range = (-1, 1) if not true_scale else (-colored_a[2], colored_a[2])
        if mid_slices:
            plot_mid_slice(
                mri=colored_a[0],
                figname=figname,
                cmap=create_cmap(gregoire_black_firered),
                c_range="full",
                cbar=True,
                cbar_range=cbar_range,
                edges=False,
                save=save_plot,
                save_folder=save_folder,
                **kwargs,
            )
        else:
            # could specify axis and slice in args
            for axis in range(3):
                for sl in range(colored_a[0].shape[axis]):
                    if not plot_empty:
                        sli = [slice(None)] * 3  # init slicer
                        sli.insert(axis, sl)
                        if colored_a[0][tuple(sli)].sum() == 0.0:
                            # Do not plot empty slices
                            print(
                                f"Won't plot slice {sl} (axis={axis}) for {figname} since it contains only zeros.",
                                end="\r",
                            )
                            continue

                    print(f"Plotting slice {sl} (axis={axis}) for {figname}...", end="\r")

                    plot_mid_slice(
                        mri=colored_a[0],
                        axis=axis,
                        slice_idx=sl,
                        figname=figname,
                        cmap=create_cmap(gregoire_black_firered),
                        c_range="full",
                        cbar=True,
                        cbar_range=cbar_range,
                        edges=False,
                        save=save_plot,
                        save_folder=save_folder,
                        **kwargs,
                    )

    else:
        msg = "rendermode must be either 'alpha' or 'overlay'"
        raise ValueError(msg)


def create_heatmap_nifti(
    sic: str,
    model_name: str,
    analyzer_type: str = "lrp.sequential_preset_a",
    analyzer_obj: np.ndarray = None,
    aggregated: bool = True,
    save: bool = False,
    logging: bool = False,
    **kwargs,
) -> nb.Nifti1Image | None:
    """
    Create NifTi version of the given relevance map.

    :param sic: subject ID
    :param model_name: name of the model
    :param analyzer_type: LRP analyzer type
    :param analyzer_obj: relevance map
    :param save: save the NifTi version
    :param logging: whether to log issues with missing data
    :param kwargs: optional keyword arguments
    :return: NifTi version of LRP heatmap in fullsize of original MRI space
    """
    model_name = crop_model_name(model_name=model_name)  # remove '_final.h5' from name

    p2file = Path(
        paths.statsmaps.LRP,
        model_name,
        "aggregated" if aggregated else "",
        sic,
        f"{analyzer_type}_relevance_maps.nii.gz",
    )
    p2org = str(p2file).replace(".nii.", ".pkl.")

    # Load or create nifti heatmap
    if p2file.is_file():
        fullsize_a_nii = nb.load(p2file)

    else:  # Create
        if check_system() != "MPI":
            msg = "Function works only on MPI servers"
            raise OSError(msg)

        from relevancerelated.dataloader.LIFE.LIFE import load_sic_mri

        # Reverse pruning
        space = kwargs.pop("space", None)
        if space is None:
            space = "mni" if "MNI" in model_name else "raw" if "RAW" in model_name else "fs"

        mri_sequence = kwargs.pop("mri_sequence", None)
        if mri_sequence is None:
            mri_sequence = next(seq for seq in params.mri_sequences if seq.upper() in model_name)
        global_max = get_global_max_axis(space=space, mri_sequence=mri_sequence)
        _, mri_org = load_sic_mri(
            _sic=sic,
            mri_sequence=mri_sequence,
            follow_up=False,
            bm=True,
            norm=True,
            regis=space == "mni",
            raw=space == "raw",
            as_numpy=False,
        )

        # Check whether the original MRI is there
        if mri_org is None:
            if logging:
                with Path("./logs/log_data_issue.txt").open("r+") as file:  # r+ read & write/append mode
                    for line in file:
                        if sic in line:
                            break
                    else:
                        file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {model_name}, {sic}\n")
            return None

        # Prepare the saving path
        if save:
            # Create parent dirs if not there
            p2file.parent.mkdir(parents=True, exist_ok=True)

        # Create the analyzer object if necessary
        if analyzer_obj is None:
            # Check whether it was computed before:
            if Path(p2org).is_file():
                analyzer_obj = load_obj(
                    name=p2org.split("/")[-1], folder="/".join(p2org.split("/")[:-1]), functimer=False
                )

            else:
                analyzer_obj = analyze_model(
                    mri=prune_mri(
                        x3d=mri_org.get_fdata(),
                        # here we assume pruning
                        make_cube=True,
                        max_axis=global_max,
                    ).reshape([1, *[global_max] * 3, 1]),
                    analyzer_type=analyzer_type,
                    model_=load_trained_model(model_name=model_name),
                    norm=False,  # can be normalized later, too
                    **kwargs,
                )  # neuron_selection (for classification)

                # Save also original heatmap for the given model
            save_obj(
                obj=analyzer_obj,
                name=p2org.split("/")[-1],
                folder="/".join(p2org.split("/")[:-1]),
                as_zip=True,
                functimer=False,
            )

        # Check whether the MRI was pruned
        if analyzer_obj.shape[0] == global_max:
            fullsize_a = reverse_pruning(
                original_mri=mri_org.get_fdata(),
                pruned_mri=prune_mri(x3d=mri_org.get_fdata(), make_cube=True, max_axis=global_max),
                pruned_stats_map=analyzer_obj,
            )

        else:
            fullsize_a = analyzer_obj

        # Create NifTi version out of it:
        fullsize_a_nii = nb.Nifti1Image(dataobj=fullsize_a, affine=mri_org.affine)

        # Save
        if save:
            fullsize_a_nii.to_filename(p2file)

    return fullsize_a_nii


def create_heatmap_surface(sic, p2hm_nii, return_output=False):
    """
    Use here freesurfer's mri_vol2surf for both hemispheres.

    Do this, e.g., via nipype:
    ```
    mri_vol2surf --mov MY_HEATMAP.nii --o lh.MY_HEATMAP.mgz --regheader SIC --hemi lh # (and for rh)
    ```

    From nipype documentation:
    * https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.freesurfer.utils.html

    Activate freesurfer environment before running this script (if necessary)
    """
    if check_system() != "MPI":
        msg = "Function works only on MPI servers"
        raise OSError(msg)

    import nipype.interfaces.freesurfer as fs

    return_output_ls = []

    for fsaverage in [False, True]:
        for hemi in ["lh", "rh"]:
            sampler = fs.SampleToSurface(hemi=hemi)
            sampler.inputs.subjects_dir = ".../freesurfer_all/"  # subjects directory
            sampler.inputs.source_file = p2hm_nii  # --mov
            sampler.inputs.reg_header = True  # --regheader
            sampler.inputs.subject_id = sic  # --regheader SIC
            sampler.inputs.sampling_method = "point"  # 'average' , 'max'
            sampler.inputs.sampling_range = 0  # default in FS; should be only for average
            sampler.inputs.sampling_units = "frac"
            sampler.inputs.out_file = f"{p2hm_nii.rstrip('.nii.gz')}{'_fsavg' if fsaverage else ''}_{hemi}.mgz"  # noqa: B005  # --o
            if fsaverage:
                sampler.inputs.target_subject = "fsaverage"

            res = sampler.run()

            # Collect outputs
            return_output_ls.append(res)

    if return_output:
        return return_output_ls
    return None


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
