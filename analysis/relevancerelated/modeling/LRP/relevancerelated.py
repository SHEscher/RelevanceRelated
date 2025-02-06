"""
Relating LRP relevance maps to brain features.

Author: Simon M. Hofmann, Ole Goltermann & Frauke Beyer | 2021-2024
"""

# %% Import
from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import ants

# import matplotlib; matplotlib.use("TkAgg")  # use for Mac # noqa: ERA001
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import binarize_img, math_img, mean_img, smooth_img
from nilearn.masking import apply_mask, unmask  # compute_brain_mask
from nilearn.plotting import plot_glass_brain, plot_img_on_surf, plot_stat_map  # , plot_img
from scipy.stats import pearsonr
from tqdm import tqdm

from relevancerelated.configs import params, paths
from relevancerelated.dataloader.atlases import (
    get_atlas,
    load_atag_combined_mask,
    load_subject_atlas,
    possible_atlases,
)
from relevancerelated.dataloader.LIFE.LIFE import age_of_sic, convert_id, load_sic_mri
from relevancerelated.dataloader.statsmaps import (
    P2_WML_MAPS,
    SUBJECTS_WM,
    apply_mask_dilation,
    get_brain_stats_per_parcel,
    get_fa_map,
    get_pvs_map,
    list_of_sics_with_pvs_map,
    stats_map_to_t1_space,
)
from relevancerelated.dataloader.transformation import file_to_ref_orientation, get_list_ants_warper
from relevancerelated.modeling.LRP.apply_LRP import load_sic_heatmap
from relevancerelated.modeling.LRP.LRP import create_heatmap_nifti
from relevancerelated.modeling.MRInet.trained import load_trained_model
from relevancerelated.statistics.stats import correct_data_for
from relevancerelated.utils import ask_true_false, cosine_similarity, cprint
from relevancerelated.visualizer.regressions import model_summary_publication_ready, plot_poly_fit
from relevancerelated.visualizer.visualize_mri import plot_mid_slice

if TYPE_CHECKING:
    from collections.abc import Callable

# %% Set paths & global vars >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# FLAIR sub-ensembles with corresponding relevance maps used for the WML analysis among others
ALL_FLAIR_SUBENS_NAMES = [
    "...",  # should be filled with the actual sub-ensemble names
]

# T1 sub-ensembles used for the CS analysis among others
ALL_T1_SUBENS_NAMES = [
    "...",  # should be filled with the actual sub-ensemble names
]


# %% General functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_paths_to_sic_heatmap_niftis(
    sic: str, mri_sequence: str, generate: bool = True, verbose: bool = False
) -> list[Path]:
    """
    Get path(s) to relevance map(s) in NiFTI format for a given SIC.

    In case, there are multiple heatmaps (relevance maps) return all paths.
    If NiFTis are not present, they can be generated if corresponding heatmaps as numpy arrays exist.

    :param sic: SIC to load
    :param mri_sequence: the MRI sequence of the corresponding sub-ensemble
    :param generate: if heatmap but NiFTI format is not present, generate it?
    :param verbose: being verbose or not
    """
    if mri_sequence.lower() not in {"t1", "flair"}:
        msg = "mri_sequence must be either 't1' or 'flair'!"
        raise ValueError(msg)

    # Get either T1 or FLAIR sub-ensembles (from whole brain MLENS type I)
    list_of_subens = ALL_T1_SUBENS_NAMES if mri_sequence.lower() == "t1" else ALL_FLAIR_SUBENS_NAMES

    # Convert SIC if the new SIC format is used
    if not sic.startswith("..."):  # anonymized
        sic = convert_id(sic)
        # fails if not given a valid new SIC

    # Set filename profile of relevance maps in NiFTI format
    analyzer_type: str = "lrp.sequential_preset_a"
    nii_hm_name = f"{analyzer_type}_relevance_maps.nii.gz"

    # Save paths to NiFTI heatmaps
    paths_to_sic_heatmap_niis = []  # init
    # For each sub-ensemble check if the given SIC has a corresponding heatmap
    for subens_name in list_of_subens:
        sic_subens_heatmap_dir = list(Path(paths.statsmaps.LRP, subens_name, "aggregated").glob(sic))
        if sic_subens_heatmap_dir:  # is of length 1 (or 0)
            sic_subens_heatmap_dir = sic_subens_heatmap_dir[0]
            path_to_nii = sic_subens_heatmap_dir / nii_hm_name

            # Check if heatmap NiFTI in FreeSurfer space is present
            if path_to_nii.exists():
                paths_to_sic_heatmap_niis.append(path_to_nii)
                continue

            # Check if the NiFTI should be generated
            if not generate:
                if verbose:
                    cprint(
                        string=f"For sub-ensemble '{subens_name}' the relevance map as NIFTI image of SIC '{sic}' "
                        "is not present.\n"
                        "It could be generated, if the argument 'generate' is set to True at function call!",
                        col="y",
                    )
                continue

            # Load heatmap as a pruned numpy array
            hm = load_sic_heatmap(sic=sic, model_name=subens_name, aggregated=True, mni=False, verbose=verbose)

            # Create and save an (unpruned) NiFTI version of the relevance map in FreeSurfer space
            hm_nii = create_heatmap_nifti(sic=sic, model_name=subens_name, analyzer_obj=hm, aggregated=True, save=True)

            # The following could be removed after testing
            if hm_nii.get_filename() != str(path_to_nii):
                msg = "paths must match!"
                raise ValueError(msg)

            # Check again if heatmap NiFTI in FreeSurfer space is present
            if path_to_nii.exists():
                paths_to_sic_heatmap_niis.append(path_to_nii)
            else:
                cprint(
                    string=f"For sub-ensemble '{subens_name}' the relevance map as NIFTI image of SIC '{sic}' "
                    "is missing!",
                    col="r",
                )
    return paths_to_sic_heatmap_niis


def load_sic_heatmap_nifti(
    sic: str, subens_name: str, aggregated: bool = True, verbose: bool = True
) -> nib.Nifti1Image:
    """
    Load heatmap of a given subject in NifTi format.

    The heatmap will be in the original space before pruning and re-orientation to the project space.

    :param sic: subject identifier code
    :param subens_name: name of sub-ensemble
    :param aggregated: whether to take aggregated heatmaps (otherwise specify base model)
    :param verbose: verbose or not
    :return: return aggregated LRP heatmap of given sub-ensemble in NifTi format
    """
    # load heatmap as pruned numpy
    hm = load_sic_heatmap(sic=sic, model_name=subens_name, aggregated=aggregated, mni=False, verbose=verbose)
    return create_heatmap_nifti(
        sic=sic, model_name=subens_name, analyzer_obj=hm, aggregated=aggregated, save=False
    )  # returns nii-version of heatmaps (not pruned)


def mask_heatmap_with_atlas_region(
    sic: str, subens_name: str, label_id: int, atlas_name: str = "destrieux", fill: int | None = None
) -> np.ndarray:
    """
    Mask a relevance map (heatmap) with an atlas region.

    :param sic: subject identifier code
    :param subens_name: name of the sub-ensemble
    :param label_id: ID of the atlas label
    :param atlas_name: atlas to be used [check 'possible_atlases']
    :param fill: if masked areas should be filled with given int, otherwise return masked object
    :return: masked heatmap
    """
    sic_atlas_volume = load_subject_atlas(sic=sic, atlas=atlas_name)
    sic_heatmap = load_sic_heatmap_nifti(sic=sic, subens_name=subens_name)

    # Create a mask for a heatmap using the brain atlas region
    masked_hm = np.ma.masked_where(
        condition=sic_atlas_volume.get_fdata() != label_id,
        # masked everything, which is not the label ID
        a=sic_heatmap.get_fdata(),
        copy=True,
    )
    if isinstance(fill, int):
        return masked_hm.filled(fill_value=fill)
    return masked_hm


def plot_brain_with_region(sic: str, label_id: int, atlas: str = "destrieux", mri_sequence: str = "t1"):
    """
    Plot brain with an atlas-region overlaid.

    :param sic: subject identifier code
    :param label_id: ID of atlas label
    :param atlas: atlas to be used [check 'possible_atlases']
    :param mri_sequence: T1, FLAIR, or SWI
    :return: return aggregated LRP heatmap of given sub-ensemble in NifTi format
    """
    atlas = atlas.lower()
    if atlas not in possible_atlases:
        msg = f"atlas must be in {possible_atlases}!"
        raise ValueError(msg)

    # Get the MRI
    sic_mri = load_sic_mri(_sic=sic, mri_sequence=mri_sequence, bm=True, norm=True, as_numpy=False)[1].get_fdata()

    # Get the atlas
    sic_atlas = load_subject_atlas(sic=sic, atlas=atlas)

    # Masked everything but the region with the given label ID
    masked_mri = np.ma.masked_where(condition=sic_atlas.get_fdata() != label_id, a=sic_mri, copy=True)

    # Find slices (x,y,z) where the region is visible the best
    max_slice = [v[0] for v in np.where(masked_mri == masked_mri.max())]

    # Plot (overlay) brain (in gray) and region (in red)
    plot_mid_slice(
        mri=sic_mri, crosshairs=False, edges=False, slice_idx=max_slice, cmap="gray", figname="masked-brain"
    )
    plot_mid_slice(
        mri=masked_mri,  # plt.imshow works with masked np.array's
        slice_idx=max_slice,
        crosshairs=False,
        edges=False,
        cmap="Reds",
        c_range=None,
        figname="masked-brain",
    )  # , interpolation='none')


def estimated_brain_age(sic: str, subens_name: str, subset: str = "test", verbose: bool = False) -> float | None:
    """
    Get brain age estimate of a LIFE subject made by given sub-ensemble.

    This functions searches only for brain-age estimates made on test set data,
    i.e., data the model has not seen during training.

    :param sic: subject identifier code
    :param subens_name: name of sub-ensemble
    :param subset: get BA estimate in 'validation' or 'test' set
    :param verbose: verbose or not
    :return: brain-age estimate
    """
    subset = subset.lower()
    if subset not in {"validation", "test"}:
        msg = "subset must be in either 'validation' or 'test'!"
        raise ValueError(msg)

    try:
        grand_ens = load_trained_model(model_name=subens_name.split("/")[0])  # load full grand ensemble
    except OSError:
        cprint(string=f"Given model '{subens_name}' does not exist.", col="r")
        return None
    grand_ens.set_active_model(model_name=subens_name.split("/")[-1], verbose=False)
    if grand_ens.active_model is None:
        # cprint(f"Given sub-ensemble '{subens_name.split('/')[-1]}' is not part of MLENS "
        #        f"'{grand_ens.name}'", col='r')
        return None

    predictions = grand_ens.get_predictions(subset=subset, verbose=False)
    pred_col = next(col for col in predictions.columns if "pred" in col)

    sic_row = predictions.loc[predictions.sic == sic]
    if len(sic_row) == 0:
        if verbose:
            cprint(
                string=f"For '{sic}' there is no brain-age estimate of sub-ensemble '{subens_name}' "
                f"(at least not in its test set)!",
                col="r",
            )
        return None
    pred = sic_row[pred_col].item()

    if verbose:
        y_col = (subset if subset == "test" else "val") + "_y"
        cprint(
            string=f"{sic} has an estimated brain-age of {pred:.2f}. It's true age is {int(sic_row[y_col].item())}.",
            col="g",
        )

    return pred


# %% White matter hyperintensities / lesions (WML) ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_wml_stats():
    """Run WML analysis with heatmaps."""
    # List all subjects with binary WML maps
    subjects_wm_binary = [sic for sic in SUBJECTS_WM if Path(P2_WML_MAPS.format(sic=sic)).is_file()]
    subjects_wm_binary = [convert_id(element[4:]) for element in subjects_wm_binary]
    print("Number of subjects from WM lesion analysis:", len(subjects_wm_binary))

    subject = []  # SIC
    a_voxel = []  # average relevance of all nonzero heatmap voxels
    s_voxel = []  # sum of relevance among all nonzero heatmap voxels
    n_wml_voxel = []  # number of WM lesion voxels
    a_wml_voxel = []  # average relevance of WM lesion voxels
    s_wml_voxel = []  # sum of relevance among WM lesion voxels
    as_voxel = []  # average relevance of all nonzero heatmap voxels * number of WM lesion voxels
    s_pos_voxel = []  # sum of all positive relevance values
    s_neg_voxel = []  # sum of all negative relevance values
    p_wml_voxel = []  # percentage of positive relevance related to WM lesion voxels
    p_voxel = []  # as_voxel (only positives) / s_pos_voxel
    ap_voxel = []  # average relevance of all positive heatmap voxels
    age = []  # age of subjects
    brain_age = []  # brain-age of subjects

    # Go through all sub-ensembles and their corresponding SICs with relevance maps and compute WML stats
    for subens_name in ALL_FLAIR_SUBENS_NAMES:
        for sic in os.listdir(Path(paths.statsmaps.LRP, subens_name, "aggregated")):
            if sic not in subjects_wm_binary or sic == "...":  # anonymized
                # ...
                cprint(string=f"'{sic}' has no WML map.", col="y")
                continue

            if sic in subject:
                cprint(string=f"'{sic}' has been processed for another sub-ensemble already.", col="y")
                continue

            hm_nii = load_sic_heatmap_nifti(sic=str(sic), subens_name=subens_name)
            wml_t1 = stats_map_to_t1_space(sic=str(sic), stats_map_name="wml")
            hm_np = hm_nii.get_fdata()  # heatmap as np.array
            wml_np = wml_t1.get_fdata()  # save WM lesion map as np.array
            m = np.ma.masked_where(wml_np == 0, hm_np)  # create mask for heatmap using WM lesion map
            masked_hm_np = np.ma.compressed(m)  # select only heatmap values associated with WM lesions
            hm_np_nz = hm_np[hm_np != 0]  # select only heatmap values which are non-zeros
            hm_np_pos = hm_np[hm_np > 0]  # select only positive heatmap values
            hm_np_neg = hm_np[hm_np < 0]  # select only negative heatmap values

            subject.append(sic)
            a_voxel.append(hm_np_nz.mean())
            s_voxel.append(hm_np_nz.sum())
            n_wml_voxel.append(len(masked_hm_np))
            a_wml_voxel.append(masked_hm_np.mean())
            s_wml_voxel.append(masked_hm_np.sum())
            as_voxel.append(hm_np_nz.sum() * len(masked_hm_np))
            s_pos_voxel.append(hm_np_pos.sum())
            s_neg_voxel.append(hm_np_neg.sum())
            p_wml_voxel.append(masked_hm_np.sum() / hm_np_pos.sum())
            p_voxel.append((hm_np_pos.mean() * len(masked_hm_np)) / hm_np_pos.sum())
            ap_voxel.append(hm_np_pos.mean())
            age.append(age_of_sic(sic=str(sic), follow_up=False))
            brain_age.append(estimated_brain_age(sic=str(sic), subens_name=subens_name))

    # Save data in table
    variables = {
        "sic": subject,
        "a_voxel": a_voxel,
        "s_voxel": s_voxel,
        "n_wml_voxel": n_wml_voxel,
        "a_wml_voxel": a_wml_voxel,
        "s_wml_voxel": s_wml_voxel,
        "as_voxel": as_voxel,
        "s_pos_voxel": s_pos_voxel,
        "s_neg_voxel": s_neg_voxel,
        "p_wml_voxel": p_wml_voxel,
        "p_voxel": p_voxel,
        "ap_voxel": ap_voxel,
        "age": age,
        "brain_age": brain_age,
    }

    df_wml = pd.DataFrame(variables)

    # Fill missing BA values from val-set of initial MLens '2020-08-23_17-11_AGE_Grand_ens10'
    # df_wml = pd.read_csv(Path(paths.results.WML, "WMlesion.csv"), index_col=0) # noqa: ERA001
    for i, row in df_wml.loc[df_wml.brain_age.isna()].iterrows():  # assuming 'SIC' is not the index yet
        bas = []
        for flair_subens_name in ALL_FLAIR_SUBENS_NAMES:
            ba = estimated_brain_age(sic=row.sic, subens_name=flair_subens_name, subset="validation", verbose=False)
            if ba is not None:
                bas.append(ba)
        ba = np.mean(bas) if bas else None  # take mean if there are two or more estimates from different sub-ensembles
        print(row.sic, ba)
        df_wml.loc[i, "brain_age"] = ba
    if df_wml.brain_age.isna().any():
        cprint(
            string=f"These SICSs do not have a BA in the PVS table yet:\n{df_wml[df_wml.brain_age.isna()]}",
            col="y",
        )  # still two missing?

    # Save dataframe externally
    df_wml.to_csv(Path(paths.results.WML, "WMlesion.csv"))
    # note all other tables are saved in paths.DERIVATIVES


# %% Cortical surface (CS) features >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_cs_stats():
    """Run the cortical surface (CS) analysis with the heatmaps."""
    # Create logger
    logger = logging.getLogger(name="run_cs_stats().logger")
    handler = logging.FileHandler(Path(paths.statsmaps.DERIVATIVES, "run_cs_stats.error.log"))
    handler.setFormatter(fmt=logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s"))
    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.ERROR)

    # Get the list of subjects for the analysis
    subjects = []
    # Go through all sub-ensembles and their corresponding SICs to compute cortical surface stats

    for subens_name in tqdm(iterable=ALL_T1_SUBENS_NAMES, position=0, desc="sub-ensemble", colour="#00FFB8"):
        for sic in tqdm(
            iterable=os.listdir(Path(paths.statsmaps.LRP, subens_name, "aggregated")),
            position=1,
            leave=False,
            desc="sic",
            colour="#00FFF9",
        ):
            if not sic.startswith("..."):  # anonymized
                # it is not a SIC but a log file for instance
                continue

            if sic in subjects:
                cprint(string=f"'{sic}' has been processed for other sub-ensemble already.", col="y")
                continue

            sic_heatmap = load_sic_heatmap_nifti(sic=sic, subens_name=subens_name, verbose=False).get_fdata()
            print(subens_name)
            # Load atlases, use as mask, calculate relevance indices per region and save them

            for atlas_name in [  # undo after done and iterate over relevant atlases
                "dktbase"
            ]:  # possible_atlases:  # cortical: dktbase=DKT, dkt=DKT40 and destrieux=Destrieux atlas
                print(atlas_name)
                sic_path = Path(paths.statsmaps.DERIVATIVES, subens_name, f"{sic}_cs_{atlas_name}.csv")
                if sic_path.is_file():
                    print("continuing as subject", sic, " has file", atlas_name)
                    continue

                sic_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    sic_atlas_stats = get_brain_stats_per_parcel(sic=sic, atlas=atlas_name)
                except ValueError:  # stats file was not found
                    msg = "No atlas statistics found"
                    cprint(string=f"{msg} for '{sic}'!", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: {msg}"
                    logger.exception(msg=msg)
                    continue
                except IndexError as e:  # explore this error further
                    cprint(string=f"IndexError in get_brain_stats_per_parcel() for '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: {e}"
                    logger.exception(msg=msg)
                    continue
                except AssertionError as e:  # explore this error further
                    cprint(string=f"AssertionError in get_brain_stats_per_parcel() for '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: AssertionError {e}"
                    logger.exception(msg=msg)
                    continue

                try:
                    sic_atlas_volume = load_subject_atlas(sic=sic, atlas=atlas_name).get_fdata()
                except FileNotFoundError as e:
                    cprint(string=f"For '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: FileNotFoundError {e}"
                    logger.exception(msg=msg)
                    continue

                for label_id in sic_atlas_stats.label_id:
                    masked_hm = np.ma.masked_where(condition=sic_atlas_volume != label_id, a=sic_heatmap, copy=True)

                    sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "mean_relevance"] = masked_hm.mean()
                    sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "sum_relevance"] = masked_hm.sum()
                    sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "min_relevance"] = masked_hm.min()
                    sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "max_relevance"] = masked_hm.max()

                # Save
                sic_atlas_stats.to_csv(sic_path)

            # Add SIC to the list of processed subjects
            subjects.append(sic)

    # Merge all files
    if ask_true_false("Do you want to merge all files into one big table? "):
        merge_cs_files()


def merge_files(atlas_name: str, feature_abbr: str, mri_sequence: str | None = None) -> None:
    """
    Merge all atlas stats into one big table per atlas type.

    :param atlas_name: atlas name
    :param feature_abbr: abbreviation of the brain feature (e.g., 'cs', 'fa')
    :param mri_sequence: MRI sequence (e.g., 't1', 'flair') or None
    """
    feature_abbr = feature_abbr.lower()
    atlas_name = atlas_name.lower()
    if isinstance(mri_sequence, str):
        mri_sequence = mri_sequence.lower()
        if mri_sequence not in {"t1", "flair"}:
            msg = "mri_sequence must be either 't1' or 'flair'!"
            raise ValueError(msg)

    # Check if the file already exists
    seq_str = f"-{mri_sequence}" if mri_sequence is not None else ""
    merged_path = Path(paths.statsmaps.DERIVATIVES, f"merged_{feature_abbr}{seq_str}_{atlas_name}.csv")
    if merged_path.is_file():
        cprint(string=f"File '{merged_path.name}' already exists in '{merged_path.parent}'.", col="g")
        return

    # Load files from individuals in each sub-ensemble
    big_table = None  # init
    for filename in tqdm(
        iterable=Path(paths.statsmaps.DERIVATIVES).glob(pattern=f"**/LI*_{atlas_name}.csv"),
        desc="Create big table",
        position=1,
        colour="#00FFF9",
    ):
        sic_table = pd.read_csv(filename, converters={"subject": str})
        if sic_table.columns[0].startswith("Unnamed"):
            sic_table = sic_table.drop(columns=sic_table.columns[[0]])

        big_table = sic_table if big_table is None else pd.concat([big_table, sic_table])

        if big_table.columns[0].startswith("Unnamed"):
            cprint(string=f"There is an issue with {merged_path.name}", col="r")
            break

    # Add age information
    big_table = add_brain_age_to_table(atlas_name=atlas_name, feature_abbr=feature_abbr, merged_table=big_table)

    # Save
    big_table.to_csv(merged_path)
    cprint(string=f"Saved big table to: {merged_path}", col="b")


def merge_cs_files() -> None:
    """Merge all CS atlas stats into one big table per atlas type."""
    # Merge per atlas
    for atlas_name in tqdm(iterable=["dktbase", "dkt", "destrieux"], position=0, desc="atlas", colour="#00FFB8"):
        # DTK and Destrieux atlas
        merge_files(atlas_name=atlas_name, feature_abbr="cs")


def merge_subcort_files() -> None:
    """Merge all subcortical atlas stats into one big table per atlas type."""
    merge_files(atlas_name="aseg", feature_abbr="subcort")


def get_merged_table(atlas_name: str, feature_abbr: str) -> pd.DataFrame:
    """
    Load merged atlas stats table for given feature.

    :param atlas_name: name of the atlas (destrieux, dkt; check also variable 'possible_possible_atlases')
    :param feature_abbr: abbreviation of brain feature (e.g., 'cs', 'fa')
    :return: merged (e.g., CS or FA) atlas stats table (pandas DataFrame)
    """
    return pd.read_csv(
        Path(paths.statsmaps.DERIVATIVES, f"merged_{feature_abbr}_{atlas_name}.csv"),
        index_col=0,
        converters={"subject": str},
    )


def add_brain_age_to_table(
    atlas_name: str, feature_abbr: str, merged_table: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add age and brain-age information to merged (e.g., CS or FA) table."""
    atlas_name = atlas_name.lower()
    if not (atlas_name in possible_atlases or atlas_name == "jhu"):
        msg = f"atlas_name must be in {possible_atlases}!"
        raise ValueError(msg)

    # Merged CS table must have been computed already, if not given as argument
    if merged_table is None:
        merged_table = get_merged_table(atlas_name=atlas_name, feature_abbr=feature_abbr)

    # Init names of (brain-)age columns
    age_col = "age"
    ba_col = "brain_age"

    # Check if columns already present in table
    if age_col in merged_table.columns and ba_col in merged_table.columns:
        cprint(f"age and brain-age info already in merged CS table ('{atlas_name}')", col="g")
        return merged_table

    # Init empty columns
    merged_table["age"] = None  # init
    merged_table["brain_age"] = None  # init

    # For each subject fill age and brain-age (BA) information into table
    all_subjects = pd.unique(merged_table.subject)
    for subject in tqdm(all_subjects, position=1, desc="Add age to table", colour="#00FFF9"):
        # Get (old) SIC if necessary
        if not subject.startswith("..."):  # anonymized
            try:
                sic = convert_id(subject)
            except ValueError:
                cprint(f"There is an issue with subject '{subject}'! This subject will be ignored.", "r")
                continue
        else:
            sic = subject

        # Fill age in table
        merged_table.loc[merged_table.subject == subject, "age"] = age_of_sic(sic=sic, follow_up=False)

        # Fill brain-age in table
        # check each sub-ensemble whether it has a test-set prediction for the subject (take mean for n>2)
        list_of_subens = ALL_T1_SUBENS_NAMES if feature_abbr == "cs" else ALL_FLAIR_SUBENS_NAMES
        subject_brain_age = pd.Series([
            estimated_brain_age(sic=sic, subens_name=subens_name) for subens_name in list_of_subens
        ]).mean()
        merged_table.loc[merged_table.subject == subject, "brain_age"] = subject_brain_age

    return merged_table


# %% Fractional Anisotropy (FA) < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_fa_stats(mri_sequence: str = "flair"):
    """Run FA analysis on heatmaps."""
    mri_sequence = mri_sequence.lower()
    if mri_sequence not in {"flair", "t1"}:
        msg = "mri_sequence must be either 'flair' or 't1'!"
        raise ValueError(msg)

    # Create logger
    logger = logging.getLogger(name="run_fa_stats().logger")
    handler = logging.FileHandler(Path(paths.statsmaps.DERIVATIVES, "run_fa_stats.error.log"))
    handler.setFormatter(fmt=logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s"))
    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.ERROR)

    # Get JHU WM atlas (MNI152 2mm)
    atlas_name: str = "jhu"
    atl_nii, atl_labels = get_atlas(name=atlas_name, reorient=True, mm=2, prob_atlas=False, as_nii=True, verbose=False)
    atl = atl_nii.get_fdata()  # (91, 91, 109) | 49 atlas labels

    # Get the list of subjects for analysis
    subjects = []  # init
    list_sic_nii = []  # flair (or t1) MRI
    list_sic_hm_masked_nii = []
    list_sic_fa_map_masked_nii = []

    # Prepare data frame for voxel-wise analysis within subjects
    path_to_df = Path(paths.statsmaps.DERIVATIVES, f"voxel_wise_corr_df_FA-{mri_sequence}-relevance.csv")
    if path_to_df.is_file():
        voxel_wise_corr_df = pd.read_csv(filepath_or_buffer=path_to_df)
    else:
        voxel_wise_corr_df = pd.DataFrame(data=None, columns=["SIC", "R", "p_value", "cosine_sim"])
    voxel_wise_corr_df = voxel_wise_corr_df.set_index(keys="SIC")
    df_new_entry: bool = False  # init

    # Prepare save paths across subjects, within-voxel analysis
    stats_save_path: str = str(Path(paths.statsmaps.DERIVATIVES, "{}_FA" + f"-{mri_sequence}-relevance.nii"))  # init
    across_subjects_stats_done = (
        Path(stats_save_path.format("R_pearson")).exists() and Path(stats_save_path.format("cosine-sim")).exists()
    )

    # Go through all sub-ensembles and their corresponding SICs to compute FA stats
    all_subens_names = ALL_FLAIR_SUBENS_NAMES if mri_sequence == "flair" else ALL_T1_SUBENS_NAMES
    for subens_name in tqdm(
        iterable=all_subens_names,
        position=0,
        desc=f"Running over {mri_sequence.upper()} sub-ensembles",
        colour="#00FFB8",
    ):
        for sic in tqdm(
            iterable=os.listdir(Path(paths.statsmaps.LRP, subens_name, "aggregated")),
            position=1,
            leave=False,
            desc="Running over SICs in sub-ensemble",
            colour="#00FFF9",
        ):
            if not sic.startswith("..."):  # anonymized
                # it is not a SIC but a log file for instance
                continue

            if sic in subjects:
                cprint(string=f"'{sic}' has been processed for other sub-ensemble already.", col="y")
                continue

            # Prepare the save path for within-subject analysis
            sic_fa_stats_path = Path(paths.statsmaps.DERIVATIVES, subens_name, f"{sic}_fa_{atlas_name}.csv")

            if sic_fa_stats_path.is_file() and sic in voxel_wise_corr_df.index:
                continue

            sic_fa_stats_path.parent.mkdir(parents=True, exist_ok=True)

            # Load maps in MNI (2 mm) space
            sic_heatmap_nii = load_sic_heatmap(  # MNI152 (2 mm) space
                sic=sic, model_name=subens_name, mni=True, verbose=False
            )

            try:
                # Get binary FA map
                sic_fa_map_nii = get_fa_map(sic=sic, threshold=True)  # (160, 160, 200), MNI 1 mm
            except FileNotFoundError:
                msg = f"FA map for '{sic}' not found  | {subens_name}!"
                logger.exception(msg)
                continue

            # Align maps
            # Load original FLAIR (T1) MRI
            _, sic_mri_nii = load_sic_mri(_sic=sic, mri_sequence=mri_sequence, regis=True, as_numpy=False)
            sic_heatmap_nii = file_to_ref_orientation(sic_heatmap_nii)  # (91, 91, 109)
            sic_fa_map_nii = file_to_ref_orientation(sic_fa_map_nii)

            # This basically downsamples the FA map to 2 mm and adjusts the axes-lengths
            mnitx = ants.registration(
                fixed=ants.from_nibabel(sic_mri_nii),  # (91, 91, 109)
                moving=ants.from_nibabel(sic_fa_map_nii),
                type_of_transform="Rigid",  # here linear regis only since both MNI
                verbose=False,
            )

            sic_fa_map_nii = mnitx["warpedmovout"].to_nibabel()  # (91, 91, 109)

            if not sic_fa_stats_path.is_file():
                # Compute relationship between mean FA (part of ENIGMA; JHU atlas) and relevance map
                sic_atlas_stats = pd.DataFrame(
                    columns=[
                        "hemisphere",
                        "structure_name",
                        "subject",
                        "label_id",
                        "n_voxels",
                        "min_fa",
                        "max_fa",
                        "mean_fa",
                        "sum_fa",
                        "min_relevance",
                        "max_relevance",
                        "mean_relevance",
                        "sum_relevance",
                    ]
                )  # init
                for label_id, label_name in tqdm(
                    enumerate(atl_labels),
                    desc=f"Compute stats on {atlas_name.upper()} of '{sic}'",
                    total=len(atl_labels),
                    leave=False,
                    position=2,
                    colour="#00BCBD",
                ):
                    if label_name == "Unclassified":
                        continue

                    # Extrac hemisphere
                    hemi_ = "left" if "L" in label_name else "right" if "R" in label_name else "both"

                    # Apply WM-ROI mask on both maps
                    # Option1
                    masked_hm = np.ma.masked_where(condition=atl != label_id, a=sic_heatmap_nii.get_fdata(), copy=True)
                    masked_fa = np.ma.masked_where(condition=atl != label_id, a=sic_fa_map_nii.get_fdata(), copy=True)
                    # Option2 (results are identical to Option1, but it runs slower)
                    # mask_img = nl.image.new_img_like(atl_nii, nl.image.get_data(atl_nii) == label_id)  # noqa: ERA001
                    # masker = nl.input_data.NiftiMasker(mask_img=mask_img).fit()  # noqa: ERA001
                    # masked_hm2, masked_fa2 = masker.transform_single_imgs([sic_heatmap_nii, sic_fa_map_nii])  # noqa: ERA001, E501

                    # Fill in table
                    sic_atlas_stats.loc[len(sic_atlas_stats), :] = [
                        hemi_,
                        label_name,
                        sic,
                        label_id,
                        np.sum(atl == label_id),
                        masked_fa.min(),
                        masked_fa.max(),
                        masked_fa.mean(),
                        masked_fa.sum(),
                        masked_hm.min(),
                        masked_hm.max(),
                        masked_hm.mean(),
                        masked_hm.sum(),
                    ]

                # Save
                sic_atlas_stats.to_csv(sic_fa_stats_path)

            # Voxel-wise correlation of FA maps and relevance maps
            if sic in voxel_wise_corr_df.index and across_subjects_stats_done:
                continue

            # Create WM mask from JHU atlas
            sic_hm_masked = np.ma.masked_where(
                condition=atl == 0,  # masked everything, which is not WM
                a=sic_heatmap_nii.get_fdata(),
                copy=True,
            )

            sic_fa_map_masked = np.ma.masked_where(
                condition=atl == 0,  # zero is non-WM
                a=sic_fa_map_nii.get_fdata(),
                copy=True,
            )

            # Alternatively, use the ASEG file from FreeSurfer for WM masking
            pass

            if sic not in voxel_wise_corr_df.index:
                # 1) Simple voxel-wise correlation (FA ~ relevance map) per subject
                r_, p_value = pearsonr(x=sic_hm_masked.compressed(), y=sic_fa_map_masked.compressed())
                cos_sim = cosine_similarity(sic_hm_masked.compressed(), sic_fa_map_masked.compressed())

                # Fill the correlation coefficients in the correlation table
                voxel_wise_corr_df.loc[sic, :] = r_, p_value, cos_sim
                df_new_entry = True

            # Concatenate masked stats maps as Nifti for 2) (see below)
            list_sic_nii.append(sic_mri_nii)
            list_sic_hm_masked_nii.append(nib.Nifti1Image(sic_hm_masked, affine=sic_heatmap_nii.affine))
            list_sic_fa_map_masked_nii.append(nib.Nifti1Image(sic_fa_map_masked, affine=sic_fa_map_nii.affine))

            # Add SIC to the list of processed subjects
            subjects.append(sic)

    # Save
    if df_new_entry:
        voxel_wise_corr_df.to_csv(path_to_df)

    # 2) Cosine similarity / Pearson R per voxel across subjects
    if not across_subjects_stats_done:

        def split_any_apply(array_to_split_1d: np.ndarray, func1d: Callable):  # noqa: ANN202
            """
            Split a 1-d array into two parts and apply the given function(arr1, arr2).

            :param array_to_split_1d: Array to split.
            :param func1d: Function to apply to both parts.
            :return: Result of given function.
            """
            # Split array into chunks
            split_idx = array_to_split_1d.shape[-1] // 2  # must be even
            array_1 = array_to_split_1d[:split_idx]
            array_2 = array_to_split_1d[split_idx:]

            # Apply function to both parts
            return func1d(array_1, array_2)

        # Get mean FLAIR (T1) image (MNI152, 2 mm space)
        mean_mni_nii = mean_img(imgs=list_sic_nii)

        # Masked the mean FLAIR (T1) image as it was done for the FA & relevance maps above
        bg_wm_masked_mni = np.ma.masked_where(
            condition=atl == 0,  # mask everything, which is not WM
            a=mean_mni_nii.get_fdata(),
            copy=True,
        )

        # Create WM mask from MNI template (based on JHU atlas)
        bg_wm_mask_mni = bg_wm_masked_mni.copy()
        bg_wm_mask_mni[bg_wm_mask_mni.nonzero()] = 1
        bg_wm_mask_mni = bg_wm_mask_mni.filled(fill_value=0)  # masked_array -> array
        bg_wm_mask_mni_nii = nib.Nifti1Image(bg_wm_mask_mni, affine=mean_mni_nii.affine)

        # WM-mask FA and relevance maps, respectively
        concat_masked_hm_maps = apply_mask(
            imgs=list_sic_hm_masked_nii, mask_img=bg_wm_mask_mni_nii
        )  # (n_sub, n_voxel)
        concat_masked_fa_maps = apply_mask(imgs=list_sic_fa_map_masked_nii, mask_img=bg_wm_mask_mni_nii)

        # Concatenate FA and relevance maps
        across_sic_map_masked_concat = np.concatenate(
            [concat_masked_hm_maps, concat_masked_fa_maps], axis=0
        )  # (n_sub*2, n_voxel)

        # Run stats over non-masked voxels across subjects, that is, correlate for each voxel the
        # FA values and relevance values over subjects
        for func_stats in [pearsonr, cosine_similarity]:
            cprint(f"Compute voxel-wise correlation across subjects using '{func_stats.__name__}()' ...", col="b")
            func_to_apply_per_voxel_across_subjects = partial(split_any_apply, func1d=func_stats)

            results_across_subject = np.apply_along_axis(
                func1d=func_to_apply_per_voxel_across_subjects,
                axis=0,  # i.e., apply for each voxel
                arr=across_sic_map_masked_concat,
            )  # -> # pearsonr: (2,n_voxel); -> cosine-sim: (n_voxel,)

            # Bring 1d-array stats back to brain space (using unmasking)
            if func_stats is pearsonr:
                # note that pearsonr() -> tuple (R, p-value)
                r_map = results_across_subject[0, ...]  # R
                p_map = results_across_subject[1, ...]  # p-value

                # Unmask
                r_map_nii = unmask(r_map, mask_img=bg_wm_mask_mni_nii)
                p_map_nii = unmask(p_map, mask_img=bg_wm_mask_mni_nii)

                # Save stats images
                nib.save(img=r_map_nii, filename=stats_save_path.format("R_pearson"))
                nib.save(img=p_map_nii, filename=stats_save_path.format("p_pearson"))

            else:
                stats_map_nii = unmask(results_across_subject, mask_img=bg_wm_mask_mni_nii)

                # Save stats images
                nib.save(img=stats_map_nii, filename=stats_save_path.format("cosine-sim"))

        # Save average FLAIR (T1) MNI as well
        nib.save(img=mean_mni_nii, filename=Path(paths.statsmaps.DERIVATIVES, f"mean_{mri_sequence}.nii"))

    # Merge all files
    if ask_true_false("Do you want to merge all files into one big table? "):
        merge_fa_files(mri_sequence=mri_sequence)


def merge_fa_files(mri_sequence: str) -> None:
    """Merge all FA atlas stats into one big table for JHU."""
    # Merge over JHU atlas
    merge_files(atlas_name="jhu", feature_abbr="fa", mri_sequence=mri_sequence)


def plot_fa_stats_map(
    stats_name: str,
    mri_sequence: str = "flair",
    threshold: float = 1e-6,
    smooth: float | None = None,
    save: bool = False,
):
    """Plot correlation between FA and relevance maps."""
    possible_stats = {"R_pearson", "p_pearson", "cosine-sim"}
    if stats_name not in possible_stats:
        msg = f" stats_name must be in {possible_stats} !"
        raise ValueError(msg)

    mri_sequence = mri_sequence.lower()
    if mri_sequence not in {"flair", "t1"}:
        msg = "mri_sequence must be either 'flair' or 't1'!"
        raise ValueError(msg)

    # Load stats image
    stats_save_path = Path(paths.statsmaps.DERIVATIVES, f"{stats_name}_FA-{mri_sequence}-relevance.nii")
    stats_nii = nib.load(stats_save_path)
    stats_np = stats_nii.get_fdata()
    non_zero_stats = stats_np[stats_np.nonzero()]
    mean_mni_nii = nib.load(Path(paths.statsmaps.DERIVATIVES, f"mean_{mri_sequence}.nii"))

    alpha = params.alpha  # 0.05

    # Plot
    title: str = ""  # f"{stats_name} FA - relevance maps"
    if stats_name == "p_pearson":
        # Following is an adaptation of:
        # https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_second_level_association_test.html#sphx-glr-auto-examples-05-glm-second-level-plot-second-level-association-test-py
        n_voxels = len(non_zero_stats)

        alpha /= n_voxels  # Bonferroni-corrected

        # Correcting the p-values for multiple testing and taking negative logarithm
        stats_nii = math_img(
            f"-np.log10(np.minimum(1, img * {n_voxels!s}))",  # this equivalent to Bonferroni
            img=stats_nii,
        )

        non_zero_stats = non_zero_stats[non_zero_stats <= 1 / n_voxels]

        threshold = 1
        # Since negative log p-values are plotted and a threshold equal to 1 is used,
        # it corresponds to corrected p-values lower than 10%.
        # This means that there is less than 10% probability to make a single false discovery
        # (90% chance that no false discoveries at all are made).
        title = "neg-log of parametric corrected p-values (FWER < 10%)"

    stats_nii = stats_nii if smooth is None else smooth_img(imgs=stats_nii, fwhm=smooth)

    for bg_fix, bg in zip(["bg-black", "bg_white"], ["auto", False], strict=False):
        plot_obj = plot_stat_map(
            stat_map_img=stats_nii,
            bg_img=mean_mni_nii,  # OR use: bg_mni_wm_mask_nii
            threshold=threshold,
            draw_cross=False,
            # cmap="bwr", # noqa: ERA001
            title=title,
            black_bg=bg,  # False
        )
        if save:
            for ext in ["png", "svg"]:
                plot_obj.savefig(
                    str(stats_save_path).replace(".nii", f"_th-{threshold}_smooth-{smooth}_{bg_fix}.{ext}"), dpi=300
                )
            plt.close()
        else:
            plt.show()

    # Note that the plots do have a flipped colorbar which still needs to be fixed
    for bg_fix, bg_img in zip(["_no-bg", ""], [None, mean_mni_nii], strict=False):
        plot_obj = plot_glass_brain(
            stat_map_img=math_img("-img", img=stats_nii),  # note: == neg. img, hence the colorbar is flipped
            bg_img=bg_img,  # OR use: bg_mni_wm_mask_nii
            threshold=threshold,
            draw_cross=False,
            # cmap="bwr", # noqa: ERA001
            title=title,
            colorbar=True,
            plot_abs=False,
        )
        if save:
            for ext in ["png", "svg"]:
                plot_obj.savefig(
                    str(stats_save_path).replace(
                        ".nii", f"_th-{threshold}_smooth-{smooth}_glass{bg_fix}_neg-img.{ext}"
                    ),
                    dpi=300,
                )
            plt.close()
        else:
            plt.show()

    plot_obj = plot_img_on_surf(stat_map=stats_nii, title=title, threshold=threshold)
    if save:
        plot_obj[0].savefig(
            str(stats_save_path).replace(".nii", f"_th-{threshold}_smooth-{smooth}_surface.png"), dpi=300
        )
        plt.close()
    else:
        plt.show()

    # Plot also histogram
    fig = plt.figure()
    h = plt.hist(non_zero_stats, bins=100, log=False)
    # Pearson's R values; p-values in [1, :]
    plt.title(f"Non-zero {stats_name} FA~relevance maps")
    plt.vlines(
        x=0 if stats_name != "p_pearson" else alpha,
        ymin=0,
        ymax=h[0].max(),
        colors="r",
        linestyles="dashed",
        alpha=0.5,
    )
    if save:
        fig.savefig(str(stats_save_path).replace(".nii", "_histogram.png"), dpi=300)
    else:
        plt.show()


# %% Sub-cortical volumes T1 >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_subcortical_stats():
    """Run heatmap analysis changes in subcortical areas."""
    # Create logger
    logger = logging.getLogger(name="run_subcortical_stats().logger")
    handler = logging.FileHandler(Path(paths.statsmaps.DERIVATIVES, "run_subcort_stats.error.log"))
    handler.setFormatter(fmt=logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s"))
    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.ERROR)

    # Get the list of subjects for the analysis
    subjects = []
    # Go through all sub-ensembles and their corresponding SICs to compute cortical surface stats

    path_to_merged_df = Path(paths.statsmaps.DERIVATIVES, "merged_subcort_aseg.csv")
    if not path_to_merged_df.is_file():
        for subens_name in tqdm(iterable=ALL_T1_SUBENS_NAMES, position=0, desc="sub-ensemble", colour="#00FFB8"):
            for sic in tqdm(
                iterable=os.listdir(Path(paths.statsmaps.LRP, subens_name, "aggregated")),
                position=1,
                leave=False,
                desc="sic",
                colour="#00FFF9",
            ):
                if not sic.startswith("..."):  # anonymized
                    # it is not a SIC but a log file for instance
                    continue

                if sic in subjects:
                    cprint(string=f"'{sic}' has been processed for other sub-ensemble already.", col="y")
                    continue

                sic_heatmap = load_sic_heatmap_nifti(sic=sic, subens_name=subens_name, verbose=False).get_fdata()

                # Load atlases, use as mask, calculate relevance indices per region and save them
                atlas_name = "aseg"  # do not loop over atlases
                sic_path = Path(paths.statsmaps.DERIVATIVES, subens_name, f"{sic}_subcort_{atlas_name}.csv")

                if sic_path.is_file():
                    cprint(string=f"'{sic}' has been processed already.", col="y")
                    continue

                sic_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    sic_atlas_stats = get_brain_stats_per_parcel(sic=sic, atlas=atlas_name)
                except ValueError:  # stats file was not found
                    msg = "     No atlas statistics found"
                    cprint(string=f"{msg} for '{convert_id(sic)}'!", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: {msg}"
                    logger.exception(msg=msg)
                    continue
                except IndexError as e:  # explore this error further
                    cprint(string=f"IndexError in get_brain_stats_per_parcel() for '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: {e}"
                    logger.exception(msg=msg)
                    continue
                except AssertionError as e:  # explore this error further
                    cprint(string=f"AssertionError in get_brain_stats_per_parcel() for '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: AssertionError {e}"
                    logger.exception(msg=msg)
                    continue

                try:
                    sic_atlas_volume = load_subject_atlas(sic=sic, atlas=atlas_name).get_fdata()
                except FileNotFoundError as e:
                    cprint(string=f"For '{sic}'! {e}", col="r")
                    msg = f"{sic} | {atlas_name} | {subens_name}: FileNotFoundError {e}"
                    logger.exception(msg=msg)
                    continue

                try:
                    for label_id in sic_atlas_stats.label_id:
                        masked_hm = np.ma.masked_where(
                            condition=sic_atlas_volume != label_id, a=sic_heatmap, copy=True
                        )

                        sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "mean_relevance"] = masked_hm.mean()
                        sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "sum_relevance"] = masked_hm.sum()
                        sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "min_relevance"] = masked_hm.min()
                        sic_atlas_stats.loc[sic_atlas_stats.label_id == label_id, "max_relevance"] = masked_hm.max()

                    # Save
                    sic_atlas_stats.to_csv(sic_path)

                    # Add SIC to the list of processed subjects
                    subjects.append(sic)
                except NameError:
                    continue

        # Merge all files
        if ask_true_false("Do you want to merge all files into one big table? "):
            merge_subcort_files()

    # Prepare data frame for voxel-wise analysis within subjects
    path_to_df = Path(paths.results.GM, "corr_subcort_vol-relevance.csv")

    sc_tab = get_merged_table(atlas_name="aseg", feature_abbr="subcort")

    sc_rel_corr_df = pd.DataFrame(columns=["label_name", "r", "p"])
    for label_name, structure_df in sc_tab.groupby("label_name"):
        if label_name in params.subcortical.rois:
            r, p = pearsonr(structure_df.sum_relevance, structure_df.volume)
            # Fill df by structure_name and hemisphere
            sc_rel_corr_df = sc_rel_corr_df.append({"label_name": label_name, "r": r, "p": p}, ignore_index=True)
    sc_rel_corr_df = sc_rel_corr_df.reindex(sc_rel_corr_df.r.abs().sort_values(ascending=False).index)
    cprint(string="\nSubcortical volume (SC) - relevance correlation maps:", col="b")
    print(sc_rel_corr_df.head(3))
    sc_rel_corr_df.to_csv(path_to_df, index=False)


# %% Perivascular spaces (PVS) << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_pvs_stats(
    multi_modal: bool = True,
    mri_sequence: str = "t1",
    dilate_pvs: bool = True,
    basal_ganglia: bool = False,
    only_border: bool = False,
):
    """
    Run heatmap analysis on perivascular spaces (PVS).

    T1 sub-ensembles are used for this analysis.
    PVS maps are threshold at 0.5 & get the binary mask of PVS.

    :param multi_modal: False: T1 OR True: T1+FLAIR images have been used to train the PVS segmentation model.
    :param mri_sequence: run PVS analysis on heatmaps of the "t1" OR "flair" sub-ensembles.
    :param dilate_pvs: for T1, dilate the PVS regions to capture relevance around the otherwise zero values on the MRI.
    :param basal_ganglia: run analysis only around Basal Ganglia, ignoring PVS in upper cortical regions.
    :param only_border: for T1 PVS maps, one can only take the border of the dilated PVS map into account.
    """
    # Create logger
    Path(paths.results.PVS).mkdir(exist_ok=True)

    logger = logging.getLogger(name="run_pvs_stats().logger")
    handler = logging.FileHandler(Path(paths.results.PVS, "run_pvs_stats.error.log"))
    handler.setFormatter(fmt=logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s"))
    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.ERROR)

    # Prepare data frame for PVS analysis within subjects
    mri_sequence = mri_sequence.lower()
    if mri_sequence not in params.mri_sequences[:2]:  # "t1" or "flair"
        msg = f"Unknown mri_sequence. Must be one of the following: {params.mri_sequences[:2]}"
        raise ValueError(msg)
    # change df name based on dilation argument and whether the basal ganglia mask is applied
    if dilate_pvs and mri_sequence == "t1":
        dil_sfx = f"dilate-{params.pvs.dilate_by}_"
        if only_border:
            dil_sfx += "ob_"
    else:
        dil_sfx = ""
    bas_gan_sfx = f"basal_ganglia-d{params.pvs.basal_ganglia.dilate_mask_by}_" if basal_ganglia else ""
    path_to_df = Path(
        paths.results.PVS, f"T1{'-FLAIR' if multi_modal else ''}.PVS_{dil_sfx}{bas_gan_sfx}{mri_sequence}-subens.csv"
    )
    path_to_df.parent.mkdir(exist_ok=True, parents=True)
    if path_to_df.is_file():
        # Load existing table
        df_pvs = pd.read_csv(filepath_or_buffer=path_to_df)
        df_pvs.plot(x="age", y="n_pvs_voxel")
    else:
        # Init table
        df_pvs = pd.DataFrame(
            data=None,
            columns=[
                "SIC",
                "a_voxel",  # average relevance of all nonzero heatmap voxels
                "s_voxel",  # sum of relevance among all nonzero heatmap voxels
                "n_pvs_voxel",  # number of PVS voxels
                "a_pvs_voxel",  # average relevance of PVS voxels
                "s_pvs_voxel",  # sum of relevance among PVS voxels
                "as_voxel",  # average relevance of all nonzero heatmap voxels * number of PVS voxels
                "s_pos_voxel",  # sum of all positive relevance values
                "s_neg_voxel",  # sum of all negative relevance values
                "p_pvs_voxel",  # percentage of positive relevance related to PVS voxels
                "p_voxel",  # as_voxel (only positives) / s_pos_voxel
                "ap_voxel",  # average relevance of all positive heatmap voxels
                "age",  # age of subjects
                "brain_age",  # brain-age of subjects
            ],
        )
    df_pvs = df_pvs.set_index(keys="SIC")
    df_new_entry: bool = False  # init

    # List all subjects with PVS maps
    subjects_pvs = list_of_sics_with_pvs_map(multi_modal=multi_modal)
    print("Number of subjects from PVS analysis:", len(subjects_pvs))

    subjects = []  # list of processed SICs

    # Go through all sub-ensembles and their corresponding SICs with relevance maps and compute PVS stats
    list_of_subens = ALL_T1_SUBENS_NAMES if mri_sequence == "t1" else ALL_FLAIR_SUBENS_NAMES
    basal_ganglia_atl = load_atag_combined_mask(mm=2, binarized=False, reorient=False) if basal_ganglia else None
    for subens_name in tqdm(
        iterable=list_of_subens,
        position=0,
        desc=f"Running over {mri_sequence.upper()} sub-ensembles",
        colour="#00FFB8",
    ):
        for sic in tqdm(
            iterable=os.listdir(Path(paths.statsmaps.LRP, subens_name, "aggregated")),
            position=1,
            leave=False,
            desc="Running over SICs in sub-ensemble",
            colour="#00FFF9",
        ):
            if not sic.startswith("..."):  # anonymized
                # it is not a SIC but a log file for instance
                continue

            if convert_id(sic) not in subjects_pvs:
                cprint(string=f"'{sic}' has no PVS map.", col="y")
                continue

            if sic in subjects:
                cprint(string=f"'{sic}' has been processed for another sub-ensemble already.", col="y")
                continue

            if sic in df_pvs.index and not df_pvs.loc[sic].isna().any():
                continue

            hm_nii = load_sic_heatmap_nifti(sic=str(sic), subens_name=subens_name, verbose=False)  # (256, 256, 256)

            # Get the PVS map is in (T1-)FreeSurfer space
            pvs_fs = get_pvs_map(
                sic=sic,
                multi_modal=multi_modal,
                threshold=True,
                cluster_size=params.pvs.cluster_size,
            )

            # Dilate the PVS masks to avoid zero-value regions of PVS in T1-images (if requested)
            if mri_sequence == "t1" and dilate_pvs:
                orig_pvs_fs = pvs_fs
                pvs_fs = apply_mask_dilation(mask_nii_or_arr=pvs_fs, by=params.pvs.dilate_by)  # dilate by n voxel(s)
                if only_border:
                    # For T1 PVS masks, only consider the borders of the PVS, due to zero-intensity,
                    # hence zero-relevance values within PVS.
                    pvs_fs = nib.Nifti1Image(
                        pvs_fs.get_fdata() - orig_pvs_fs.get_fdata(), pvs_fs.affine, pvs_fs.header
                    )
                    # This takes about 3x longer: pvs_fs = math_img("img1 - img2", img1=pvs_fs, img2=orig_pvs_fs)

            # Run the analysis whole brain and for different regions: deep WM & basal ganglia
            if basal_ganglia:
                # Use the inverse warp file to bring atlas (in MNI 2 mm) to individual subject FreeSurfer (T1) space
                path_to_transforms = Path(paths.DATA, "mri", sic, "baseline", "transforms2mni")
                invmnitx = get_list_ants_warper(folderpath=path_to_transforms, inverse=True)

                # Apply inverse warping to get atlas in FreeSurfer (FS) space of SIC
                _, mri_fs = load_sic_mri(
                    _sic=sic,
                    mri_sequence=mri_sequence,
                    bm=True,
                    norm=True,
                    as_numpy=False,
                )  # (256, 256, 256)
                mri_fs = nib.Nifti1Image(dataobj=mri_fs.get_fdata(), affine=mri_fs.affine)  # mgz -> nii for ANTs
                basal_ganglia_atl_fs = ants.apply_transforms(
                    fixed=ants.from_nibabel(mri_fs),  # target is the MRI image in the individual FreeSurfer space
                    moving=ants.from_nibabel(basal_ganglia_atl),
                    transformlist=invmnitx,
                    verbose=False,
                )  # (256, 256, 256)
                basal_ganglia_atl_fs = ants.utils.convert_nibabel.to_nibabel(basal_ganglia_atl_fs)  # ANTs -> nib image

                # Re-binarize (after warping not all values arn in {0, 1} anymore)
                basal_ganglia_atl_fs = binarize_img(img=basal_ganglia_atl_fs, threshold=0.05)

                # Dilate the atlas to get also white matter regions
                basal_ganglia_atl_fs = apply_mask_dilation(
                    mask_nii_or_arr=basal_ganglia_atl_fs,
                    by=params.pvs.basal_ganglia.dilate_mask_by,  # == 5, currently.
                )

                # Mask both the relevance-/heatmap and PVS map with the individually transformed Basal Ganglia atlas
                basal_ganglia_mask_indices = basal_ganglia_atl_fs.get_fdata() == 1
                hm_nii = nib.Nifti1Image(hm_nii.get_fdata() * basal_ganglia_mask_indices, affine=hm_nii.affine)
                pvs_fs = nib.Nifti1Image(pvs_fs.get_fdata() * basal_ganglia_mask_indices, affine=pvs_fs.affine)
                # (Note, convert to nii, since the script below should work for both basal_ganglia is True/False)

            # Now mask the relevance map with the PVS map of the given SIC
            hm_np = hm_nii.get_fdata()  # heatmap as np.array
            pvs_np = pvs_fs.get_fdata()  # PVS map as np.array
            m = np.ma.masked_where(pvs_np == 0, hm_np)  # create a mask for the heatmap using the PVS map
            masked_hm_np = np.ma.compressed(m)  # select only heatmap values associated with PVS
            hm_np_nz = hm_np[hm_np != 0]  # select only heatmap values which are non-zeros
            hm_np_pos = hm_np[hm_np > 0]  # select only positive heatmap values
            hm_np_neg = hm_np[hm_np < 0]  # select only negative heatmap values

            # Fill PVS results in table
            subjects.append(convert_id(sic))
            df_pvs.loc[sic, :] = (
                hm_np_nz.mean(),  # a_voxel
                hm_np_nz.sum(),  # s_voxel
                len(masked_hm_np),  # n_pvs_voxel
                masked_hm_np.mean(),  # a_pvs_voxel
                masked_hm_np.sum(),  # s_pvs_voxel
                hm_np_nz.sum() * len(masked_hm_np),  # as_voxel
                hm_np_pos.sum(),  # s_pos_voxel
                hm_np_neg.sum(),  # s_neg_voxel
                masked_hm_np.sum() / hm_np_pos.sum(),  # p_pvs_voxel
                (hm_np_pos.mean() * len(masked_hm_np)) / hm_np_pos.sum(),  # p_voxel
                hm_np_pos.mean(),  # ap_voxel
                age_of_sic(sic=str(sic), follow_up=False),  # age
                estimated_brain_age(sic=str(sic), subens_name=subens_name),  # brain_age
            )

            df_pvs.to_csv(path_to_df)
            df_new_entry = True

    # Save data in table
    # If there are missing BA values:
    for sic, _ in df_pvs.loc[df_pvs.brain_age.isna()].iterrows():  # assuming 'SIC' is the index
        bas = []
        for subens_name in list_of_subens:
            ba = estimated_brain_age(sic=sic, subens_name=subens_name, subset="validation", verbose=False)
            if ba is not None:
                bas.append(ba)
        ba = np.mean(bas) if bas else None  # take mean if there are two or more estimates from different sub-ensembles
        df_pvs.loc[sic, "brain_age"] = ba
    if df_pvs.brain_age.isna().any():
        cprint(
            string=f"These SICSs do not have a BA in the PVS table yet:\n{df_pvs[df_pvs.brain_age.isna()]}",
            col="y",
        )

    # Save dataframe externally
    if df_new_entry:
        df_pvs.to_csv(path_to_df)


def check_pvs_age_relationship(
    multi_modal: bool = True, mri_sequence: str = "t1", basal_ganglia: bool = False, save_fig: bool = False
) -> None:
    """Check the relationship between PVS size and age."""
    dilate_pvs = False

    # Prepare data frame for PVS analysis within subjects
    mri_sequence = mri_sequence.lower()
    if mri_sequence not in params.mri_sequences[:2]:  # "t1" or "flair"
        msg = f"Unknown mri_sequence. Must be one of the following: {params.mri_sequences[:2]}"
        raise ValueError(msg)
    # change df name based on dilation argument and whether the basal ganglia mask is applied
    dil_sfx = f"dilate-{params.pvs.dilate_by}_" if (dilate_pvs and mri_sequence == "t1") else ""
    bas_gan_sfx = f"basal_ganglia-d{params.pvs.basal_ganglia.dilate_mask_by}_" if basal_ganglia else ""
    path_to_df = Path(
        paths.results.PVS, f"T1{'-FLAIR' if multi_modal else ''}.PVS_{dil_sfx}{bas_gan_sfx}{mri_sequence}-subens.csv"
    )

    if not path_to_df.is_file():
        msg = f"'{path_to_df}' can not be found!"
        raise FileNotFoundError(msg)

    # Load existing table
    df_pvs = pd.read_csv(filepath_or_buffer=path_to_df)
    corr_r, p_value = pearsonr(x=df_pvs.age, y=df_pvs.n_pvs_voxel)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    fig.suptitle(f"{path_to_df.stem.split('-subens')[0]} : #PVS ~ Age")
    sns.regplot(
        x="age",
        y="n_pvs_voxel",
        data=df_pvs,
        order=1,
        scatter_kws={
            "color": "dodgerblue",
            "facecolors": "none",
        },
        line_kws={"color": "red", "label": f"R = {corr_r:.3f}, p-value < {p_value:.2g}"},
        ax=ax,
    )
    ax.set_xlabel("Age", fontsize=16)
    ax.set_ylabel("Number of PVS voxels", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop={"size": 14})
    plt.tight_layout()
    cprint(
        string=f"{path_to_df.stem}: Correlation between the number of PVS voxel and age: "
        f"R = {corr_r:.3f}, p < {p_value:.2g}",
        col="g",
    )
    if save_fig:
        fig.savefig(Path(paths.results.PVS, f"{path_to_df.stem}_PVS-AGE.pdf", dps=300))
    else:
        plt.show()


def check_pvs_dba_relationship(
    multi_modal: bool = True, mri_sequence: str = "t1", basal_ganglia: bool = False, save_fig: bool = False
) -> None:
    """Check the relationship between PVS size and the diverging (or delta) brain age (DBA)."""
    dilate_pvs = False

    # Prepare data frame for PVS analysis within subjects
    mri_sequence = mri_sequence.lower()
    if mri_sequence not in params.mri_sequences[:2]:  # "t1" or "flair"
        msg = f"Unknown mri_sequence. Must be one of the following: {params.mri_sequences[:2]}"
        raise ValueError(msg)
    # change df name based on dilation argument and whether the basal ganglia mask is applied
    dil_sfx = f"dilate-{params.pvs.dilate_by}_" if (dilate_pvs and mri_sequence == "t1") else ""
    bas_gan_sfx = f"basal_ganglia-d{params.pvs.basal_ganglia.dilate_mask_by}_" if basal_ganglia else ""
    path_to_df = Path(
        paths.results.PVS, f"T1{'-FLAIR' if multi_modal else ''}.PVS_{dil_sfx}{bas_gan_sfx}{mri_sequence}-subens.csv"
    )

    if not path_to_df.is_file():
        msg = f"'{path_to_df}' can not be found!"
        raise FileNotFoundError(msg)

    # Load existing table
    df_pvs = pd.read_csv(filepath_or_buffer=path_to_df)
    df_pvs["dba"] = df_pvs.brain_age - df_pvs.age

    degree = 3
    df_pvs = correct_data_for(var2cor="dba", correct_for="age", data=df_pvs, deg=degree, plot=False)
    fit_model, fig, ax = plot_poly_fit(df=df_pvs, x_col="age", y_col="dba", degree=degree, n_std=3, verbose=True)
    _ = model_summary_publication_ready(model=fit_model)  # text for publication

    # Note that for degree = 1:
    # fit_model = poly_model_fit(df=df_pvs, x_col="age", y_col="dba", degree=1)  # noqa: ERA001
    # corr_r, p_value = pearsonr(x=df_pvs["dba"], y=df_pvs.age)  # noqa: ERA001
    # assert np.allclose(corr_r**2, fit_model.rsquared)  # noqa: ERA001
    # assert np.allclose(p_value, fit_model.pvalues["x1"])  # noqa: ERA001

    # Plot linear fit and save the figure
    for dba_col in ("dba", "poly-3-corrected_dba"):
        fit_model, fig, ax = plot_poly_fit(
            df=df_pvs, x_col=dba_col, y_col="n_pvs_voxel", degree=1, n_std=3, dpi=150, verbose=True
        )
        dba_name = "DBA" if dba_col == "dba" else "Corrected DBA"
        fig.suptitle(f"{path_to_df.stem.split('-subens')[0]} : #PVS ~ {dba_name}")
        ax.set_xlabel(dba_name, fontsize=16)
        ax.set_ylabel("Number of PVS voxels", fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

        # Print the correlation values
        corr_r2 = fit_model.rsquared
        p_value = fit_model.pvalues["x1"]
        cprint(
            string=f"{path_to_df.stem}: Correlation between the number of PVS voxel and {dba_name}: "
            f"R^2 = {corr_r2:.3f}, p < {p_value:.2g}",
            col="g",
        )
        if save_fig:
            fig.savefig(Path(paths.results.PVS, f"{path_to_df.stem}_PVS-{dba_name}.pdf", dps=300))
            plt.close()
        else:
            plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


if __name__ == "__main__":
    # GMV in cortical surface (CS) - relevance correlation maps <---><---><---><---><---><---><---><---><---><---><--->
    cs_tab = get_merged_table(atlas_name="dkt", feature_abbr="cs")
    cs_gmv_rel_corr_df = pd.DataFrame(columns=["structure_name", "hemisphere", "r", "p"])
    for structure_name, structure_df in cs_tab.groupby("structure_name"):
        for hemi, hemi_df in structure_df.groupby("hemisphere"):
            r, p = pearsonr(hemi_df.sum_relevance, hemi_df["gray_matter_volume_mm^3"])
            # Fill df by structure_name and hemisphere
            cs_gmv_rel_corr_df = cs_gmv_rel_corr_df.append(
                {"structure_name": structure_name, "hemisphere": hemi, "r": r, "p": p}, ignore_index=True
            )

    cs_gmv_rel_corr_df = cs_gmv_rel_corr_df.reindex(cs_gmv_rel_corr_df.r.abs().sort_values(ascending=False).index)
    cprint(string="\nGray-Matter-Volume (GMV) - relevance correlation maps:", col="b")
    print(cs_gmv_rel_corr_df.head(3))

    # Cortical Thickness (CT) in CS - relevance correlation maps
    cs_ct_rel_corr_df = pd.DataFrame(columns=["structure_name", "hemisphere", "r", "p"])
    for structure_name, structure_df in cs_tab.groupby("structure_name"):
        for hemi, hemi_df in structure_df.groupby("hemisphere"):
            r, p = pearsonr(hemi_df.sum_relevance, hemi_df["average_thickness_mm"])
            # Fill df by structure_name and hemisphere
            cs_ct_rel_corr_df = cs_ct_rel_corr_df.append(
                {"structure_name": structure_name, "hemisphere": hemi, "r": r, "p": p}, ignore_index=True
            )

    cs_ct_rel_corr_df = cs_ct_rel_corr_df.reindex(cs_ct_rel_corr_df.r.abs().sort_values(ascending=False).index)
    cprint(string="\nCortical Thickness (CT) - relevance correlation maps:", col="b")
    print(cs_ct_rel_corr_df.head(3))

    # Fractional anisotropy (FA) - relevance correlation maps -><---><---><---><---><---><---><---><---><---><---><--->
    # Uncomment the following line for plotting
    # for mri_seq in ("t1", "flair"):
    #     plot_fa_stats_map(stats_name="R_pearson", mri_sequence=mri_seq, threshold=0.01, smooth=2.5, save=True)  # noqa: E501, ERA001

    # Find structure (region) with the strongest correlation between FA and relevance scores in FLAIR
    fa_tab = get_merged_table(atlas_name="jhu", feature_abbr="fa")

    # Iterate through structures_name
    fa_rel_corr_df = pd.DataFrame(columns=["structure_name", "hemisphere", "r", "p"])
    for structure_name, structure_df in fa_tab.groupby("structure_name"):
        for hemi, hemi_df in structure_df.groupby("hemisphere"):
            r, p = pearsonr(hemi_df.sum_relevance, hemi_df.mean_fa)
            # Fill df by structure_name and hemisphere
            fa_rel_corr_df = fa_rel_corr_df.append(
                {"structure_name": structure_name, "hemisphere": hemi, "r": r, "p": p}, ignore_index=True
            )

    fa_rel_corr_df = fa_rel_corr_df.reindex(fa_rel_corr_df.r.abs().sort_values(ascending=False).index)
    cprint(string="\nFractional anisotropy (FA) - relevance correlation maps:", col="b")
    print(fa_rel_corr_df.head(3))

# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
