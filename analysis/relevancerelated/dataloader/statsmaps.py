#!/usr/bin/env python3
"""
Get statistical maps (including WM lesion maps, FA maps, etc.).

Author: Ole Goltermann & Simon M. Hofmann | 2021-2023
"""

# %% Import
from __future__ import annotations

from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from freesurfer_stats import CorticalParcellationStats
from nilearn.image import binarize_img, math_img, threshold_img
from scipy import ndimage

from relevancerelated.configs import params, paths
from relevancerelated.dataloader.atlases import atlas_label_name_to_id_converter, possible_atlases
from relevancerelated.dataloader.LIFE.LIFE import convert_id, load_sic_mri
from relevancerelated.dataloader.transformation import file_to_ref_orientation, get_list_ants_warper

# %% Set paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

P2_RAW_FLAIR_FS = str(Path(paths.DATA, "mri/{sic}/baseline/flair"))  # transform FLAIR-2-FreeSurfer
P2_WML_MAPS_ORG = str(Path(paths.statsmaps.WML, "{sic}/ples_lpa_mFLAIR_bl.nii.gz"))
P2_WML_MAPS = str(Path(paths.statsmaps.WML, "{sic}/ples_lpa_mFLAIR_bl_thr0.8_bin.nii.gz"))  # binary maps
SUBJECTS_WM = [sub.name for sub in Path(paths.statsmaps.WML).glob("sub-*")]  # list of all SICs with WML maps
P2_FA: str = str(Path(paths.statsmaps.FA, "{sic}/{sic}_fa.nii.gz"))


# %% Load statistical brain maps  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_wml_binary(sic: str, save: bool = False) -> nib.Nifti1Image:
    """
    Compute binary white matter lesion (WML) map of given SIC.

    :param sic: subject identifier code
    :param save: save binary WML map or not
    :return: binary WM lesion map [wml]
    """
    # Adapt SIC input to BIDS convention if necessary
    if sic.startswith("..."):  # anonymized
        sic = convert_id(sic)
    if not sic.startswith("sub-"):
        sic = f"sub-{sic}"

    if not Path(P2_WML_MAPS.format(sic=sic)).is_file():
        wml = nib.load(filename=P2_WML_MAPS_ORG.format(sic=sic))
        # This creates a binary map (0,1) given a threshold
        wml = math_img(formula="img > 0.8", img=wml)
        # could be replaced by binarize_img()

        if save:
            nib.save(wml, P2_WML_MAPS.format(sic=sic))
    else:
        wml = nib.load(filename=P2_WML_MAPS.format(sic=sic))

    return wml


def get_lesion_map(sic: str) -> nib.Nifti1Image:
    """
    Load white matter (WM) lesion map of given SIC.

    :param sic: subject identifier code
    :return: WM lesion probability map (wml)
    """
    # Adapt SIC input to the BIDS convention if necessary
    if sic.startswith("..."):  # anonymized
        sic = convert_id(sic)
    if not sic.startswith("sub-"):
        sic = f"sub-{sic}"

    wml = nib.load(filename=P2_WML_MAPS.format(sic=sic))
    return file_to_ref_orientation(wml)


def stats_map_to_t1_space(sic: str, stats_map_name: str, save: bool = False, verbose: bool = True) -> nib.Nifti1Image:
    """
    Transform the given statistical map into T1-(FreeSurfer) space.

    :param sic: subject identifier code
    :param stats_map_name: name of the statistical map (e.g., WM lesion map) to move to T1 (FreeSurfer space)
    :param save: save the transformed statistical map externally.
    :param verbose: verbose or not.
    :return: statistical map in t1 space
    """
    stats_map_name = stats_map_name.lower()
    possible_maps = ["wml"]  # extend this list when more are maps are implemented
    if stats_map_name not in possible_maps:
        msg = f"stats_map must be in {possible_maps}"
        raise ValueError(msg)

    stats_map = None  # init
    path2mat = None  # init
    if stats_map_name == "wml":
        # Load stats map
        stats_map = get_lesion_map(sic=sic)
        # Define path to transformation (affine) matrix
        path2mat = P2_RAW_FLAIR_FS.format(sic=sic)

    # Load in affine-matrix to transform to FreeSurfer space T1 image
    to_t1_tx = get_list_ants_warper(folderpath=path2mat, inverse=False, only_linear=True)

    # Load t1-image in FreeSurfer space as fixed
    _, t1_mri = load_sic_mri(
        _sic=sic, mri_sequence="T1", follow_up=False, bm=True, norm=True, regis=False, raw=False, as_numpy=False
    )
    t1_mri = nib.Nifti1Image(t1_mri.get_fdata(), affine=t1_mri.affine)

    # Apply transformation
    stats_map_in_t1_space = ants.apply_transforms(
        fixed=ants.from_nibabel(t1_mri), moving=ants.from_nibabel(stats_map), transformlist=to_t1_tx, verbose=verbose
    ).astype("uint8")
    if save:
        msg = "Save function not implemented yet!"
        raise NotImplementedError(msg)

    return stats_map_in_t1_space.to_nibabel()


def get_brain_stats_per_parcel(sic: str, atlas: str | None = None) -> pd.DataFrame:
    """
    Get brain stats per parcel of atlas.

    ...freesurfer_all/[SIC]/stats/
    """
    if atlas is None:

        def load_whole_brain_measurements(stats_path: str | Path) -> pd.DataFrame:
            """Load the whole brain measurement."""
            stats = CorticalParcellationStats.read(stats_path)
            stats.whole_brain_measurements["subject"] = stats.headers["subjectname"]
            stats.whole_brain_measurements["source_basename"] = Path(stats_path).name
            stats.whole_brain_measurements["hemisphere"] = stats.hemisphere

            return stats.whole_brain_measurements

        # Load all stats
        sic_stats_path = Path(
            paths.statsmaps.GM, convert_id(sic) if sic.startswith("...") else sic, "stats", "*h.aparc*.stats"
        )

        whole_brain_measurements = pd.concat(
            map(load_whole_brain_measurements, Path.glob(sic_stats_path, "*")), sort=False
        )
        # Pial is missing: 'pial_surface_total_area_mm^2', while 'white_surface_total_area_mm^2' is there
        return whole_brain_measurements.reset_index(drop=True, inplace=False).set_index("source_basename")

    # Get stats per parcel or subcortical region
    atlas = atlas.lower()

    if atlas not in possible_atlases:
        msg = f"atlas must be in {possible_atlases}!"
        raise ValueError(msg)

    if (atlas not in "aseg") & (atlas not in "atag"):
        if atlas == "dktbase":
            atl_fix = ""
        elif atlas == "dkt":
            atl_fix = "DKTatlas40."
        else:
            atl_fix = "a2009s."

        print(atl_fix)

        def load_structural_measurements(stats_path: str) -> pd.DataFrame:
            """Load structural measurements."""
            stats = CorticalParcellationStats.read(stats_path)
            stats.structural_measurements["subject"] = stats.headers["subjectname"]
            stats.structural_measurements["source_basename"] = Path(stats_path).name
            stats.structural_measurements["hemisphere"] = stats.hemisphere

            stats.structural_measurements["label_id"] = stats.structural_measurements.structure_name.apply(
                atlas_label_name_to_id_converter, args=(atlas, stats.hemisphere)
            )

            return stats.structural_measurements

        sic_stats_path = Path(
            paths.statsmaps.GM, convert_id(sic) if sic.startswith("...") else sic, "stats", f"*h.aparc.{atl_fix}stats"
        )

        structural_measurements = pd.concat(
            map(load_structural_measurements, Path.glob(sic_stats_path, "*")), sort=False
        )

        return structural_measurements.reset_index(drop=True, inplace=False).set_index("hemisphere")

    def load_subcortical_measurements(stats_path: str | Path) -> pd.DataFrame:
        """Load subcortical measurements."""
        with Path(stats_path).open() as f:
            for line in f:
                if line.startswith("# subjectname "):
                    tmp = line.rstrip()

        stats = pd.read_csv(
            stats_path,
            sep=r"\s+",
            skiprows=78,
            header=0,  # index_col="label_id",
            names=[
                "Column",
                "label_id",
                "voxel",
                "volume",
                "label_name",
                "Mean",
                "STD",
                "Min",
                "Max",
                "Range",
                "NA1",
                "NA2",
            ],
        )
        stats["subject"] = tmp[-10:]
        stats["source_basename"] = Path(stats_path).name
        return stats

    sic_stats_path: Path = Path(
        paths.statsmaps.GM, convert_id(sic) if sic.startswith("...") else sic, "stats", "aseg.stats"
    )

    return load_subcortical_measurements(sic_stats_path)  # structural_measurements


def get_fa_map(sic: str, threshold: bool) -> nib.Nifti1Image:
    """
    Load fractional anisotropy (FA) map of given SIC.

    :param sic: subject identifier code
    :param threshold: threshold data above 0.2
    :return: FA map (binary if a threshold is given)
    """
    fa = nib.load(filename=P2_FA.format(sic=sic))
    if threshold:
        fa = math_img(formula=f"img > {params.fa.threshold}", img=fa)  # threshold==0.2
        # could be replaced by binarize_img()
    return file_to_ref_orientation(fa)


def list_of_sics_with_pvs_map(multi_modal: bool) -> list[str]:
    """Get a list of SICs with a PVS map."""
    # Get PVS maps for LIFE baseline (bl) data
    maps_model_dir = Path(paths.pvs.MAPS.format("bl"), f"T1{'-FLAIR' if multi_modal else ''}.PVS")
    list_sic_pvs = maps_model_dir.glob("*/pvs_segmentation.nii.gz")
    return [sd.parent.name for sd in list_sic_pvs]


def get_pvs_map(sic: str, multi_modal: bool, threshold: bool, cluster_size: int | None) -> nib.Nifti1Image:
    """
    Load perivascular spaces (PVS) map of given SIC.

    :param sic: subject identifier code
    :param multi_modal: False: T1 or True: T1+FLAIR images have been used to train the model.
    :param threshold: threshold data >= 0.5
    :param cluster_size: minimum number of connected voxels to keep in the PVS map.
    :return: PVS map (in T1-FreeSurfer space, with globally set orientation)
    """
    # Set paths for PVS maps of LIFE baseline data
    maps_model_dir = Path(paths.pvs.MAPS.format("bl"), f"T1{'-FLAIR' if multi_modal else ''}.PVS")

    if sic.startswith("..."):  # anonymized
        sic = convert_id(sic)
    path_to_map = Path(maps_model_dir, sic, "pvs_segmentation.nii.gz")

    pvs = nib.load(path_to_map)  # (256, 256, 256)

    if threshold:
        # Note in contrast to math_img, threshold_img does not return a binarized map,
        # and the threshold is equivalent to the formular f"img >= {th}"
        pvs = binarize_img(
            img=threshold_img(img=pvs, threshold=params.pvs.threshold, cluster_threshold=cluster_size or 0),
            threshold=0,  # threshold is applied already in threshold_img()
        )  # params.pvs.threshold == 0.5
        # Without cluster: pvs = math_img(formula=f"img >= {params.pvs.threshold}", img=pvs)

    # Reorient to project space
    return file_to_ref_orientation(pvs)


def apply_mask_dilation(
    mask_nii_or_arr: nib.Nifti1Image | np.ndarray, by: int, isoform: bool = False
) -> nib.Nifti1Image | np.ndarray:
    """
    Apply dilation to the mask.

    This leads to in increased size of masked regions.

    Note, this is not proportional to the individual mask regions.

    :param mask_nii_or_arr: binary Nifti or numpy array mask to dilate
    :param by: dilating by n pixels/voxels, extend masked regions -> binary_dilation(structure=None, ...)
    :param isoform: isotropic dilation, which increases the size of the mask in all directions equally
    :return: dilated mask
    """
    mask, is_nii = (
        (mask_nii_or_arr, False) if isinstance(mask_nii_or_arr, np.ndarray) else (mask_nii_or_arr.get_fdata(), True)
    )

    if set(np.unique(mask)) != {0, 1}:
        msg = "The mask must be binary!"
        raise ValueError(msg)

    if isoform:
        struct = np.ones(tuple(2 * by + 1 for _ in range(mask.ndim)))
        dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=1, mask=None)
    else:
        dilated_mask = ndimage.binary_dilation(mask, structure=None, iterations=by, mask=None)

    if is_nii:
        return nib.Nifti1Image(dilated_mask, mask_nii_or_arr.affine, mask_nii_or_arr.header)
    return dilated_mask


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
