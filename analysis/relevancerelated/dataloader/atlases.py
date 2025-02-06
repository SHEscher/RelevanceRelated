"""
Script to load various brain atlases.

Atlases Info:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases
    http://www.diedrichsenlab.org/imaging/propatlas.htm
    http://neuromorphometrics.com/2016-03/ProbAtlas.html

Author: Simon M. Hofmann | 2020
"""

# %% Import
from __future__ import annotations

import os
from pathlib import Path
from xml.etree import ElementTree as ET

import nibabel as nib
import nilearn as nl
import numpy as np
import pandas as pd

from relevancerelated.configs import paths
from relevancerelated.dataloader.LIFE.LIFE import convert_id
from relevancerelated.dataloader.prune_image import get_global_max_axis, prune_mri
from relevancerelated.dataloader.transformation import file_to_ref_orientation
from relevancerelated.utils import cprint

# %% Set global vars  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
possible_atlases = ["dktbase", "dkt", "destrieux", "aseg", "atag"]  # dkt = DKT40
ATLAS_DIR = Path(paths.DATA, "atlas")
LIFE_FS_DIR = ".../freesurfer_all/"


# %% Load atlases < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_atlas(  # noqa: ANN201
    name: str,
    reorient: bool = True,
    mm: int = 2,
    prob_atlas: bool = True,
    thr: float = 0,
    reduce_overlap: bool = True,
    as_nii: bool = False,
    verbose: bool = False,
):
    """
    Get atlas.

    > Harvard-Oxford:
        Name of atlas to load. Can be:
            These have labels from 0-len(label), atlas-shape (91, 91, 109), thr := probability threshold:
                cort-maxprob-thr0-2mm,  # thr=0: covers most
                cort-maxprob-thr25-2mm,
                cort-maxprob-thr50-2mm,  # thr=50: has more un-labeled spots
                sub-maxprob-thr0-2mm,
                sub-maxprob-thr25-2mm,
                sub-maxprob-thr50-2mm

            These have probability maps between 0-~98%, atlas-shape (91, 91, 109, n_labels):
                cort-prob-2mm,
                sub-prob-2mm
    > jhu
    > juelich
    > cerebellum
    and more
    """
    name = name.lower()

    # # Load atlas
    # Set save path for atlas
    p2atlas = Path(paths.DATA, "BrainAtlases", "nilearn_data")

    if verbose:
        print(f"Load '{name}' atlas ...")

    if "harvard_oxford" in name:
        # Harvard Oxford Atlas (Note: has no Cerebellum)
        # Harvard Oxford Atlas Subcortical regions
        prefx = "sub" if name == "harvard_oxford_sub" else "cort"
        atl_kind = f"{prefx}-prob-{mm}mm" if prob_atlas else f"{prefx}-maxprob-thr{thr}-{mm}mm"

        harvard_oxford = nl.datasets.fetch_atlas_harvard_oxford(atl_kind, data_dir=p2atlas)

        # Load atlas (nifti) from the path
        atlas = nib.load(harvard_oxford.maps)  # could use: nl._utils.check_niimg_4d(harvard_oxford.maps)
        # 'harvard_oxford.maps' is a datapath

        # Re-orient atlas
        if reorient:
            atlas = file_to_ref_orientation(atlas)

        # Prep atlases
        atlabels = harvard_oxford.labels[1 if prob_atlas else 0 :]
        # list of labels, leave 'Background' label at [0] out for prob_atlas

        # Reduce overlap (currently mainly for overlap between subcortical and cortical atlas)
        if reduce_overlap and prefx == "sub":
            if as_nii:
                msg = "reduce_overlap=True is not implemented for 'as_nii=True'"
                raise NotImplementedError(msg)
            atl = atlas.get_fdata()  # get volume to np.array
            # Get indices for cortex labels
            clab = [i for i, label in enumerate(atlabels) if "Cortex" in label]
            # 1, 12 (for prob_atlas) | 2, 13 (for not prob_atlas)

            print("\nNote: '{}' and '{}' are removed from '{}' atlas!".format(*[atlabels[cl] for cl in clab], name))

            # Remove from atlas, and from labels
            for cl in clab[::-1]:  # reverse order for label-pop() & np.delete()
                if prob_atlas:
                    atl = np.delete(arr=atl, obj=cl, axis=-1)
                else:
                    atl[atl == cl] = 0
                    # Adapt labeling
                    atl[atl > cl] -= 1

                atlabels.pop(cl)

        elif not as_nii:
            atl = atlas.get_fdata()  # get volume to np.array

        # Correct typo
        if verbose and prefx == "sub":
            print(f"{name} atlas labels:", *atlabels, sep="\n * ")
            print("\nsee writing:")
            print("\tRight ... 'Ventricle'", "|", "Left ... 'Ventrical'\n")
            print("Rename ...")

        atlabels = [attl.replace("Ventrical", "Ventricle") for attl in atlabels]

    elif name == "cerebellum":
        # # Load cerebellum atlas
        # Check also: nl.datasets.fetch_coords_dosenbach_2010(ordered_regions=True)

        atl_kind = (
            f"Cerebellum-MNIfnirt-prob-{mm}mm.nii.gz"
            if prob_atlas
            else f"Cerebellum-MNIfnirt-maxprob-thr{thr}-{mm}mm.nii.gz"
        )

        if verbose:
            print("Cerebellum Atlases in:\n\t", paths.atlas.CEREBELLUM)
            print(
                "",
                *os.listdir(paths.atlas.CEREBELLUM),
                sep="\n* ",
            )
            print(f"Load: {atl_kind}")

        atlas_cereb = nib.load(Path(paths.atlas.CEREBELLUM, atl_kind))  # '...-MNIflirt-prob-2mm.nii.gz'

        # Re-orient atlas
        if reorient:
            atlas_cereb = file_to_ref_orientation(atlas_cereb)

        atl = atlas_cereb if as_nii else atlas_cereb.get_fdata()

        # Get labels
        cereb_label_file = Path(paths.atlas.CEREBELLUM).parent.joinpath("Cerebellum_MNIfnirt.xml")
        # '...-MNIflirt'
        names = {}
        for label in ET.parse(cereb_label_file).findall("//label"):  # noqa: S314
            names[int(label.get("index")) + 1] = label.text
        atlabels = list(names.values())

        if not prob_atlas:
            atlabels = ["Background", *atlabels]

        # could be provided as sklearn.datasets.base.Bunch, as in nl.dataset.fetch_atlas...()
        # labels could be of form 'Cereb. Left I-IV' instead of 'Left I-IV'

    elif name == "jhu":
        # White-Matter Atlas: JHU DTI-based white-matter atlases
        #   jhulabels = p2jhu.split("JHU/")[0] + 'JHU-labels.xml' # labels also in folder '**/JHU/'  # noqa: ERA001
        if prob_atlas:
            msg = "prob_atlas=True not implemented for 'jhu' atlas yet!"
            raise NotImplementedError(msg)
        p2_jhu = Path(paths.atlas.JHU)
        atl = nib.load(Path(paths.atlas.JHU, f"JHU-ICBM-labels-{mm}mm.nii.gz"))
        atlabels_file = p2_jhu.parent.joinpath("JHU-labels.xml")

        names = {}
        for label in ET.parse(atlabels_file).findall("//label"):  # noqa: S314
            names[int(label.get("index"))] = " ".join(label.text.replace("\n        ", "").split())
        atlabels = list(names.values())

        if reorient:
            atl = file_to_ref_orientation(atl)

        atl = atl if as_nii else atl.get_fdata()

    elif name == "juelich":
        atl_kind = f"prob-{mm}mm" if prob_atlas else f"maxprob-thr{thr}-{mm}mm"
        juelich_atl = nl.datasets.fetch_atlas_juelich(
            atlas_name=atl_kind,  # "maxprob-thr[0, 25, 50]-2mm",
            data_dir=None,  # default: None
            symmetric_split=False,  # default: False
            resume=True,
            verbose=True,
        )

        # Load atlas (nifti)
        atlas = juelich_atl.maps

        # Re-orient atlas
        if reorient:
            atlas = file_to_ref_orientation(atlas)

        atl = atlas if as_nii else atlas.get_fdata()

        # Prep atlases
        atlabels = juelich_atl.labels[1 if prob_atlas else 0 :]

    elif name == "atag":
        # Basal Ganglia
        msg = "Loading the basal ganglia atlas ATAG is not implemented yet!"
        raise NotImplementedError(msg)

    else:
        msg = f"Given atlas '{name}' not found."
        raise NameError(msg)

    if verbose:
        print(f"{name} atlas.shape:", atl.shape)
        print(f"N {name} atlas labels:", len(atlabels))

    return atl, atlabels


def load_atag_combined_mask(
    mm: int, binarized: bool, reorient: bool = True, non_linear: bool = True, norm: bool = True
) -> nib.Nifti1Image:
    """
    Load the combined atlas mask image.

    :param mm: resolution
    :param reorient: reorient to project orientation space
    :param non_linear: take the non-linear version
    :param norm: take the normalized version
    """
    # Note that there are many versions of the atlas, here we just take ATAG V1.0, MNI04 (MNI 1mm)
    # For the other, see files in: Final_Neuroimage_2014_ATAG_prop_masks/
    p2atlas = Path(paths.atlas.ATAG)
    lin_sfx = "Non-Linear" if non_linear else "Linear"
    norm_sfx = "normalized" if norm else "non-normalized"
    th_sfx = "033" if norm else "10"
    atl = nib.load(
        Path(
            p2atlas,
            "ATAG_1mm_space",
            f"{lin_sfx}",
            f"{norm_sfx}",
            f"{lin_sfx.replace('-', '').title()}_combined_masks_threshold{th_sfx}_1mm.nii.gz",
        )
    )  # (182, 218, 182), there are no labels

    if mm > 1:
        atl = nl.image.resample_to_img(
            source_img=atl,
            target_img=nib.load(paths.atlas.MNI.format(mm=mm)),  # MNI152 with shape: (91, 109, 91) or (182, 218, 182)
        )  # note, that mm must be in {1, 2}

    # Re-orient atlas
    if reorient:
        atl = file_to_ref_orientation(atl)  # (182, 182, 218) | (92, 92, 110)

    if binarized:
        bin_atl = atl.get_fdata()
        bin_atl[bin_atl > 0.05] = 1  # noqa: PLR2004
        atl = nib.Nifti1Image(bin_atl.round(0).astype(bool).astype(bin_atl.dtype), atl.affine)

    return atl


def merge_atlases(prune: bool = True, prob_atlas: bool = True, verbose: bool = True):  # noqa: ANN201
    """Merge atlases."""
    atl, atlabels = get_atlas(
        name="harvard_oxford", prob_atlas=prob_atlas
    )  # atl.sum(axis=3).max() == 100, i.e. prob. adds to 1
    subatl, subatlabels = get_atlas(name="harvard_oxford_sub", prob_atlas=prob_atlas)  # ~ 100.00001162290573
    ceratl, cereb_labels = get_atlas(name="cerebellum", prob_atlas=prob_atlas)  # 101.0 (??)

    # Concatenate all atlases and labels accordingly, then prune and plot...
    if prob_atlas:
        poolatl = np.concatenate([atl, subatl, ceratl], axis=-1)
        # Note: poolatl.sum(axis=3).max() >> 100 !!, but should add to 100
        #  poolatl[np.where(poolatl.sum(axis=3) > 100)]  # (172818, 97)  # noqa: ERA001
        #  on the other hand, regions are treated separately in further analysis anyway

    else:
        # Labels must be stacked
        subatl[subatl > 0] += atl.max()  # => 0, 49-69
        ceratl[ceratl > 0] += subatl.max()  # => 0, 70-97

        # atlases are (still) not mutally exclusive
        poolatl = np.add(np.add(atl, subatl), ceratl)
        if verbose:
            cprint(
                string="WARNING: Still under construction for max-prob atlases. Atlasses are not mutually "
                "exclusive.\nHence, labelling of this merged version is off.",
                col="r",
            )

    print("Three atlases are merged.")

    poolatlabels = atlabels + subatlabels + cereb_labels  # len = 97

    if prune:
        poolatl = prune_atlas(poolatl)

    return poolatl, poolatlabels


def prune_atlas(atlas):
    """Prune an atlas."""
    prob_atlas = atlas.ndim == 4  # noqa: PLR2004

    # Add brain mask to avoid centering while pruning, needs to be subtracted after
    # Create brain-mask from poolatl
    poolatl, _ = merge_atlases(prune=False, prob_atlas=prob_atlas, verbose=False)  # get full/pooled atlas
    f_atlas_bm = np.sum(poolatl, axis=3) if prob_atlas else poolatl
    f_atlas_bm[f_atlas_bm > 0] = 1
    pf_atlas_bm = prune_mri(
        x3d=f_atlas_bm, make_cube=True, max_axis=get_global_max_axis(space="mni")
    )  # pruned brain-mask

    # Needs to be done via "atlas brain mask", since MNI brain mask does not match perfectly (see above)
    if prob_atlas:
        patlas = np.stack(
            [
                prune_mri(
                    x3d=atlas[..., ch] + f_atlas_bm,  # 98**3
                    make_cube=True,
                    max_axis=get_global_max_axis("mni"),
                )
                - pf_atlas_bm
                for ch in range(atlas.shape[-1])
            ],
            axis=3,
        )

    else:
        patlas = (
            prune_mri(x3d=atlas + f_atlas_bm, make_cube=True, max_axis=get_global_max_axis(space="mni")) - pf_atlas_bm
        )

    return patlas


def atlas_label_id_to_name_converter(label_id: int, atlas: str, hemisphere: str) -> str:
    """Convert label ID to label name."""
    atlas = atlas.lower()
    if atlas not in possible_atlases:
        msg = f"atlas must be in {possible_atlases}!"
        raise ValueError(msg)

    label_name = None
    if atlas == "dktbase":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.ctab"), sep=" ", header=None, index_col=0)
        _id = label_id - (1000 if hemisphere.lower() in "left" else 2000)
        label_name = a.loc[_id].to_numpy()[0]

    elif atlas == "dkt":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.DKTatlas40.ctab"), sep=" ", header=None, index_col=0)
        # exactly the same as aparc.annot.DKTatlas40.ctab
        _id = label_id - (1000 if hemisphere.lower() in "left" else 2000)
        label_name = a.loc[_id].to_numpy()[0]

    elif atlas == "destrieux":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.a2009s.ctab"), sep=" ", header=None, index_col=0)

        _id = label_id - (11100 if hemisphere.lower() in "left" else 12100)
        label_name = a.loc[_id].to_numpy()[0]

    elif atlas == "aseg":
        a = pd.read_csv(Path(ATLAS_DIR, "ASegStatsLUT.txt"), header=None, index_col=0, delim_whitespace=True)

        _id = label_id
        label_name = a.loc[_id].to_numpy()[0]

    else:
        cprint(string=f"Atlas '{atlas}' not found!", col="r")

    return label_name


def atlas_label_name_to_id_converter(label_name: str, atlas: str, hemisphere: str) -> str | None:
    """Convert atlas name of label to ID."""
    atlas = atlas.lower()

    if atlas not in possible_atlases:
        msg = f"atlas must be in {possible_atlases}!"
        raise ValueError(msg)

    label_id = None
    if atlas == "dkt":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.DKTatlas40.ctab"), sep=" ", header=None, index_col=0)
        label_id = a[a[1] == label_name].index[0] + (1000 if hemisphere.lower() in "left" else 2000)
    elif atlas == "dktbase":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.ctab"), sep=" ", header=None, index_col=0)
        label_id = a[a[1] == label_name].index[0] + (1000 if hemisphere.lower() in "left" else 2000)
    elif atlas == "destrieux":
        a = pd.read_csv(Path(ATLAS_DIR, "aparc.annot.a2009s.ctab"), sep=" ", header=None, index_col=0)
        label_id = a[a[1] == label_name].index[0] + (11100 if hemisphere.lower() in "left" else 12100)
    elif atlas == "aseg":
        a = pd.read_csv(Path(ATLAS_DIR, "ASegStatsLUT.txt"), header=None, index_col=0, delim_whitespace=True)
        label_id = a[a[1] == label_name].index[0]
    else:
        cprint(string=f"Atlas '{atlas}' not found!", col="r")

    return label_id


def load_subject_atlas(sic: str, atlas: str) -> nib.Nifti1Image:
    """Load atlas for a SIC."""
    atlas = atlas.lower()
    if atlas not in possible_atlases:
        msg = f"atlas must be in {possible_atlases}!"
        raise ValueError(msg)

    atl_fn = "aparc+aseg.mgz" if atlas not in "destrieux" else "aparc.a2009s+aseg.mgz"
    # Destrieux atlas in volume space: aparc.a2009s+aseg.mgz
    # DKT in volume space: atlas: aparc+aseg.mgz
    # both load aseg so doesn't matter
    atl_path = Path(LIFE_FS_DIR, convert_id(sic) if sic.startswith("...") else sic, "mri", atl_fn)  # anonymized

    return nib.load(atl_path)


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
