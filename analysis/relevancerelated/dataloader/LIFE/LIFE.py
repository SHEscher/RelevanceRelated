"""
Load LIFE MRI and target data from the server.

i)   T1-weighted images
ii)  Fluid-attenuated inversion recovery (FLAIR)
iii) Susceptibility weighted imaging (SWI)
iv) later also more

!!! note "For demonstration purposes only"
    Note that this script was partially anonymized and simplified for the public repository.
    In general, this script is for demonstration purposes only and not executable in its current form.
    This is due to the restricted access to the LIFE data and the specific server structure.

Author: Simon M. Hofmann | 2021-2023
"""  # noqa: N999

# %% Import
from __future__ import annotations

import ast
import concurrent.futures
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from random import sample

import nibabel as nib  # Read/write access to some common neuroimaging file formats
import nilearn as nl
import numpy as np
import pandas as pd
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_img

from relevancerelated.configs import params, paths
from relevancerelated.dataloader.LIFE.prepare_sic_table import load_raw_study_table
from relevancerelated.dataloader.prune_image import (
    compress_mri,
    get_global_max_axis,
    permute_mri,
    prune_mri,
    set_dtype_mriset,
)
from relevancerelated.dataloader.transformation import (
    add_bg_noise_mriset,
    all_manip_opt,
    file_to_ref_orientation,
    get_raw_brainmask,
    get_t1_brainmask,
    random_transform_mri,
)
from relevancerelated.utils import (
    check_system,
    chop_microseconds,
    cprint,
    function_timed,
    load_obj,
    loop_timer,
    normalize,
    only_mpi,
    save_obj,
)

# %% Global vars << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# # Brains regions
brain_regions = {  # with mapping to corresponding atlas name
    "cerebellum": "cerebellum",
    "subcortical": "harvard_oxford_sub",
    "cortical": "harvard_oxford",
}  # can be extended, but fix order

# # MRI Sequences
mri_sequences = params.mri_sequences  # ["t1", "flair", "swi"]

# # Grouping/binning of target variable
# Dict for boundaries (relatively arbitrary at this stage)
binary_boundaries = {
    "age": (
        46,  # young group: < 47
        60,
    ),  # old group: 60+, n_old > n_young
    "bmi": (
        24,  # approx. normal-weight
        29.131,  # approx. definition of obese
    ),
}


def pred_classes(target: str) -> str:
    """Provide class labels for target variable."""
    pred_cls = {  # extend if more classes
        "age": ["young", "old"],
        "bmi": ["norm", "obese"],
    }
    return pred_cls[target.lower()]


START_YEAR_FOLLOW_UP = 2018

# %% Classes and functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class DataSet:
    """Utility class to handle dataset structure."""

    def __init__(
        self,
        name: str,
        mri: dict[str, np.ndarray],
        target: dict[str, str],
        target_name: str,
        target_min_max: tuple,
        target_bias: float,
        target_encoding: str | None,
        follow_up: bool,
        sics2augment: list[str] | None = None,
        transform_types: list[str] | None = None,
        bg_noise: (bool, float) = False,
    ):
        """
        Build dataset with MRI data and target variable.

        :param name: Name of dataset ("Training", "Validation", "Test")
        :param target_name: Name of target variable, e.g. "age"
        :param mri: MRI data
        :param target: age or other variable to be predicted
        :param target_min_max: save min&max of original target (before normalization) for later reconstruct
        :param target_bias: global bias in target (i.e. mean)
        :param target_encoding: None OR 'softmax': for one-hot-encoding in classification
        :param follow_up: whether data is part of LIFE follow-up
        :param sics2augment: None OR list of SICs to augment
        :param bg_noise: True: add noise to background (default scale); else indicate scale with float
        """
        self.name = name
        self.target_name = target_name
        self._mri = mri
        self._target = target
        self._sics = list(mri.keys())
        self._tminmax = target_min_max
        self._target_bias = target_bias
        self._target_encoding = target_encoding
        self.n_classes = len(np.unique(list(target.values()))) if self._target_encoding == "softmax" else None
        self._follow_up = follow_up
        self._epochs_completed = 0
        self._sics_remain = np.array(self._sics)
        self._current_sic = None
        # Following to 'self'-inits are primarily for training data sets:
        self._sics2augment = sics2augment
        self._transform_types = transform_types
        if self._sics2augment:
            self.augment_sets()
        self.bg_noise = bg_noise
        if self.bg_noise or isinstance(self.bg_noise, float):
            self.add_bg_noise()

    @property
    def sics(self):
        """Get the list of SICs."""
        return self._sics

    @property
    def mri(self):
        """Get MRI."""
        return self._mri

    @property
    def target(self):
        """Get target."""
        return self._target

    @property
    def tminmax(self):
        """Get min / max of target."""
        return self._tminmax

    @property
    def target_bias(self):
        """Get target bias."""
        if self.sics2augment is not None:
            cprint(string="Note: Augmented training data is included in global target bias.", col="y")
        return self._target_bias

    @property
    def target_encoding(self):
        """Get target encoding."""
        return self._target_encoding

    @property
    def is_follow_up(self) -> bool:
        """Check if this is the follow-up dataset."""
        print(f"This is {'follow-up' if self._follow_up else 'base-line'} data.")
        return self._follow_up

    @property
    def transform_types(self):
        """Return transform types."""
        return self._transform_types

    @property
    def sics2augment(self):
        """Return SICs to augment."""
        if self.name != "Training":
            cprint(string="Only training set should be augmented!", col="r")
        return self._sics2augment

    @property
    def current_sic(self):
        """Get current SIC."""
        return self._current_sic

    @property
    def epochs_completed(self):
        """Return number of completed epochs."""
        return self._epochs_completed

    @property
    def sics_remain(self):
        """Return remaining SICs."""
        return self._sics_remain

    def new_epoch(self):
        """Add new epoch."""
        self._epochs_completed += 1

    def reset_sic_remain(self):
        """Reset remaining SICs."""
        self._sics_remain = np.array(list(self._mri.keys()))
        # Do not use self.sics here, since variable does not include keys of augmented data

    def update_sic_remain(self):
        """Update remaining SICs."""
        self._sics_remain = np.delete(self._sics_remain, 0)

        if len(self._sics_remain) == 0:
            self.reset_sic_remain()
            self.new_epoch()

    def shuffle_order(self):
        """Shuffle order of remaining SICs."""
        np.random.shuffle(self._sics_remain)

    def next_batch(self):
        """
        Return the next batch (size: 1) from dataset.

        :return: next batch
        """
        # Feed whole brain subject per subject, where x-dimension of 3D MRI data is treated as batch
        # batch_size = x_shape = self._mri[list(self._mri.keys())[0]].shape[0]  # noqa: ERA001
        self._current_sic = self.sics_remain[0]
        xs = self._mri[self._current_sic]  # shape: (256, 256, 256). X=batch_size, y=256, z=256)
        ys = np.reshape(self._target[self._current_sic], (1, 1))  # scalar, shape: (1, 1)

        if self._target_encoding == "softmax":
            onehot = np.zeros(shape=(1, self.n_classes), dtype=float)
            onehot[:, self.tminmax[1] - ys.item()] = 1.0
            ys = onehot
            # Check out: keras.utils.np_utils.to_categorical (as in self.to_keras())

        self.update_sic_remain()
        # No other mode implemented yet that depends on batch_size (now only 1 batch = whole brain)

        return xs, ys

    def augment_sets(self, sics2augment: list[str] | None = None) -> None:
        """Augment data set."""
        if sics2augment:
            # This allows augmenting a dataset also after initialization
            self._sics2augment = sics2augment

        if sics2augment is None and self._sics2augment is None:
            cprint(string="No augmentation list of SICs is given.\n> Dataset remains unchanged.", col="y")

        else:
            cprint(
                string=f"\nAugment {self.name} dataset from (N={len(self.mri)} -> "
                f"{len(self.mri) + len(self.sics2augment)}) ...\n",
                col="b",
            )
            self._mri, self._target = augment_mriset(
                samples=self.sics2augment, mris=self.mri, targets=self.target, transform_types=self._transform_types
            )

    def add_bg_noise(self, noise_scalar: float | None = None) -> None:
        """
        Add background noise to the whole data set.

        :param noise_scalar: None OR recommended between [.008, .02[
        """
        if not self.bg_noise:
            self.bg_noise = True

        if noise_scalar is not None and (0.0 < noise_scalar <= 1.0):
            self.bg_noise = noise_scalar

        self._mri = add_bg_noise_mriset(
            mriset=self._mri, **{"noise_scalar": self.bg_noise} if isinstance(self.bg_noise, float) else {}
        )
        # **{"karg": val} if COND else {}: passes argkey-value pair if TRUE, otherwise sets default value
        # This is necessary, since self.bg_noise can be bool OR float

    def to_keras(self, verbose=False):
        """Return all MRIs and target data in one tensor, respectively."""
        if verbose:
            cprint(string=f"Make {self.name} data keras-readable ...\n", col="b")
        xs = np.array(list(self._mri.values()))  # (N, X, Y, Z)
        xs = np.expand_dims(xs, axis=xs.ndim)  # (..., 1)
        ys = np.array(list(self._target.values()))

        if self._target_encoding == "softmax":
            from tensorflow.keras.utils import to_categorical

            ys = to_categorical(ys, dtype=int)  # onehot-encoding: e.g, [0, 0, 1, 0] (4 classes)

        return xs, ys


@lru_cache(maxsize=4)
def load_study_table(
    exclusion: bool = True, follow_up: bool = False, specify_vars: list[str] | None = None
) -> pd.DataFrame:
    """
    Load study table with baseline or follow-up data.

    :param exclusion: True: remove SICs w.r.t exclusion criteria
    :param follow_up: True: use LIFE follow-up data
    :param specify_vars: list of variables which shall be in table (columns), e.g. "age", "bmi", ...
    :return: study table
    """
    if follow_up:
        study_table = load_table_with_follow_up_age()
        study_table["SIC_FS"] = study_table.index
        study_table = study_table.reset_index(drop=True)
        study_table = study_table.drop(columns=["...", "...", "..."])

        if exclusion:
            # TODO: define exclusion in follow-up  # noqa: FIX002
            msg = "For follow-up data, SICs to exclude are not defined yet!"
            raise NotImplementedError(msg)

    else:  # baseline
        table_name = Path(
            paths.PROJECT_ROOT, "data", "subject_tables", f"sic_tab{'_reduced' if exclusion else ''}.csv"
        )

        if not table_name.is_file():
            study_table = load_raw_study_table(exclusion=exclusion, full_table=False, drop_logfile=False)

            # Create SIC_FS variable in study_tab, and correct naming confusions
            study_table["SIC_FS"] = study_table["SIC"]  # init
            study_table = study_table.drop(labels="SIC", axis=1)
            study_table.to_csv(table_name)

        else:
            study_table = pd.read_csv(table_name, index_col=0)

    # Return only specified columns (if provided)
    if specify_vars is not None:
        if not isinstance(specify_vars, list):
            msg = "'specify_vars' must be list or None!"
            raise AssertionError(msg)
        not_in_table = [v for v in specify_vars if v not in study_table.columns]
        in_table = [v for v in specify_vars if v in study_table.columns]

        if len(not_in_table) > 0:
            cprint(
                f"Following variables were not found in LIFE {'follow-up' if follow_up else 'base-line'} table:",
                col="r",
            )
            print(not_in_table)

        study_table = study_table[["SIC_FS", *in_table]]

    return study_table


def load_table_with_follow_up_age() -> pd.DataFrame:
    """
    Write age of a SIC at follow-up data-acquisition in table(s).

    :param update: True: check the follow-up folder for SICs which are not in the table.
    :return: table with age information in LIFE follow-up
    """
    p2table = Path(paths.PROJECT_ROOT, "data", "subject_tables", "sic_tab_fu.csv")
    # ...
    return pd.read_csv(p2table, index_col=0)


@only_mpi
@lru_cache(maxsize=1)
def load_sic_converter_table() -> pd.DataFrame:
    """Load SIC converter table."""
    path_to_pseudo_tab = Path(paths.data.life.CONVERSION_TAB)
    return pd.read_csv(path_to_pseudo_tab)


@only_mpi
def convert_id(life_id: str) -> str:
    """
    For given SIC return new/old SIC.

    :param life_id: new/old SIC
    :return: old/new SIC
    """
    tab = load_sic_converter_table()
    if life_id.startswith("..."):
        return tab[tab.sic == life_id].pseudonym.to_numpy().item()
    return tab[tab.pseudonym == life_id].sic.to_numpy().item()


def load_sic_raw_mri(
    _sic: str, mri_sequence: str, follow_up: bool, brain_masked: bool, reorient: bool, path_only: bool = False
) -> nib.Nifti1Image | Path:
    """
    Load raw NifTis from server.

     * T1/MPRAGE:
     * T2/FLAIR
     * SWI

    :param _sic: SIC
    :param mri_sequence: which MRI sequence to load
    :param follow_up: whether to load LIFE follow-up data
    :param brain_masked: whether to return brain-masked image or not
    :param reorient: whether to reorient MRI to global project space
    :param path_only: return only the path to the raw file
    :return: raw MRI OR path to it
    """
    bids = True  # remove after all life reformatted to BIDS

    sequence_pattern = {
        "t1": "T1w.nii" if bids else "MPRAGE",
        "flair": "FLAIR.nii" if bids else "_t2_",
        "swi": f"{'fu' if follow_up else 'bl'}_swi.nii" if bids else "_SWI_Images_",
    }

    if mri_sequence.lower() not in sequence_pattern:
        msg = f"'{mri_sequence}' unknown OR not implemented yet."
        raise ValueError(msg)

    fc = sequence_pattern[mri_sequence.lower()]  # file criterion
    file_pattern = f"sub-{convert_id(life_id=_sic)}_*{fc}*" if bids else f"S[0-9]*{fc}*.nii*"

    p2data = Path(
        ("..." if check_system() == "MPI" else paths.PROJECT_ROOT),
        f"Data/mri/{_sic}/{'followup' if follow_up else 'baseline'}/{mri_sequence.lower()}/raw/",
    )  # project data folder

    no_link = False  # init, to check for symbolic links to raw data in p2data
    image_file, p2mri = None, None  # init

    if p2data.is_dir():
        fname = list(p2data.glob(pattern=file_pattern))
        if len(fname) > 1:
            msg = f"Too many files for '{p2data / file_pattern}'!"
            raise AssertionError(msg)
        p2mri = None if len(fname) == 0 else fname[0]  # is max-length == 1
        if p2mri is not None:
            # Load MRI file
            image_file = nib.load(p2mri)

        else:
            no_link = True
    else:
        no_link = True

    if no_link and check_system() == "MPI":  # Search for the raw file in LIFE-raw folder
        p2raw = "..."
        if bids:
            p2sic = Path(
                p2raw, "bids", f"sub-{convert_id(life_id=_sic)}", f"ses-{'fu' if follow_up else 'bl'}", "anat"
            )

            if Path(str(p2sic).replace("/anat", "2/anat")).is_dir():
                p2sic = Path(str(p2sic).replace("/anat", "2/anat"))
                cprint(
                    string=f"For '{_sic}' ({convert_id(life_id=_sic)}) there were 2 folders found. "
                    f"Proceeding with data from the most recent folder:\n'{p2sic}'",
                    col="y",
                )

            if p2sic.is_dir():
                for p2mri in p2sic.glob(file_pattern):
                    # Load MRI file
                    image_file = nib.load(p2mri)

        else:
            p2sic = Path(p2raw, "...", _sic)

            for p2mri in p2sic.glob("**/" + file_pattern):
                # Only use baseline or follow-up data

                if follow_up:
                    if int(str(p2mri).split(f"/{_sic}_")[1][0:4]) < START_YEAR_FOLLOW_UP:
                        continue
                elif int(str(p2mri).split(f"/{_sic}_")[1][0:4]) >= START_YEAR_FOLLOW_UP:
                    continue

                # Load MRI file
                image_file = nib.load(p2mri)

        # Create symlink in project folder
        if p2mri is not None:
            p2data.mkdir(exist_ok=True, parents=True)
            (p2data / p2mri.name).symlink_to(target=p2mri, target_is_directory=False)

    if brain_masked and (image_file is not None):
        # For brain-masked version:

        bm_fname = f"{mri_sequence.lower()}_raw_brain_masked.nii.gz"
        p2mri = p2data / bm_fname

        if p2mri.is_file():
            image_file = nib.load(p2mri)

        else:
            # Create a brain-masked version of the raw image (if not there yet)
            bm_file = get_raw_brainmask(sic=_sic, mri_sequence=mri_sequence, follow_up=follow_up)
            image_file = nib.Nifti1Image(
                dataobj=image_file.get_fdata() * bm_file.get_fdata(), affine=image_file.affine
            )  # Create a masked version of the image

            # Write the brain-masked version to a file
            image_file.to_filename(filename=p2mri)

    if reorient and (image_file is not None):
        image_file = file_to_ref_orientation(image_file=image_file)

    if path_only:
        return p2mri
    return image_file


def age_of_sic(sic: str, follow_up: bool) -> int:
    """
    Get the age of a SIC.

    :param sic: SIC
    :param follow_up: whether age at follow-up data acquisition
    :return: age
    """
    age_col = "AGE_FU" if follow_up else "AGE_FS"
    study_table = load_study_table(exclusion=False, follow_up=follow_up)[["SIC_FS", age_col]]
    return study_table.loc[sic == study_table.SIC_FS][age_col].to_numpy().item()


def load_sic_mri(
    _sic: str,
    mri_sequence: str = "t1",
    follow_up: bool = False,
    bm: bool | None = None,
    norm: bool | None = None,
    regis: bool | None = None,
    raw: bool = False,
    dtype: type = np.float16,
    compressed: bool = False,
    as_numpy: bool = True,
    raiserr: bool = False,
) -> tuple[str, nib.Nifti1Image | np.ndarray]:
    """
    Load MR image of single SIC.

    for t1:
    T1.mgz / brainmask.mgz are intensity normalized
    See: https://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllDevTable

    :param _sic: SIC
    :param mri_sequence: MRI sequence (T1, FLAIR, SWI)
    :param follow_up: True: load follow-up MRI
    :param bm: True: apply brain-mask
    :param norm: True: normalize image
    :param regis: True: load MRI in MNI space
    :param raw: True: load raw image
    :param dtype: data type of image
    :param compressed: whether to compress image
    :param as_numpy: whether to return MRI as numpy array, or as NifTi
    :param raiserr: True: if image not findable raise an error
    :return: MR image of SIC, if not available raise error or return None
    """
    image_data = None  # init

    if raw:
        image_file = load_sic_raw_mri(
            _sic=_sic, mri_sequence=mri_sequence, follow_up=follow_up, brain_masked=bm, reorient=True
        )

        if (image_file is not None) and as_numpy:
            image_data = image_file.get_fdata(caching="unchanged")

    elif mri_sequence.lower() == "t1" and not regis:
        # Set MRI path to pre-processed data
        # all corrected are in the path below (cf /freesurfer_correction/)
        path_to_fs_all = ".../freesurfer_all/"
        mri_path = Path(path_to_fs_all, convert_id(_sic) + ("_fu" if follow_up else ""), "mri")

        file_name = "brain.finalsurfs.mgz" if norm else "brainmask.mgz" if bm else "T1.mgz"  # "/orig/001.mgz"

        if Path(mri_path, file_name).is_file():
            image_file = nib.load(Path(mri_path, file_name))
            # Re-transform to global project orientation space
            image_file = file_to_ref_orientation(image_file=image_file)  # FS standard
            image_data = image_file.get_fdata(caching="unchanged") if as_numpy else image_file

    elif mri_sequence.lower() in {"t1", "flair", "swi"}:
        p2data = Path(
            (".../Data" if check_system() == "MPI" else Path(paths.PROJECT_ROOT, "data")),
            f"mri/{_sic}/",
        )

        mri_path = Path(
            p2data,
            "followup" if follow_up else "baseline",
            f"{mri_sequence.lower()}" if mri_sequence.lower() != "t1" else "",
        )

        if regis:  # nonlinear registered in MNI space
            file_name = f"{mri_sequence.upper()}_brain2mni.nii.gz"
        else:
            # Only for non-T1 data
            file_name = mri_sequence.lower() + "_in_t1-space.nii.gz"

        if not Path(mri_path, file_name).is_file():
            if regis:
                from relevancerelated.dataloader.mri_registration import (
                    register_mri_sequence_in_t1_space_to_mni,
                )

                register_mri_sequence_in_t1_space_to_mni(sic=_sic, mri_sequence=mri_sequence, follow_up=follow_up)
            else:
                from relevancerelated.dataloader.mri_registration import register_native_to_t1_space

                register_native_to_t1_space(sic=_sic, mri_sequence=mri_sequence, follow_up=follow_up)

        if Path(mri_path, file_name).is_file():
            image_file = nib.load(Path(mri_path, file_name))
            # Re-transform to global project orientation space
            image_file = file_to_ref_orientation(image_file=image_file)  # FS standard

            if bm and not regis:  # MNI data already brain-masked
                brainmask = get_t1_brainmask(sic=_sic, follow_up=follow_up)
                image_file = nib.Nifti1Image(
                    dataobj=image_file.get_fdata() * brainmask.get_fdata(), affine=image_file.affine
                )  # Create a masked version of the image

            image_data = image_file.get_fdata(caching="unchanged") if as_numpy else image_file

    else:
        msg = f"mri_sequence '{mri_sequence}' unknown or not implemented yet."
        raise ValueError(msg)

    # If there is no file
    if image_data is None:
        if raiserr:
            msg = f"For {_sic} no {mri_sequence}-file found."
            raise FileNotFoundError(msg)
        cprint(
            string=f"For {_sic} no {'follow-up' if follow_up else 'baseline'} {mri_sequence}-file was "
            f"found (bm={bm}, norm={norm}, regis={regis}, raw={raw}).",
            col="r",
        )

    elif compressed:
        if not as_numpy:
            msg = "For compressed MRIs, image can only be returned as numpy array."
            raise ValueError(msg)
        image_data = compress_mri(
            x3d=image_data, space="mni" if regis else "raw" if raw else "fs", mri_sequence=mri_sequence
        )
        dtype = image_data.dtype

    # None no astype
    return _sic, (image_data.astype(dtype=dtype) if (image_data is not None and as_numpy) else image_data)


def compute_mean_mri(
    sic_list: list[str] | np.ndarray[str],
    follow_up: bool,
    brain_mask: bool,
    normalized: bool,
    mri_sequence: str,
    as_nifti: bool,
) -> nib.Nifti1Image:
    """
    Create a mean MRI for a given sequence.

    Mean MRI will be returned in MNI space.

    :param sic_list: list of SICs for average MRI
    :param follow_up: True: use LIFE follow-up data
    :param brain_mask: True if only brain should be returned
    :param normalized: whether MRIs should be normalized
    :param mri_sequence: T1, FLAIR, or SWI
    :param as_nifti: return as NiFTi object
    :return: average MRI (MNI)
    """
    if as_nifti:
        mris = []
        for sic in sic_list:
            mris.append(
                load_sic_mri(
                    _sic=sic,
                    mri_sequence=mri_sequence,
                    follow_up=follow_up,
                    bm=brain_mask,
                    norm=normalized,
                    regis=True,
                    as_numpy=not as_nifti,
                )[-1]
            )
            if mris[-1] is None:
                mris.pop(-1)

        return file_to_ref_orientation(nl.image.mean_img(nib.concat_images(mris)), reference_space="LAS")

    # as numpy
    msg = "Mean MRI as numpy output is not implemented yet."
    raise NotImplementedError(msg)


def get_mni_average(
    mri_sequence: str,
    follow_up: bool,
    sic_list: list[str] | np.ndarray[str] | None = None,
) -> np.ndarray:
    """
    Get the average MR image in a specific MRI sequence in MNI.

    Get this for the full cohort or a given subset.
    :param mri_sequence: MRI sequence (T1, FLAIR, SWI, ...)
    :param follow_up: True: use LIFE follow-up data
    :param sic_list: [list] of SICs OR None: full cohort
    :return: average MR image, pruned normed, and reoriented
    """
    if mri_sequence.lower() not in mri_sequences:
        msg = f"{mri_sequence} must be in {mri_sequences}!"
        raise ValueError(msg)

    mriset_name = f"{mri_sequence.lower()}MNIbmnp_sets.pkl"
    if mri_sequence.lower() != "t1":
        mriset_name = mriset_name.replace("bmnp", "bmp")
        if follow_up:
            mriset_name = mriset_name.replace(".pkl", "_fu.pkl")
    p2avgimg = mriset_name.replace("_sets", "_mean_image")
    if sic_list is None and Path("./TEMP", p2avgimg).is_file():
        return load_obj(name=p2avgimg, folder="./TEMP/", functimer=False)

    mri_set = load_obj(name=mriset_name, folder="./TEMP/")

    if sic_list is not None:
        # Exclude
        for sic in list(mri_set.keys()):
            if sic not in sic_list:
                del mri_set[sic]

    # Create average
    avg_img = np.zeros(shape=mri_set[next(iter(mri_set.keys()))].shape)
    ctn = 0
    for sic in mri_set:
        if mri_set[sic] is not None:
            avg_img += mri_set[sic]
            ctn += 1
    avg_img /= ctn

    if sic_list is None:
        save_obj(obj=avg_img, name=p2avgimg, folder="./TEMP/")

    return avg_img


def get_mni_template(
    low_res: bool = True,
    reorient: bool = True,
    prune: bool = True,
    norm: tuple[int, int] = (0, 1),
    mask: bool = False,
    original_template: bool = True,
    as_nii: bool = False,
) -> nib.Nifti1Image | np.ndarray:
    """
    Get MNI template.

    :param low_res: True: 2mm; False: 1mm isotropic resolution
    :param reorient: whether to reorient to project space
    :param prune: whether to prune image, or keep original MNI shape
    :param norm: whether to normalize image values between 0-1
    :param mask: return only 0/1 mask
    :param original_template: With v.0.8.1 nilearn reshaped its MNI template (91,109,91) -> (99,117,95),
                              if toggled True: this functions uses the previous template
    :param as_nii: return template as NifTi
    :return: MNI image
    """
    # MNI152 since 2009 (nonlinear version)

    if norm[0] != 0:
        msg = "Function works only for zero-background (i.e. min-value = 0)!"
        raise AssertionError(msg)

    if low_res:
        # Nilearn has 2mm resolution
        mni_temp = load_mni152_template(resolution=2)
        if original_template:
            # With v.0.8.1, nilearn:
            #  1) reshaped (91, 109, 91) -> (99, 117, 95) &
            #  2) changed the affine to np.array([[2., 0., 0., -98.],
            #                                     [0., 2., 0., -134.],  # noqa: ERA001
            #                                     [0., 0., 2., -72.],  # noqa: ERA001
            #                                     [0., 0., 0., 1.]])
            #  3) rescaled (0-8339) -> (0-255) the MNI template
            #  https://github.com/nilearn/nilearn/blob/d91545d9dd0f74ca884cc91dca751f8224f67d99/doc/changes/0.8.1.rst#enhancements
            mni_temp = resample_img(
                img=mni_temp,
                target_affine=np.array([
                    [-2.0, 0.0, 0.0, 90.0],
                    [0.0, 2.0, 0.0, -126.0],
                    [0.0, 0.0, 2.0, -72.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                target_shape=(91, 109, 91),
            )

            # Remove very small values from interpolation
            mni_temp = nib.Nifti1Image(
                dataobj=mni_temp.get_fdata().round(3), affine=mni_temp.affine, header=mni_temp.header
            )

    else:
        # ANTs template has 1 mm resolution
        from ants import get_ants_data  # , image_read

        mnipath = get_ants_data("mni")
        mni_temp = nib.load(mnipath)  # ants.image_read(mnipath)

    # Re-orient to global/project orientation space
    if reorient:
        mni_temp = file_to_ref_orientation(image_file=mni_temp)

    if as_nii:
        if prune or mask:
            cprint(string="No pruning or masking is done for MNI templates returned as NifTi!", col="r")
        return mni_temp
    if prune:
        global_max = get_global_max_axis(space="mni" if low_res else "freesurfer")
        mni_temp = prune_mri(x3d=mni_temp.get_fdata(), make_cube=True, max_axis=global_max)
    else:
        mni_temp = mni_temp.get_fdata()

    # Normalize
    mni_temp = normalize(array=mni_temp, lower_bound=norm[0], upper_bound=norm[1])

    # Brain mask
    if mask:
        mni_temp[mni_temp > 0] = 1

    return mni_temp


def load_mri_set(
    sic_list: iter,
    mni: bool,
    brain_mask: bool,
    normalized: bool,
    raw: bool,
    prune: bool,
    mri_sequence: str = "T1",
    follow_up: bool = False,
    dtype: type = np.float16,
    mri_set_name: str | None = None,
    save_set: bool = False,
) -> dict:
    """
    Load MRIs for given list of SICs.

    :param sic_list: list of SICs
    :param mni: whether to load data in MNI space
    :param brain_mask: whether to load brain-only data
    :param normalized: whether MRI is normalized
    :param raw: whether MRI is in raw form
    :param prune: whether MRIs are to be pruned
    :param mri_sequence: MRI sequence
    :param follow_up: whether LIFE follow-up is to be loaded
    :param dtype: data type of MRIs (important for compression)
    :param mri_set_name: name of MRI set
    :param save_set: name of MRI set. If found in storage, the set (*.pkl) will be loaded instead of
                     composing the set from single MRIs from the storage
    :return: MRI set [dict]
    """
    # Variable checks
    if save_set and mri_set_name is None:
        msg = "'set_name' is required in order to save MRI set in './TEMP/'."
        raise ValueError(msg)
    if mri_set_name is not None and not Path(f"./TEMP/{mri_set_name}.pkl").is_file():
        msg = f"'{mri_set_name}.pkl' wasn't found. If you want to load data from server, turn 'save_set=True'"
        raise ValueError(msg)

    # Load data
    if Path(f"./TEMP/{mri_set_name}.pkl").is_file():
        cprint(string=f"\nLoading {mri_sequence.upper()}-scans from {mri_set_name}.pkl ...", col="b")

        # Load external file
        mri_set = load_obj(name=mri_set_name)

    else:  # if file MRIset.pkl doesn't exist:
        cprint(string=f"\nLoading {mri_sequence.upper()}-scans from server ...", col="b")
        start_time_load = datetime.now()

        # Load MRIs for all available SICs independently:
        allsics = (
            load_study_table(
                exclusion="_all" not in mri_set_name,  # saves diskspace
                follow_up=follow_up,
                specify_vars=None,
            )["SIC_FS"].to_list()
            if save_set
            else sic_list.copy()
        )
        # In case we save the set, we load the data for all SICs and then remove those MRIs, which
        # are not needed.
        if mri_sequence.lower() == "swi":
            allsics = [sic for sic in allsics if sic != "..."]

        # For raw data, we need to compress data during loading phase of single MRIs
        compressed = raw

        # Create a dictionary with all T1 files per subject
        with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 1) as executor:  # parallel processes
            mri_set = executor.map(
                load_sic_mri,
                list(allsics),  # arg 0: _sic
                [mri_sequence] * len(allsics),  # arg 1: mri_sequence
                [follow_up] * len(allsics),  # arg 2: follow_up
                [brain_mask] * len(allsics),  # arg 3: bm
                [normalized] * len(allsics),  # arg 4: norm
                [mni] * len(allsics),  # arg 5: regis
                [raw] * len(allsics),  # arg 6: raw
                [dtype] * len(allsics),  # arg 7: dtype
                [compressed] * len(allsics),
            )  # arg 8: compressed (arg 7,8 = Default)

        mri_set = dict(tuple(mri_set))
        # Delete those without MRI
        for sic in list(mri_set.keys()):
            if mri_set[sic] is None:
                del mri_set[sic]

        print(
            f"Duration of loading {mri_sequence.upper()} datasets "
            f"{chop_microseconds(datetime.now() - start_time_load)} [h:m:s]"
        )

        # Prune
        # check whether pruning could be done in `load_sic_mri`
        if prune:
            global_max = get_global_max_axis(
                space="mni" if mni else "raw" if raw else "freesurfer",
                mri_sequence=mri_sequence,
                per_axis=raw,  # == True for raw else False
                padding=True,
            )
            print(
                f"\nBefore pruning, {mri_sequence.upper()} images are of shape: {next(iter(mri_set.values())).shape}"
            )
            cprint(string="Pruning the dataset...", col="y")
            start_time_load = datetime.now()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Threading here superior over Pooling [tested]
                mri_set = executor.map(
                    prune_mri,  # prune function
                    mri_set.items(),  # arg: x3d
                    [not raw] * len(mri_set),  # arg: make_cube (not for raw)
                    [global_max] * len(mri_set),  # arg: max_axis
                    [0] * len(mri_set),
                )  # arg: padding

            mri_set = dict(tuple(mri_set))

            print(
                f"Duration of pruning all {mri_sequence.upper()} images (via threading) "
                f"{chop_microseconds(datetime.now() - start_time_load)} [h:m:s]"
            )
            print(f"{mri_sequence.upper()} images are now of shape: {next(iter(mri_set.values())).shape}")

        # Due to memory issues, MRIs in the set must be of dtype 'uint8', and NOT 'float' (!)
        mri_set = set_dtype_mriset(mri_set, dtype=np.uint8)

        # Save file externally (to save time for loading the data later)
        if save_set:
            Path("./TEMP").mkdir(exist_ok=True)
            cprint(string=f"\nSaving the MRIset to ./TEMP/{mri_set_name}.pkl ...", col="y")
            save_obj(obj=mri_set, name=mri_set_name)

    # Remove MRIs, which are not needed, &/or do not exist
    for sic in list(mri_set.keys()):
        if sic not in sic_list or mri_set[sic] is None:
            del mri_set[sic]

    return mri_set


# @function_timed(ms=True)
def norm_imageset(dataset: np.ndarray | None, norm: tuple = (0, 1)) -> np.ndarray | None:
    """
    Normalize images of whole dataset in given range.

    :param dataset: dataset
    :param norm: normalization range
    :return: normalized dataset
    """
    if norm is not None and dataset is not None:
        try:
            if norm == (0, 1):
                if dataset.max() > 1.0:  # assumes that dataset.min() 0
                    dataset = (dataset / 255.0).astype(np.float16)  # float32 (not necessary)
                    # / dataset.max(), but could vary depending on dataset
            else:
                dataset = normalize(
                    array=dataset,
                    lower_bound=norm[0],
                    upper_bound=norm[1],
                    global_max=255.0 if dataset.max() > 1.0 else 1.0,
                )

        except MemoryError:
            if norm == (0, 1):
                if dataset.max() > 1.0:
                    for idx in range(dataset.shape[0]):
                        dataset[idx] /= 255.0

                dataset = (dataset / 255.0).astype(np.float16)

            else:
                for idx in range(dataset.shape[0]):
                    dataset[idx] = normalize(
                        array=dataset[idx],
                        lower_bound=norm[0],
                        upper_bound=norm[1],
                        global_max=255.0 if dataset.max() > 1.0 else 1.0,
                    )

        if norm[1] == 255.0:  # i.e., data was (re-)normalized to (0, 255) # noqa: PLR2004
            dataset = np.round(dataset).astype(np.uint8)  # compress data

    return dataset  # normed


def create_region_mask(region: str, prune: bool = True, reduce_overlap: bool = True) -> np.ndarray:
    """
    Create an image mask over a specified brain region.

    :param region: brain region
    :param prune: whether image is to be pruned
    :param reduce_overlap: reduce the overlap between different regions.
    :return: mask for the brain region
    """
    from relevancerelated.dataloader.atlases import get_atlas, prune_atlas

    atl, _ = get_atlas(  # _ = label
        name=brain_regions[region.lower()],
        prob_atlas=False,  # is same as for: prob_atlas=True
        reduce_overlap=reduce_overlap,
    )

    region_mask = prune_atlas(atl) if prune else atl  # prune asserts MNI
    # region_mask = np.sum(region_mask, axis=3) # only for: prob_atlas=True

    region_mask[region_mask > 0] = 1  # region mask

    return region_mask


def stamp_region(
    dataset: dict[str, np.ndarray] | np.ndarray, region_mask: np.ndarray
) -> dict[str, np.ndarray] | np.ndarray:
    """Stamp region."""
    print("Stamping the given brain region ...\n")

    if isinstance(dataset, dict):
        for sic in dataset:  # takes about 12 sec
            dataset[sic] *= region_mask.astype(dataset[sic].dtype)

    else:  # isinstance(dataset, np.ndarray)
        for i_sic, mri in enumerate(dataset):  # takes about 12 sec
            dataset[i_sic, ..., -1] = mri[..., -1] * region_mask.astype(mri.dtype)

    return dataset


def write_list_of_available_mris(follow_up: bool) -> None:
    """
    For each SIC write information in table.

    Whether the following MRI data is available (LIFE baseline only):
        - 'T1_brain2mni.nii.gz': regis=True
        - 'brain.finalsurfs.mgz' (regis==None)
        - 'brainmask.mgz',  bm=True (regis & norm == None)
        - 'T1.mgz': all none
        - SWI raw  (if the raw is available, the derivatives are there too or can be computed)
        - FLAIR raw

    Note: Function needs to be extended if other MRI modalities are required

    :param follow_up: True: use LIFE follow-up data.
    :return None
    """
    # Load table with all SICs
    full_study_tab = load_study_table(exclusion=False, follow_up=follow_up)["SIC_FS"]

    # Prepare table
    available_data_tab = pd.DataFrame(
        columns=[
            "T1_brain2mni.nii.gz",  # regis
            "T1.mgz",  # all none
            "brainmask.mgz",  # bm
            "brain.finalsurfs.mgz",
            "FLAIR_raw",
            "SWI_raw",
        ]
    )  # norm
    available_data_tab = pd.DataFrame(full_study_tab).set_index("SIC_FS").add(available_data_tab)

    # Fill table with Information
    start = datetime.now()  # time the whole loop (due to long duration)

    for idx, sic in enumerate(full_study_tab):
        # sic  # str
        for regis in [True, False]:
            if regis:
                available_data_tab.loc[sic]["T1_brain2mni.nii.gz"] = (
                    0 if load_sic_mri(_sic=sic, follow_up=follow_up, regis=regis, as_numpy=False)[1] is None else 1
                )

            else:  # If regis False
                for norm in [True, False]:
                    if norm:
                        available_data_tab.loc[sic]["brain.finalsurfs.mgz"] = (
                            0
                            if load_sic_mri(_sic=sic, follow_up=follow_up, norm=norm, regis=regis, as_numpy=False)[1]
                            is None
                            else 1
                        )

                    else:  # If norm (also) False
                        for bm in [True, False]:
                            available_data_tab.loc[sic]["brainmask.mgz"] = (
                                0
                                if load_sic_mri(
                                    _sic=sic, follow_up=follow_up, norm=norm, bm=bm, regis=regis, as_numpy=False
                                )[1]
                                is None
                                else 1
                            )
                        available_data_tab.loc[sic]["T1.mgz"] = (
                            0
                            if load_sic_mri(
                                _sic=sic, follow_up=follow_up, norm=norm, bm=bm, regis=regis, as_numpy=False
                            )[1]
                            is None
                            else 1
                        )

        for seq in ["flair", "swi"]:
            available_data_tab.loc[sic][f"{seq.upper()}_raw"] = (
                0
                if load_sic_raw_mri(
                    _sic=sic, mri_sequence=seq, follow_up=follow_up, brain_masked=False, reorient=False, path_only=True
                )
                is None
                else 1
            )

        # Report remaining time
        loop_timer(start_time=start, loop_length=len(full_study_tab), loop_idx=idx)

    # # Save table
    table_name = Path(paths.PROJECT_ROOT, f"data/subject_tables/sic_available_mri_tab{'_fu' if follow_up else ''}.csv")
    available_data_tab.to_csv(table_name)


def get_table_of_available_mris(follow_up: bool) -> pd.DataFrame:
    """
    Get the table of available MRIs.

    :param follow_up: True: use LIFE follow-up data.
    :return: table of available MRIs
    """
    table_name = Path(paths.PROJECT_ROOT, f"data/subject_tables/sic_available_mri_tab{'_fu' if follow_up else ''}.csv")
    return pd.read_csv(table_name, index_col="SIC_FS")


def split_set_sizes(list_all_sics: iter, split_prop: list) -> tuple[int, int, int]:
    """
    Split data into training, validation and (if requested) test sets.

    :param list_all_sics: list of all SICs to split in the subsets
    :param split_prop: list, indicating data split, e.g. [.8, .1, .1] (must add up to 1)
    :return: size of each split, i.e., subset of data
    """
    # split such that distributions of targets match in all sets
    if sum(split_prop) != 1:
        msg = "split_prop must add up to 1"
        raise ValueError(msg)

    if len(split_prop) == 2:  # in case no test split is given # noqa: PLR2004
        split_prop.append(0.0)

    _ntrain = int(len(list_all_sics) * split_prop[0])
    _nvali = int(len(list_all_sics) * split_prop[1])
    _ntest = int(len(list_all_sics) * split_prop[2])

    if _ntest == 0 and _nvali != 0:
        print(f"Only training ({split_prop[0]}) and validation ({split_prop[1]}) set.")
    elif _ntest == 0 and _nvali == 0:
        print("Only training set.")

    # Adjust set-size for leftovers
    while sum([_ntrain, _nvali, _ntest]) != len(list_all_sics):
        if sum([_ntrain, _nvali, _ntest]) > len(list_all_sics):
            _ntrain -= 1
        else:  # < len(list_all_sics)
            _ntrain += 1

    cprint(string="\nFinal set sizes:", fm="ul")
    print(f"\ttrain:\t{_ntrain}\n\tval:\t{_nvali}\n\ttest:\t{_ntest}")

    return _ntrain, _nvali, _ntest


def create_split_dict(life_data: dict[str, DataSet]) -> dict[str, list[str]]:
    """Create split dict."""
    # Note life_data[subset].sics does not include augmented SICs
    return {subset: life_data[subset].sics for subset in ["train", "validation", "test"]}


def datasplit_for_classification(
    target: str, follow_up: bool, study_table: pd.DataFrame | None = None, correct_for: str | None = None
) -> dict[str, list[str]]:
    """
    Compute data split for classification tasks.

    :param target: target variable, e.g. 'bmi'
    :param follow_up: True: use LIFE follow-up data
    :param study_table: LIFE study table
    :param correct_for: variable to correct for, e.g. 'age'
    :return: data split for classification
    """
    print(f"\nPreparing now data for binary classification of '{target.upper()}' ...\n")

    # # Load study table
    if study_table is None:
        study_table = load_study_table(follow_up=follow_up)  # Note: exclusion=True

    study_table = study_table.set_index("SIC_FS")
    col_target = next(col for col in study_table.columns if target in col.lower())  # colname for target

    # # Set Groups
    try:
        low_bound, up_bound = binary_boundaries[target.lower()]  # TODO: only binary so far  # noqa: FIX002
    except KeyError:
        low_bound, up_bound = np.sort(study_table[target].unique())  # in case target == "binary_..."

    # # Equalize group size
    group_low = study_table[study_table[col_target] <= low_bound]
    group_up = study_table[study_table[col_target] >= up_bound]

    # Get the size of each group
    print(f"\tN\u2080 of lower {target.upper()} group:", len(group_low))
    print(f"\tN\u2081 of upper {target.upper()} group:", len(group_up))

    # Balance groups e.g. of BMI w.r.t. other variable, e.g., AGE (advanced approach)
    if correct_for is not None:
        # Find column name in table corresponding to var which is to be corrected for:
        corr4_col = next(col for col in study_table.columns if correct_for.lower() in col.lower())

        # Extract variable for each group/class
        low_corr4 = group_low[corr4_col].copy()
        up_corr4 = group_up[corr4_col].copy()

        # Equalize group sizes while balancing w.r.t. given variable
        while True:
            # Discretize variable distribution in each class
            corr_var_range = (np.min([low_corr4.min(), up_corr4.min()]), np.max([low_corr4.max(), up_corr4.max()]))

            h_low = np.histogram(low_corr4, range=corr_var_range, bins=20)
            h_upper = np.histogram(up_corr4, range=corr_var_range, bins=20)
            # Then equalize these distributions between classes/groups

            # Get max difference of bin-sizes between distributions via argmax
            distr_diff = h_low[0] - h_upper[0]  # bin-size differences

            # Index the bin per group, which has the highest difference
            max_more_low = np.argwhere(distr_diff == np.amax(distr_diff)).flatten()[
                0 if low_corr4.median() < up_corr4.median() else -1
            ]
            max_more_upper = np.argwhere(distr_diff == np.amin(distr_diff)).flatten()[
                -1 if low_corr4.median() < up_corr4.median() else 0
            ]
            #  if more than 1 argmax, choose from left/right side, depending on diff. of group-medians

            # Define boundaries of the range from which we draw to drop a subject
            h_low_maxlo, h_upmaxlo = h_low[1][max_more_low], h_low[1][max_more_low + 1]
            h_low_maxup, h_upmaxup = h_upper[1][max_more_upper], h_upper[1][max_more_upper + 1]

            # If both groups have the same size: drop a subject per class
            if low_corr4.size == up_corr4.size:
                low_corr4 = low_corr4.drop(
                    low_corr4[((h_low_maxlo <= low_corr4) & (low_corr4 < h_upmaxlo))].sample(n=1, replace=False).index
                )

                up_corr4 = up_corr4.drop(
                    up_corr4[((h_low_maxup <= up_corr4) & (up_corr4 < h_upmaxup))].sample(n=1, replace=False).index
                )

            # If one class is bigger than the other: drop a subject for this class
            elif low_corr4.size > up_corr4.size:
                low_corr4 = low_corr4.drop(
                    low_corr4[((h_low_maxlo <= low_corr4) & (low_corr4 < h_upmaxlo))].sample(n=1, replace=False).index
                )

            # and vice versa
            else:
                up_corr4 = up_corr4.drop(
                    up_corr4[((h_low_maxup <= up_corr4) & (up_corr4 < h_upmaxup))].sample(n=1, replace=False).index
                )

            # print("Median low:", low_corr4.median(), "\nMedian upper:", up_corr4.median())  # noqa: ERA001

            if np.abs(up_corr4.median() - low_corr4.median()) < 0.5:  # TODO: refine threshold  # noqa: FIX002, PLR2004
                break

        group_low = group_low.loc[low_corr4.index]
        group_up = group_up.loc[up_corr4.index]

    # Throw out data such that both groups have the same size (naive approach)
    group_low = group_low.sample(n=len(group_low) - np.maximum(0, len(group_low) - len(group_up)))
    group_up = group_up.sample(n=len(group_up) - np.maximum(0, len(group_up) - len(group_low)))

    print("After equalizing group sizes:")
    print(f"\tFinal N\u2080 of lower {target.upper()} group:", len(group_low))
    print(f"\tFinal N\u2081 of upper {target.upper()} group:", len(group_up))

    # # Split data: train (85%), validation (5%), test (10%)
    split_prop = [round(0.85 * len(group_low)), round(0.9 * len(group_low))]  # N0 == N1
    train_low, valid_low, test_low = np.split(
        ary=group_low.sample(frac=1),  # sample to shuffle
        indices_or_sections=split_prop,
    )
    train_up, valid_up, test_up = np.split(ary=group_up.sample(frac=1), indices_or_sections=split_prop)

    # Extract sics per subset
    train_sics = train_low.index.to_list() + train_up.index.to_list()
    valid_sics = valid_low.index.to_list() + valid_up.index.to_list()
    test_sics = test_low.index.to_list() + test_up.index.to_list()

    return {
        "train": sample(train_sics, k=len(train_sics)),  # shuffle data
        "validation": sample(valid_sics, k=len(valid_sics)),
        "test": sample(test_sics, k=len(test_sics)),
    }  # sics_split


def get_life_data(  # noqa: C901, PLR0912, PLR0913, PLR0915
    mri_sequence: str = "T1",
    region: str | None = None,
    brain_mask: bool = True,
    intensity_normalized: bool = True,
    prune: bool = True,
    mri_space: str = "fs",
    mri_scale: tuple | None = None,
    target: str = "age",
    target_scale: str = "linear",
    target_range: tuple | None = None,
    uniform_targets: bool = False,
    augment: str | None = None,
    transform_types: (list, str) = None,
    n_augment: int | None = None,
    bg_noise: (bool, float) = False,
    exclusion: bool = True,
    subset_fraction: float = 1.0,
    split_proportion: list | None = None,
    split_dict: dict[str, list[str]] | None = None,
    return_nones: bool = False,
    follow_up: bool = False,
    testmode: str | None = None,
    seed: bool = False,
    **kwargs,
) -> dict[str, DataSet]:
    """
    Prepare MRI data.

    :param mri_sequence: T1, FLAIR, SWI (DWI) [string]
    :param region: Load specific brain region only
    :param brain_mask: True: feed only brain (no skull, face etc.)
    :param intensity_normalized: True: feed normalized images w.r.t. image intensities
    :param prune: True: prune Zero-Padding (black areas) around the brain (global min-max cube-size is used)
    :param mri_space: 'fs': FreeSurfer-T1 space, OR 'mni' OR 'raw'
    :param mri_scale: scaling in range (low, high), if None load data without re-scaling.
    :param target: age or other variables (as string)
    :param target_scale: 'tanh': rescale to [-1, 1]; 'linear': keep; 'softmax': build classes
    :param target_range: set range for target (min, max) or None: full range
    :param uniform_targets: take only a subset of the data, such that targets are uniformly distributed
    :param augment: True: augment data by random sampling from the inverse target-distribution
    :param transform_types: None: applies randomly types of all transformations, OR
                                 subset of following list: all_manip_opt (see import)
    :param n_augment: number of augmented samples
    :param bg_noise: True: add noise to background (default scale, see add_background_noise()); else float
    :param exclusion: whole datasets (False) or remove dropout (True)
    :param subset_fraction: full set = 1., subset size defined by fraction, e.g., 0.5 equals half the set
    :param split_proportion: proportion of training, validation and test set, list e.g. [.8, .1, .1]
    :param split_dict: provide a lists of sics for each data subset, e.g. {"train": [sic1, sic3, ...], ...}
    :param return_nones: whether to return SICs in MRI-sets also with missing (None) data.
    :param follow_up: False: load base-line data, True: load follow-up data of LIFE
    :param testmode: noise/artificial data: 'all_zero', 'permute_input', 'permute_target', 'simulated'
    :param seed: For reproducibility
    :return: LIFE MRI dataset (train and validation set)
    """
    # # Check given arguments
    if mri_space.lower() not in {"fs", "raw", "mni"}:
        msg = "mri_space must be 'fs', 'raw', OR 'mni'"
        raise ValueError(msg)
    mri_space = mri_space.lower()

    if isinstance(target_range, str):  # Check target range and augment
        target_range = ast.literal_eval(target_range)  # e.g., '(40, 80)' => (40, 80), or 'None' => None

    if augment:
        augment = None if augment.lower() == "none" else augment
        if not (n_augment is None or float(n_augment).is_integer()):
            msg = "'n_augment' must be int OR None!"
            raise ValueError(msg)

    if transform_types:
        transform_types = transform_types if isinstance(transform_types, (list, np.ndarray)) else [transform_types]
        transform_types = [ttype.lower() for ttype in transform_types]
        if not all(ttype in all_manip_opt for ttype in transform_types):
            msg = f"'transform_types' must be None or subset of {all_manip_opt}."
            raise AssertionError(msg)

    if isinstance(region, str):
        if region.lower() not in brain_regions:
            msg = f"Given list of regions must be in {list(brain_regions.keys())} OR None"
            raise AssertionError(msg)
        if "mni" not in mri_space:
            msg = "Region specific training not implemented for native data yet."
            raise NotImplementedError(msg)

    if seed:
        np.random.seed(42)

    # # Define target variable:
    # include all variables of interest, name must match study_tab below
    target = target.lower()
    ls_of_vars = [
        "sex",
        "AGE_FU" if follow_up else "AGE_FS",
        "...",
        "lesionload",
        # ...
    ]

    # # Load table with comprehensive study data
    study_tab = load_study_table(exclusion=exclusion, follow_up=follow_up, specify_vars=ls_of_vars)

    # Find for target the corresponding column in the study table and adjust the column-name
    if np.any([target in var.lower() for var in ls_of_vars]):
        table_target = ls_of_vars[np.where([target in var.lower() for var in ls_of_vars])[0][0]]
        study_tab = study_tab.rename(columns={table_target: target})  # if target != table_target

    else:
        msg = f"Given target={target} not implemented yet!"
        raise ImportError(msg)

    # # Remove subjects which have no target-value or are not in given split_dict (if not None):
    subsets = ["train", "validation", "test"]
    if isinstance(split_dict, dict):
        if list(split_dict.keys()) != subsets:
            msg = f"'split_dict' must contain a (at least empty) list for each subset in {subsets}."
            raise AssertionError(msg)
        # List all SICs in given data split dictionary
        all_sics = [sic for sublist in split_dict.values() for sic in sublist]
        # Remove others SICs from table
        study_tab = study_tab.loc[study_tab.SIC_FS.isin(all_sics)].reset_index(drop=True)
        # drop=True: prevents creating column with old indices

    # Drop SICs which have no target
    if sum(np.isnan(study_tab[target])) > 0:
        cprint(
            string=f"\nFor {sum(np.isnan(study_tab[target]))} (of {len(study_tab[target])}) subjects there "
            f"is no {target.upper()} value (NaN). These subjects get removed from the dataframe ...",
            col="y",
        )
        study_tab = study_tab.dropna(subset=[target], axis=0).reset_index(drop=True)

    # # Take full or fraction of dataset, and shuffle its order
    # In case: pd.sample() is affected by np.random.seed()
    study_tab = study_tab.sample(frac=subset_fraction).reset_index(drop=True)  # shuffles also for frac=1.
    if subset_fraction != 1.0:
        cprint(string=f"Reduce to subset {study_tab.shape}: {subset_fraction} fraction of full dataset\n", col="y")

    # Adapt target range and distribution (if required)
    if target_range:
        if not isinstance(target_range, tuple):
            msg = "target_range must be tuple (min, max)"
            raise TypeError(msg)
        study_tab = study_tab[(target_range[0] <= study_tab[target]) & (study_tab[target] <= target_range[1])]

    if uniform_targets:
        n_bins = 12  # for 'age' ok, but adapt for other vars
        study_tab["target_bins"] = pd.cut(study_tab[target], bins=n_bins, right=False)
        min_n_bin_size = min(study_tab["target_bins"].value_counts())
        cprint(
            string=f"Sample targets uniformly distributed between {n_bins} bins of size n={min_n_bin_size} each.",
            col="y",
        )
        print(study_tab["target_bins"].value_counts())

        # Sample from each bin n subjects
        sel_sics = pd.DataFrame()
        for cat in study_tab["target_bins"].to_numpy().categories:
            sel_sics = sel_sics.append(study_tab[study_tab["target_bins"] == cat].sample(n=min_n_bin_size))

        study_tab = sel_sics  # Update study_table
        cprint(string="Now:", col="y")
        print(study_tab["target_bins"].value_counts())

    # If testmode: permute_target (other testmodes see below)
    if testmode == "permute_target":
        cprint(string="\nTestmode: permute target by shuffling targets across subjects ...", col="b")
        study_tab[target] = study_tab[target].sample(frac=1.0).reset_index(drop=True)

    # # Load MRI data
    # MRI dataset name:
    if (mri_sequence.lower() != "t1" or mri_space == "raw") and intensity_normalized:
        cprint(
            string=f"For {mri_sequence.upper()} in {mri_space.upper()}-space there is no "
            f"intensity-normalized data yet!\nArgument 'intensity_normalized' is set to False.",
            col="y",
        )
        intensity_normalized = False

    mri_set_name = (
        f"{mri_sequence.lower()}{mri_space.upper() if mri_space != 'fs' else ''}"
        f"{'bm' if brain_mask else ''}{'n' if intensity_normalized else ''}"
        f"{'p' if prune else ''}"
        f"{'fu' if follow_up else ''}_sets"
    )

    mri_set_name += f"_subfrac{subset_fraction}" if subset_fraction != 1.0 else ""
    mri_set_name += "_all" if not exclusion else ""

    # Load sets
    mri_dataset = load_mri_set(
        sic_list=study_tab["SIC_FS"].to_list(),
        mni=mri_space == "mni",
        brain_mask=brain_mask,
        normalized=intensity_normalized,
        raw=mri_space == "raw",
        prune=prune,
        mri_sequence=mri_sequence,
        follow_up=follow_up,
        mri_set_name=mri_set_name,
        save_set=True,
    )

    # If not all SICs have MRI:
    if len(mri_dataset) != len(study_tab):
        # Update study table, by removing the SICs which have no MRI
        study_tab = study_tab.loc[study_tab.SIC_FS.isin(mri_dataset.keys())].reset_index(drop=True)

    # If testmode adapt data (for 'permute_target' see above)
    if testmode:
        if testmode not in {"all_zero", "permute_input", "permute_target", "simulated"}:
            msg = "testmode must be (str): 'all_zero', 'permute_input', 'permute_target', 'simulated', or None"
            raise ValueError(msg)

        if testmode == "all_zero":
            cprint(string="\nTestmode: setting all input to zero ...", col="b")

            for sic in mri_dataset:
                mri_dataset[sic][:, :, :] = 0

        if testmode == "permute_input":
            cprint(string="\nTestmode: permute input by shuffling voxels within scans ...", col="b")

            start_t = datetime.now()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # [tested] Threading here inferior to Pooling: 0:05:47 vs. 0:01:16. loop: 0:05:35 [h:m:s]
                mri_dataset = executor.map(permute_mri, mri_dataset.items())
            mri_dataset = dict(tuple(mri_dataset))

            print(
                f"Time to permute all {len(mri_dataset)} MRIs: {chop_microseconds(datetime.now() - start_t)} [h:m:s]"
            )

        # Simulated data (here: contains meaningful information vs. in permutation)
        if testmode == "simulated":
            msg = "No testmode 'simulated' implemented yet."
            raise NotImplementedError(msg)

    # # Prepare target data
    # Save min&max of original target to be able to reconstruct after normalization
    tminmax = (min(study_tab[target]), max(study_tab[target]))  # ~ target_range

    # Adjust the type of the target (target_scale)
    # for regression: 'tanh' → rescale to range [-1,1]; or 'linear' → just pass
    # for classification: 'softmax': build classes
    if target_scale == "tanh":
        study_tab[target] = normalize(array=study_tab[target], lower_bound=-1, upper_bound=1)

    elif target_scale == "softmax":
        # Create new column in study-table for binarized target
        new_target = f"binary_{target}"

        # Encode target accordingly
        conditions = [
            (study_tab[target] <= binary_boundaries[target][0]),  # lower group
            (study_tab[target] >= binary_boundaries[target][1]),
        ]  # upper group
        choices = range(len(conditions))  # binary: [0, 1], or multiclass: [0, 1, 2, ...]
        study_tab[new_target] = np.select(conditions, choices, default=np.nan)

        target = new_target  # re-set target

        # Remove MRIs, which do not fit in given bins and remove corresponding SICs from table
        for sic in study_tab.SIC_FS[study_tab[target].isna()]:
            del mri_dataset[sic]
        study_tab = study_tab.loc[~study_tab[target].isna()].reset_index(drop=True)
        # TODO: Implement also for multi class vector, e.g. via pd.value_counts(target)  # noqa: FIX002

    # # Fill in target values
    # Create dictionary with target variable per subject
    list_of_sics = np.array(list(mri_dataset.keys()))  # == study_tab.SIC_FS.to_numpy()

    targ_sets = {_sic: study_tab[target].loc[_sic == study_tab.SIC_FS].to_numpy().item() for _sic in list_of_sics}

    # # Split data (default: 80% Train, 10% Validation, 10% Test)
    # If data-split dict is given use this:
    if isinstance(split_dict, dict):
        # Check the difference to available SICs in study table for given data split
        all_sics = pd.Series(all_sics)
        if not all_sics.isin(study_tab.SIC_FS).all():
            no_data_sics = all_sics[~pd.Series(all_sics).isin(study_tab.SIC_FS)]

            if not all(_sic is None for _sic in no_data_sics):
                cprint(
                    string=f"NOTE: The data for following {len(no_data_sics)} SIC(s) in given 'split_dict' "
                    f"are not available:\n\n{no_data_sics}\n",
                    col="y",
                )
                if return_nones:
                    cprint(string="These SICs will be returned in the dataset as None's.", col="y")
                else:
                    cprint(
                        string="These SICs will not appear in the returned datasets.\n"
                        "If necessary adapt the provided 'split_dict' accordingly & re-run the script.",
                        col="y",
                    )
            # Remove those SICs from split_dict
            if not return_nones:
                for sic in no_data_sics:
                    for subset in subsets:
                        if sic in split_dict[subset]:
                            split_dict[subset].remove(sic)

        # Attribute SICs to the respective subsets
        train_sics, val_sics, test_sics = split_dict.values()
        n_train, n_vali, n_test = len(train_sics), len(val_sics), len(test_sics)

    # Else split according to given split_proportion:
    elif target_scale == "softmax":
        # Check whether bins need to be corrected for another variable (primarily 'age')
        correct_for = kwargs.pop("correct_for", None)

        # Create data split
        split_dict = datasplit_for_classification(
            target=target, follow_up=follow_up, study_table=study_tab, correct_for=correct_for
        )

        # Attribute SICs to the respective subsets
        train_sics, val_sics, test_sics = split_dict.values()
        n_train, n_vali, n_test = len(train_sics), len(val_sics), len(test_sics)

        # Update target_sets:
        for sic in list(targ_sets.keys()):
            if sic not in (train_sics + val_sics + test_sics):
                del targ_sets[sic]

    else:
        # For regression
        n_train, n_vali, n_test = split_set_sizes(list_of_sics, split_proportion or [0.8, 0.1, 0.1])

        # # Randomly assign to Train, Val, Test set
        np.random.shuffle(list_of_sics)

        train_sics = list_of_sics[:n_train]
        val_sics = list_of_sics[n_train : n_train + n_vali]
        # Check overlap: set(val_sics) & set(train_sics)
        test_sics = list_of_sics[-n_test:]
        # Check: set(test_sics) & set(val_sics) | set(train_sics) & set(test_sics)  # noqa: ERA001

    # # Brain region specific
    if region is not None:
        # could be extended to atlas labels (sub-regions) given as list
        # Remove all but the region mask from MRI
        mri_dataset = stamp_region(dataset=mri_dataset, region_mask=create_region_mask(region=region, prune=prune))

    # # data augmentation in the training set
    if augment:
        sample2augment, t_sample2augment = sample_for_augmentation(
            targets={sic: targ_sets[sic] for sic in train_sics}, n_augment=n_augment, how=augment
        )

        # Find target bias (augmented training data is included)
        target_bias = np.nanmean(np.append(t_sample2augment, np.fromiter(targ_sets.values(), float)))

    else:
        sample2augment = None
        target_bias = np.nanmean(np.fromiter(targ_sets.values(), float))
        transform_types = None  # not really necessary but for clarity in DataSet

    if return_nones and split_dict is not None:
        for sic in all_sics:
            if sic not in mri_dataset:  # == sic not in targ_sets.keys()
                if sic in targ_sets:
                    msg = "KEY SHOULD BE SAME FOR BOTH SETS"  # TODO: FOR TESTING  # noqa: FIX002
                    raise ValueError(msg)
                mri_dataset[sic] = None
                targ_sets[sic] = None

    # # Prepare Datasets
    if n_train != 0:
        train = DataSet(
            name="Training",
            target_name=target,
            # Normalize MIR data if mri_scale is given
            mri={key: norm_imageset(mri_dataset[key], mri_scale) for key in train_sics},
            target={key: targ_sets[key] for key in train_sics},
            target_min_max=tminmax,
            target_bias=target_bias,
            target_encoding=target_scale,
            follow_up=follow_up,
            sics2augment=sample2augment,
            transform_types=transform_types,
            bg_noise=bg_noise,
        )
    else:
        train = None
        cprint(string="Note: There is no data allocated to the training set!", col="r")

    if n_vali != 0:
        validation = DataSet(
            name="Validation",
            target_name=target,
            mri={key: norm_imageset(mri_dataset[key], mri_scale) for key in val_sics},
            target={key: targ_sets[key] for key in val_sics},
            target_min_max=tminmax,
            target_bias=target_bias,
            target_encoding=target_scale,
            follow_up=follow_up,
        )
    else:
        validation = None
        cprint(string="Note: There is no data allocated to the validation set!", col="r")

    if n_test != 0:
        test = DataSet(
            name="Test",
            target_name=target,
            mri={key: norm_imageset(mri_dataset[key], mri_scale) for key in test_sics},
            target={key: targ_sets[key] for key in test_sics},
            target_min_max=tminmax,
            target_bias=target_bias,
            target_encoding=target_scale,
            follow_up=follow_up,
        )
    else:
        test = None
        cprint(string="Note: There is no data allocated to the test set!", col="r")

    return {"train": train, "validation": validation, "test": test}


def plot_set_distributions(
    life_data=None,  # noqa: ANN001
    data_split: dict | None = None,
    follow_up: bool | None = None,
    target: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot the (target) distributions of all sets (training, validation, test).

    Plot can be created either via the life_data set or its datasplit.

    :param life_data: LIFE data
    :param data_split: split of the data
    :param follow_up: whether to use LIFE follow-up data (must be given if life_data is None)
    :param target: target variable
    :param save_path: the path for saving plots
    :return: None
    """
    import matplotlib  # noqa: ICN001

    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not ((life_data is not None) or (data_split is not None)):
        msg = "Either 'life_data' OR 'data_split' must be given!"
        raise AssertionError(msg)
    if life_data is None:
        if target is None:
            msg = "Target must be given if 'life_data' is not provided!"
            raise AssertionError(msg)
        if follow_up is None:
            msg = "follow_up [bool] must be given if 'life_data' is not provided!"
            raise AssertionError(msg)
    else:
        follow_up = life_data.is_follow_up

    if data_split is None:
        data_split = create_split_dict(life_data)

    all_sics = [sic for sublist in data_split.values() for sic in sublist]
    ls_of_vars = ["sex", "AGE_FU" if follow_up else "AGE_FS", "...", "lesionload", "..."]
    study_tab = load_study_table(exclusion=False, follow_up=follow_up, specify_vars=ls_of_vars)
    study_tab = study_tab.loc[study_tab.SIC_FS.isin(all_sics)].reset_index(drop=True)

    # Find for target the corresponding column in the study table and adjust the column-name
    if np.any([target in var.lower() for var in ls_of_vars]):
        table_target = ls_of_vars[np.where([target in var.lower() for var in ls_of_vars])[0][0]]
        study_tab = study_tab.rename(columns={table_target: target})  # if target != table_target

    data = study_tab[target].to_list()
    labels = list(data_split.keys())

    target_bias = study_tab[target].mean()  # global mean
    target_median = study_tab[target].median()  # global median
    n_unique = len(np.unique(data))
    n_unique = np.clip(a=n_unique, a_min=1, a_max=100)  # reduce the number of total bins for histogram
    n_total = len(data)

    # Sep data per split
    split_data = [study_tab.loc[study_tab.SIC_FS.isin(data_split[subset])][target].to_list() for subset in labels]

    # plot distributions of sets
    plt.figure("Distribution of subsets", figsize=(16, 4))
    ax = plt.subplot(1, 3, 1)
    pos = [1, 2, 3]
    for idx, dat in enumerate(split_data):
        plt.violinplot(dat, [pos[idx]], vert=False, showmeans=True, showmedians=True)
        plt.text(x=max(dat) + 1, y=pos[idx], s=f"N={len(dat)}", fontdict={"size": 8})
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(right=ax.get_xlim()[1] + ax.get_xlim()[1] / 10)  # adds 10% xlim for "N=..." text
    ax.text(
        x=ax.get_xlim()[1] - 1,
        y=pos[-1],
        s=f"$N_{{total}}$={n_total}",
        bbox={"boxstyle": "round", "alpha": 0.4, "facecolor": "white"},
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    plt.subplot(1, 3, 2)
    for idx, dat in enumerate(split_data):
        sns.distplot(dat, hist=False, label=labels[idx])

    plt.subplot(1, 3, 3)
    alph = [1.0, 1.0, 0.7]
    for idx, dat in enumerate(split_data):
        plt.hist(dat, bins=int(np.round(n_unique / 2)), label=labels[idx], alpha=alph[idx])
    plt.vlines(target_bias, ymin=0, ymax=max(np.bincount(np.concatenate(split_data).astype(np.int64))), label="mean")
    plt.vlines(
        target_median,
        ymin=0,
        ymax=max(np.bincount(np.concatenate(split_data).astype(np.int64))),
        linestyles=":",
        label="median",
    )

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(fname=save_path)
        plt.close()
    else:
        plt.show()


def sample_for_augmentation(targets, n_augment, how):
    """Sample for augmentation."""
    if how.lower() not in {"minority", "uniform"}:
        msg = f"how='{how}' is not implemented nor known."
        raise ValueError(msg)
    how = how.lower()

    if n_augment is None:
        n_augment = int(len(targets) / 1)  # 1: full set size; 2: half the set size

    def get_augment_prob(_t, kde: bool = False, normed: bool = True):  # noqa: ANN202, ANN001
        """Get probability to augment."""
        uniq_t, n_per_t_bin = np.unique(_t, return_counts=True)
        t_grid = uniq_t

        if not kde:
            ed_targets = n_per_t_bin / sum(n_per_t_bin)  # empirical distribution
            # Define probability to augment specific data-point w.r.t. target
            aug_prob = (1 - ed_targets) / sum(1 - ed_targets)  # ~ inverse of target distribution

        # Alternatively: Gaussian Kernel Density fit to target distribution
        # This is smoothed in comparison to empirical distribution
        else:  # if kde:
            from scipy import stats

            kde_gausfit = stats.gaussian_kde(_t, bw_method="scott")
            # kde_gausfit.set_bandwidth(kde_gausfit.factor / 3.)  # scott/3  # noqa: ERA001
            kde_gausfit_eval = kde_gausfit.evaluate(t_grid)
            aug_prob = (1 - kde_gausfit_eval) / sum(1 - kde_gausfit_eval)

        if normed:  # [0, 1]
            aug_prob = normalize(aug_prob, lower_bound=0, upper_bound=1)
            # These will not add up to 1: sum(aug_prob) > 1

        return uniq_t, aug_prob

    unique_t = get_augment_prob(_t=np.array(list(targets.values())), kde=False)[0]
    # Take/estimate target distribution with: [0, 1]

    # Draw n-samples from dataset based on augment_prob
    sample2augment = []
    t_sample2augment = []

    while len(sample2augment) < n_augment:
        if how == "minority":  # augment minority distribution
            p_augment = get_augment_prob(_t=list(targets.values()) + t_sample2augment)[1]

        else:  # how.lower() == "uniform":
            p_augment = np.ones(len(unique_t))  # / len(unique_t)

        if np.all(np.isnan(p_augment)):
            # This avoids all p_augment being nan, due to absolute uniform target bins
            p_augment = np.random.uniform(size=len(p_augment))

        rand_t = np.random.choice(unique_t)
        p_aug = p_augment[np.where(unique_t == rand_t)][0]

        take_it = np.random.choice(a=[False, True], size=1, p=[1 - p_aug, p_aug])[0]
        if take_it:
            temp_target_key_list = []  # Gather keys of given target
            for key, t in targets.items():
                if t == rand_t:
                    temp_target_key_list.append(key)

            sample2augment.append(np.random.choice(temp_target_key_list))  # take one random SIC from the list
            t_sample2augment.append(targets[sample2augment[-1]])

    return sample2augment, t_sample2augment


@function_timed
def augment_mriset(samples, mris, targets, transform_types):
    """
    Augment given MRI set with various transformations via random_transform_mri().

    :param samples: list of SICs whose data is to be augmented
    :param mris: MRI set which is to be augmented
    :param targets: target set which is to be augmented
    :param transform_types: which types of transformation shall be applied (list, OR None)
    :return: updated MRI and target sets
    """
    # Defines how many transforms are applied on an MRI (1 or 2), for details see random_transform_mri()
    ls_n_trans = list(np.random.randint(1, 2 + 1, len(samples)))
    ls_manis = [transform_types] * len(samples)
    # ls_manis is then either [None, None, ...] OR [["type1", "type2"], ["type1", "type2"], ...]

    # Sub-samples the MRI-set for parallel processing
    # Threading (here superior over pooling, but only marginally for full dataset)
    # with concurrent.futures.ProcessPoolExecutor(100) as executor:
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        aug_mris = executor.map(
            random_transform_mri,  # fct
            [(ky, mris[ky]) for ky in samples],  # arg1: mri
            ls_manis,  # arg2: manipulation
            ls_n_trans,
        )  # arg3: n_manips

    aug_mris = dict(tuple(aug_mris))
    mris.update(aug_mris)  # merge datasets

    # Augment target set accordingly:
    for augkey in aug_mris:
        tkey = augkey.split("_")[0]  # only original SIC
        targets.update({augkey: targets[tkey]})

    return mris, targets


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
