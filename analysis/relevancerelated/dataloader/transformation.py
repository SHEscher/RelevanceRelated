"""
Functions to transform MRIs.

Author: Simon M. Hofmann | 2018-2020
"""

# %% Import
from __future__ import annotations

import ast
import concurrent.futures
import datetime
import os
import string
from pathlib import Path
from shutil import copyfile

import ants
import nibabel as nib
import numpy as np
from nilearn import masking
from scipy import ndimage

from relevancerelated.dataloader.prune_image import find_edges
from relevancerelated.utils import chop_microseconds, cprint

# %% Set global paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

P2DATA = Path(".../Data/mri/")  # anonymized
FN_MASK = "T1_brain_mask.nii.gz"  # filename of T1-brain mask
P2MNI = ".../LIFE/preprocessed/{sic}/structural/"  # anonymized

# %% Global image orientation (according to nibabel) < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Note: We use FreeSurfer output as reference space, according to nibabel: ('L', 'I', 'A'),
# whereas nibabel (canonical) standard is: ('R', 'A', 'S') [RAS+]
# For more, see: https://nipy.org/nibabel/coordinate_systems.html
global_orientation_space = "LIA"  # Note for ANTsPy this is vice versa 'RSP'

# Set global variable
all_manip_opt = ["rotation", "translation", "noise", "none"]  # all implemented options for manipulation
# all_manip_opt = ['rotation', 'translation', 'noise', 'iloss', 'contrast', 'none'] # planed # noqa: ERA001
# 'flip': biologically implausible


# %% NiBabel based re-orientation functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_orientation_transform(affine: np.ndarray, reference_space: str = global_orientation_space) -> np.ndarray:
    """Get orientation transform."""
    return nib.orientations.ornt_transform(
        start_ornt=nib.orientations.io_orientation(affine), end_ornt=nib.orientations.axcodes2ornt(reference_space)
    )


def file_to_ref_orientation(
    image_file: nib.Nifti1Image, reference_space: str = global_orientation_space
) -> nib.Nifti1Image:
    """Take a Nibabel NifTi-file (not array) and returns a reoriented version (to global ref space."""
    ornt_trans = get_orientation_transform(affine=image_file.affine, reference_space=reference_space)
    return image_file.as_reoriented(ornt_trans)


# %% ANTspy based warping function << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><


def save_ants_warpers(tx: dict, folder_path: str | Path, image_name: str) -> None:
    """Save an ANTs warper file."""
    if not ("fwdtransforms" in list(tx.keys()) and "invtransforms" in list(tx.keys())):
        msg = "tx object misses forward and/or inverse transformation files."
        raise AssertionError(msg)

    # # Set paths
    # for forward warper
    save_path_name_fwd = Path(folder_path, f"{image_name}1Warp.nii.gz")
    # for inverse warper
    save_path_name_inv = Path(folder_path, f"{image_name}1InverseWarp.nii.gz")
    # # Save also linear transformation .mat file
    save_path_name_mat = Path(folder_path, f"{image_name}0GenericAffine.mat")

    # # Copy warper files from temporary tx folder file to new location
    copyfile(tx["fwdtransforms"][0], save_path_name_fwd)
    copyfile(tx["invtransforms"][1], save_path_name_inv)
    copyfile(tx["invtransforms"][0], save_path_name_mat)  # == ['fwdtransforms'][1]


def get_list_ants_warper(folderpath: str | Path, inverse: bool = False, only_linear: bool = False) -> list | None:
    """Get the list of ANTs warpers."""
    warp_fn = "1Warp.nii.gz" if not inverse else "1InverseWarp.nii.gz"
    lin_mat_fn = "0GenericAffine.mat"

    warp_found = False  # init
    mat_found = False  # init

    # Search for transformation files
    for file in Path(folderpath).iterdir():
        # Look for non-linear warper files
        if warp_fn in file:
            warp_fn = str(Path(folderpath, file))
            if not warp_found:
                warp_found = True
            else:
                msg = f"Too many files of type '*{warp_fn}' exist in folderpath '{folderpath}'."
                raise FileExistsError(msg)

        # Look for linear transformation (affine) *.mat
        if lin_mat_fn in file:
            lin_mat_fn = str(Path(folderpath, file))
            if not mat_found:
                mat_found = True
            else:
                msg = f"Too many files of type '*{lin_mat_fn}' exist in folderpath '{folderpath}'."
                raise FileExistsError(msg)

    if mat_found and warp_found:
        transformlist = [lin_mat_fn, warp_fn] if inverse else [warp_fn, lin_mat_fn]

    elif only_linear and mat_found:
        transformlist = [lin_mat_fn]

    else:
        transformlist = None
        cprint(string="Not all necessary transformation files were found in given path.", col="r")

    return transformlist


# %% Masking functions << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >


def create_mask(mri: nib.Nifti1Image) -> nib.Nifti1Image:
    """Create a brain mask (1:=brain; 0:=background) from a skull-stripped image."""
    return masking.compute_background_mask(mri)


def create_t1_brainmask(sic: str, follow_up: bool) -> None:
    """
    Create a T1-brain mask for the given SIC.

    :param sic: SIC
    :param follow_up: whether to load LIFE follow-up data
    :return: None
    """
    # Link to the corresponding brain-mask in F. Liem's folder OR create brain mask one
    dir_mask = Path(P2DATA, sic, "followup" if follow_up else "baseline", FN_MASK)
    # in SIC parent folder (not in sequence-specific dir)
    if not dir_mask.is_file():
        if Path(P2MNI.format(sic=sic), FN_MASK).is_file() and not follow_up:
            os.symlink(Path(P2MNI.format(sic=sic), FN_MASK), dir_mask)

        else:  # Create mask
            from relevancerelated.dataloader.LIFE.LIFE import load_sic_mri

            sic, t1_mask = load_sic_mri(
                _sic=sic,
                mri_sequence="T1",
                follow_up=follow_up,
                bm=True,
                norm=None,
                regis=False,
                dtype=np.float16,
                as_numpy=False,
                raiserr=False,
            )

            t1_mask = create_mask(mri=t1_mask)

            # Save mask as nii.gz
            nib.Nifti1Image(dataobj=t1_mask.get_fdata().astype(np.uint8), affine=t1_mask.affine).to_filename(dir_mask)


def create_raw_brainmask(sic: str, mri_sequence: str, follow_up: bool) -> None:
    """
    Create a brain mask for a raw MRI in the given MRI sequence.

    :param sic: SIC
    :param mri_sequence: MRI sequence
    :param follow_up: whether to load LIFE follow-up data
    :return: None
    """
    from .LIFE.LIFE import load_sic_raw_mri

    cprint(f"Create raw brainmask for {mri_sequence.upper()} of {sic} ...\n")

    p2reg = P2DATA / sic / ("followup" if follow_up else "baseline") / mri_sequence.lower()
    brain_mask = get_t1_brainmask(sic=sic, follow_up=follow_up)
    raw_move = load_sic_raw_mri(
        _sic=sic, mri_sequence=mri_sequence, follow_up=follow_up, brain_masked=False, reorient=False, path_only=False
    )

    transformlist = get_list_ants_warper(folderpath=p2reg, only_linear=True)
    brain_mask_native = ants.apply_transforms(
        fixed=ants.from_nibabel(raw_move),
        moving=ants.from_nibabel(brain_mask),
        transformlist=transformlist,
        whichtoinvert=[1],
        verbose=False,
    ).astype("uint8")

    # Save mask
    brain_mask_native.to_file(filename=f"{p2reg}raw/{FN_MASK.replace('T1', mri_sequence.upper())}")


def get_t1_brainmask(sic: str, follow_up: bool) -> nib.Nifti1Image:
    """
    Get brain mask (1:=brain; 0:=background) of given SIC in T1-FreeSurfer Space.

    :param sic: SIC
    :param follow_up: whether to load LIFE follow-up data
    :return: T1 brain-mask
    """
    # This creates/links brain masks only if not available yet:
    create_t1_brainmask(sic=sic, follow_up=follow_up)
    dir_mask = Path(P2DATA, sic, "followup" if follow_up else "baseline", FN_MASK)
    # in SIC parent folder (not in sequence-specific dir)
    return nib.load(filename=dir_mask)


def get_raw_brainmask(sic: str, mri_sequence: str, follow_up: bool) -> nib.Nifti1Image:
    """
    Get brain mask (1:=brain; 0:=background) of given SIC in raw/native space of the given sequence.

    :param sic: SIC
    :param mri_sequence: an MRI sequence (T1, FLAIR, SWI)
    :param follow_up: True: get brain mask of LIFE follow-up data
    :return: brain mask of given SIC
    """
    p2reg = f"{P2DATA}{sic}/{'followup' if follow_up else 'baseline'}/{mri_sequence.lower()}/"
    brain_mask_fname = Path(f"{p2reg}raw/{FN_MASK.replace('T1', mri_sequence.upper())}")

    if not brain_mask_fname.is_file():
        from relevancerelated.dataloader.mri_registration import register_native_to_t1_space

        if not register_native_to_t1_space(
            sic=sic, mri_sequence=mri_sequence, follow_up=follow_up, save_move=True, verbose=True
        ):
            create_raw_brainmask(sic=sic, mri_sequence=mri_sequence, follow_up=follow_up)

    if not brain_mask_fname.is_file():
        cprint(string=f"No brain mask for native {mri_sequence.upper()} of {sic} could be found nor created!", col="y")
        return None

    return nib.load(filename=brain_mask_fname)


# %% Transformations << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><


def clip_img(mri: np.ndarray, high_clip: float | None = None) -> np.ndarray:
    """
    Clip background and tiny deviations.

    Clip to the corresponding (global) background value (usually: 0)
    For image data in float scale, (0,1) OR (-1,1), (scipy) spline-interpolation (specifically for
    rotation) is not clean, i.e. it pushes the intensity for some images beyond the initial image range.
    For rotation, we use also 'high_clip' to clip the max of the rotated image to the initial image-max.

    :param mri: MRI image
    :param high_clip: clip image to given value
    :return: MRI with the clipped background
    """
    if mri.min() < -1.0:
        msg = "Image scale should be either (0, 255), (0, 1), OR (-1, 1)."
        raise ValueError(msg)

    # # Clip tiny deviations from background to background (can also become negative)
    # For scale (0, 255) no clipping necessary
    if (
        mri.max() <= 1.1  # noqa: PLR2004
    ):  # due to scipy.rotate()-spline-interpolation img.max can become bigger than 1.
        if mri.min() < -0.9:  # noqa: PLR2004
            # For image range (-1, 1)
            bg = -1.0
            mri[mri < (-1 + 1 / 256)] = bg  # clip
        else:
            # For image range (0, 1)
            bg = 0.0
            mri[mri < 1 / 256] = bg  # clip

        # Clip high
        high_clip = 1.0 if high_clip is None else high_clip

    else:
        high_clip = 255 if high_clip is None else high_clip

    mri[mri > high_clip] = high_clip  # ==np.clip(a=mri, a_min=bg, a_max=high_clip)

    return mri


def rotate_mri(mri: np.ndarray, axes: tuple[int], degree: float) -> np.ndarray:
    """
    Rotate given MRI.

    :param mri: 3D MRI
    :param axes: either (0,1), (0,2), or (1,2)
    :param degree: -/+ ]0, 360[
    :return: rotated MRI
    """
    return clip_img(
        mri=ndimage.interpolation.rotate(
            input=mri,
            angle=degree,
            axes=axes,
            reshape=False,
            mode="constant",  # (default: constant)
            order=3,  # spline interpolation (default: 3)
            cval=mri.min(),
        ),  # cval = background voxel value
        high_clip=mri.max(),
    )


def max_axis_translation(mri: np.ndarray, axis: int) -> tuple[int, int]:
    """
    Find the max possible translation along given axis, where the brain is not cut off at the border.

    :param mri: given MRI
    :param axis: translation Axis
    :return: tuple of absolute (!) max shift-sizes in both directions (-, +)
    """
    edges = find_edges(mri)
    len_ax = mri.shape[axis]
    return edges[0 + axis * 2], len_ax - edges[1 + axis * 2] - 1


def translate_mri(mri: np.ndarray, axis: int, by_n: int) -> np.ndarray:
    """
    Translate given MRI along given axis by n steps.

    :param mri: given MRI
    :param axis: 0, 1, or 2 (3d)
    :param by_n: shift by n steps, the sign indicates the direction of the shift
    :return: translated MRI
    """
    edges = find_edges(mri)
    ledg = edges[0 + axis * 2]
    hedg = edges[1 + axis * 2]

    # If translation too big, make shift size smaller
    if ((hedg + by_n) >= mri.shape[axis]) or ((ledg + by_n) < 0):
        max_shift = max_axis_translation(mri, axis)[0 if np.sign(by_n) < 0 else 1]
        print(f"Max shift-size is {max_shift * np.sign(by_n)}. Given 'by_n' is adapted accordingly.")
        by_n = max_shift * np.sign(by_n)
    # Alternatively, implement: raise Error

    shift_axis = [slice(None)] * 3  # init
    new_shift_axis = [slice(None)] * 3  # init
    shift_axis[axis] = slice(ledg, hedg + 1)
    new_shift_axis[axis] = slice(ledg + by_n, hedg + by_n + 1)

    # Shift/translate brain
    trans_mri = np.zeros(mri.shape, mri.dtype)  # Create empty cube to store shifted brain
    trans_mri[:] = mri.min()  # for non-zero backgroun (e.g., after normalization)
    old_pos = tuple(shift_axis)
    new_pos = tuple(new_shift_axis)
    trans_mri[new_pos] = mri[old_pos]

    return trans_mri


def noisy_mri(mri: np.ndarray, noise_type: str, noise_rate: float | None = None) -> np.ndarray:
    """
    Get a noisy MRI.

    See also: http://scipy-lectures.org/advanced/image_processing/#image-filtering
    """
    noise_types = ["random_knockout", "random_swap", "local_disturb", "image_blur"]
    noise_type = noise_type.lower()
    if noise_type not in noise_types:
        msg = f"'noise_type' must be in {noise_types}"
        raise ValueError(msg)

    n_all_vox = len(mri.flatten())  # number of all voxels
    bg = mri.min()  # image-value of the background (assumption that for all given images bg == img.min)

    if noise_rate is None:
        noise_rate = dict(zip(noise_types, [0.01, 0.01, 0.1, 0.5], strict=False))[
            noise_type
        ]  # defaults (order important!)
        # print(f"Set 'noise_rate' to default value: {noise_rate}")  # noqa: ERA001

    assert 0.0 <= noise_rate <= 1.0, f"'noise_rate' for  {noise_type} must be between [0.-1.]!"  # noqa: S101
    n_noise = round(noise_rate * n_all_vox)  # number of voxels which will be perturbed

    # # Knock out 1% of all (non-background) voxels (information loss)
    if noise_type == "random_knockout":
        # Find indices of non-background voxels
        xs, ys, zs = (mri + abs(bg)).nonzero()  # abs(...) necessary for re-normed data, e.g., between -1,1
        # Choose random voxels which are manipulated
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        # Apply knockout
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = bg  # could be any number (e.g., via arg)

    # # Swap two random non-background voxels
    # (partial information loss, intensity distribution. remains same, global noise addition)
    if noise_type == "random_swap":
        xs, ys, zs = (mri + abs(bg)).nonzero()
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        # Copy indices and shuffle them for swap
        noise_idx2 = noise_idx.copy()
        np.random.shuffle(noise_idx)
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = mri[xs[noise_idx2], ys[noise_idx2], zs[noise_idx2]]

    # # Disturb pixel values (local noise addition)
    if noise_type == "local_disturb":
        # Create a blured copy of given MRI:
        mri_med_filter = ndimage.median_filter(mri, 3)  # we sample from this
        xs, ys, zs = (mri + abs(bg)).nonzero()
        # Swap random voxels with blured MRI, i.e., on-spot distortion
        noise_idx = np.random.choice(a=range(len(xs)), size=n_noise, replace=False)
        mri[xs[noise_idx], ys[noise_idx], zs[noise_idx]] = mri_med_filter[xs[noise_idx], ys[noise_idx], zs[noise_idx]]

    # # Blur whole image
    if noise_type == "image_blur":
        # Create a blured copy of given MRI
        blur_mri = ndimage.median_filter(mri, size=2)  # 1: now filter; 3: too strong
        # Mix blured version in original
        mri = (mri * (1.0 - noise_rate) + blur_mri * noise_rate).astype(mri.dtype)  # keep datatype

    return mri


def add_background_noise(mri, noise_scalar=0.015):
    """
    Add noise drawn from absolute normal distribution abs((0, sd)).

    :param mri: MRI
    :param noise_scalar: after some testing: recommended between [.008, .02[
    :return: MRI with a noisy background
    """
    # # Check mri-format: (sic, brain) OR brain
    tp = isinstance(mri, tuple)  # (sic, brain)
    sic = mri[0] if tp else None
    mri = mri[1] if tp else mri

    # # Prepare constants & variables
    img_max = 255 if mri.max() > 1.0 else 1.0  # should be either in range (0, 255), (0, 1) OR (-1, 1)

    # Get number of background voxels
    bg = mri.min()  # background
    n_bg_vox = len(mri[mri == bg])

    # Define image data type ('keep it small')
    img_dtype = np.uint8 if img_max == 255 else np.float16  # noqa: PLR2004

    # Define scalar for normalal distribution
    sl = img_max * noise_scalar

    # # Add noise to the background
    # Noise is drawn from half-normal distribution, ie., abs(norm-distr) == scipy.stats.halfnorm.rvs()
    noise = abs(np.random.RandomState().normal(loc=0, scale=sl, size=n_bg_vox))
    mri[mri == bg] = bg + noise

    if tp:
        return sic, clip_img(mri.astype(img_dtype))
    return clip_img(mri.astype(img_dtype))


def add_bg_noise_mriset(mriset: dict, noise_scalar: float = 0.015) -> dict:
    """Add noise to the background."""
    if not isinstance(mriset, dict):
        msg = "'mriset' is expected to be dict in the from {sic: brain}"
        raise TypeError(msg)

    cprint(string="\nAdd noise to background on whole dataset...", col="b")
    start_time_load = datetime.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        mriset = executor.map(add_background_noise, mriset.items(), [noise_scalar] * len(mriset))

    mriset = dict(tuple(mriset))

    print(
        f"Duration of adding background noise to all images of dataset (via threading) "
        f"{chop_microseconds(datetime.datetime.now() - start_time_load)} [h:m:s]"
    )

    return mriset


# %% Apply transformations << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><<


def random_transform_mri(
    mri: np.ndarray, manipulation: str | list[str] | None = None, n_manips: int = 1
) -> tuple[str, np.ndarray] | np.ndarray:
    """
    Transform given MRI by random or given type of manipulation.

    !!! note "Experimental"
        The function is still in an experimental stage.

    :param mri: a 3D-MRI [tuple](sic, mri), or [array](mri)
    :param manipulation: if specific image manipulation should be applied, indicate which, str OR list
    :param n_manips: number of manipulations
    :return: (sic, transformed mri) [tuple] or only transformed mri (depends on input)
    """
    tp = isinstance(mri, tuple)
    sic = mri[0] if tp else None  # tp is True -> mri: (sic, brain), else -> mri: brain
    mri = mri[1] if tp else mri

    # # Create suffix for transformed SIC data: SIC_{mkey}
    mkey = (
        np.random.RandomState().choice(list(string.ascii_letters))
        + str(np.random.RandomState().randint(0, 10))
        + np.random.RandomState().choice(list(string.ascii_letters))
    )
    # RandomState necessary for parallelization of concurrent.futures, otherwise same value
    # manipulation_suffix
    # (Note: A negligible small probability remains that suffix is the same for two transformed MRIs)

    # # Choose the type of image manipulation (in those which are implemented so far)
    if manipulation is not None:
        manipulation = manipulation if isinstance(manipulation, (list, np.ndarray)) else [manipulation]
        manipulation = [manip.lower() for manip in manipulation]
        if not all(manip in all_manip_opt for manip in manipulation):
            msg = f"'manipulation' must be None or subset of {all_manip_opt}."
            raise AssertionError(msg)
    else:
        manipulation = all_manip_opt

    # Check for the number of manipulations to be applied
    n_manips = np.clip(a=n_manips, a_min=1, a_max=len(manipulation))
    # at least one, and maximally each manipulation shall be applied just once

    # Randomly pick image manipulation (Note: Each manipulation is applied just once, 'replace=False')
    manips = np.random.RandomState().choice(a=manipulation, size=n_manips, replace=False)

    for manip in manips:
        # # Rotation
        if manip == "rotation":
            # e.g., -40,40 degrees (Cole et al., 2017; Jonsson et al., 2019)
            # However, (after testing) seems too strong. Limit angle depending on rotation axis

            mkey = "rot" + mkey

            n_rot = np.random.RandomState().choice([1, 2, 3])  # number of rotations: 1, 2, or on all axes
            # With sequential rotations, we can rotate the brain in all directions
            # choose rnd axes (don ott rotate on the same axis: replace=False)

            # Define angle-range for each axis
            axes_angles_dict = {"(0,1)": (-15, 15), "(0,2)": (10, 10), "(1,2)": (-35, 35)}

            # Choose random axis/axes
            _axes = np.random.RandomState().choice(np.array(list(axes_angles_dict.keys())), n_rot, replace=False)

            # for ax in _axes:
            for ax in _axes:
                # Choose random angle and apply rotation
                while True:
                    rot_angle = np.random.RandomState().randint(
                        low=axes_angles_dict[ax][0], high=axes_angles_dict[ax][1] + 1
                    )
                    if rot_angle != 0:
                        break

                mri = rotate_mri(mri=mri, degree=rot_angle, axes=ast.literal_eval(ax))

        # # Translation
        if manip == "translation":
            # e.g., -10,10 voxels (Cole et al., 2017; Jonsson et al., 2019)
            # However, there seems to be no good reason not to translate the brain to the image borders.

            mkey = "tran" + mkey

            # Translate on 1, 2 or all 3 axes
            n_ax = np.random.RandomState().randint(1, 3 + 1)
            _axes = np.random.RandomState().choice(a=[0, 1, 2], size=n_ax, replace=False)

            for ax in _axes:
                direction = np.random.RandomState().choice([-1, 1])  # random direction of shift

                max_shift = max_axis_translation(mri=mri, axis=ax)[0 if direction < 0 else 1]

                move = np.random.RandomState().randint(1, np.maximum(2, max_shift))
                # or randint(1, 10+1) as e.g. Cole et al. (2017)
                move *= direction

                # Shift/translate brain
                mri = translate_mri(mri=mri, axis=ax, by_n=move)

        # # Add noise
        if manip == "noise":
            mkey = "noi" + mkey
            # revisit default noise_type here: 'image_blur'
            mri = noisy_mri(mri=mri, noise_type="image_blur", noise_rate=None)

        # Could add:
        #  - Information loss (e.g., whole areas or bigger slices)
        #  - intensity shift/contrast change: since neg. corr between age and max&mean MRI-intensity,
        #       see(z_playground: exp_mri), i.e., intensity distributions is a function of age.

        # No manipulation
        if manip == "none":
            # create an unchanged copy
            pass

    if tp:
        # Save manipulated mri in (training-)set with key: sic + manipulation_suffix
        aug_key = f"{sic}_{mkey}"
        return aug_key, mri  # augmented MRI

    return mri  # augmented MRI


# >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
