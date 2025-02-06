"""
Register raw non-T1 MRI and create corresponding brain masks.

Author: Simon M. Hofmann | 2021
"""

# %% Import
from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import ants
import nibabel as nb
import numpy as np

from relevancerelated.configs import params
from relevancerelated.dataloader.transformation import get_list_ants_warper, get_t1_brainmask, save_ants_warpers
from relevancerelated.utils import chop_microseconds, cprint

# %% Set paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

P2RAW = ".../patients/"
P2DATA = Path(".../Data/mri/")
P2MNI = ".../LIFE/preprocessed/{sic}/structural/"
FN_MASK = "T1_brain_mask.nii.gz"  # filename of T1-brain mas
# %% Registration functions  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def register_native_to_t1_space(
    sic: str, mri_sequence: str, follow_up: bool, save_move: bool = True, verbose: bool = False
) -> None:
    """
    Register MRI from individual, native space to T1 (FreeSurfer) space.

    :param sic: SIC
    :param mri_sequence: MRI sequence (T1, FLAIR, SWI)
    :param follow_up: whether to use LIFE follow-up data
    :param save_move: save transformation field
    :param verbose: verbose or not
    :return: None
    """
    if mri_sequence.lower() not in params.mri_sequences:
        msg = f"{mri_sequence} is not known OR implemented yet."
        raise ValueError(msg)

    # Check whether already done:
    p2reg = P2DATA / sic / ("followup" if follow_up else "baseline") / mri_sequence.lower()

    if p2reg.is_dir() and len(os.listdir(p2reg)) >= 2:  # noqa: PLR2004
        if verbose:
            print(f"{mri_sequence.upper()} of '{sic}' was registered to T1 (FreeSurfer space) already.")
        return

    # Register image
    from relevancerelated.dataloader.LIFE.LIFE import load_sic_mri, load_sic_raw_mri

    # Get the path to the raw MRI of the SIC
    p2r = load_sic_raw_mri(
        _sic=sic, mri_sequence=mri_sequence, brain_masked=False, follow_up=follow_up, reorient=False, path_only=True
    )

    if p2r is None:
        if verbose:
            print(f"There is no native/raw {mri_sequence.title()} of '{sic}'.")
        return

    # Create dir
    p2reg.mkdir(exist_ok=True)

    # Create Symlink to the raw file
    raw_dir = p2reg / "raw"
    raw_dir.mkdir(exist_ok=True, parents=True)

    with contextlib.suppress(FileExistsError):
        os.symlink(p2r, raw_dir / p2r.name)

    # # Get MRI data (move, fixed)
    # MRI data to be registered
    raw_move = load_sic_raw_mri(
        _sic=sic, mri_sequence=mri_sequence, brain_masked=False, follow_up=follow_up, reorient=False, path_only=False
    )

    # Reference MRI (T1) in FreeSurfer space
    sic, t1_mri = load_sic_mri(
        _sic=sic,
        mri_sequence="T1",
        follow_up=follow_up,
        bm=False,
        norm=None,
        regis=False,
        raw=False,
        dtype=np.float32,
        as_numpy=False,
        raiserr=False,
    )

    if t1_mri is None:
        cprint(
            string=f"For {sic}, it is not possible to register native (raw) '{mri_sequence.upper()}' to "
            f"FreeSurfer-T1-space, since there is no corresponding T1-file!",
            col="r",
        )
        return

    # T1 as NifTi, since ANTs has issues with MGH
    t1_mri = nb.Nifti1Image(t1_mri.get_fdata(), affine=t1_mri.affine)  # before: MGH

    # # Linear register the given MRI sequence to reference T1 image (takes about 30 sec)
    tot1tx = ants.registration(
        fixed=ants.from_nibabel(t1_mri),
        moving=ants.from_nibabel(raw_move),
        type_of_transform="Rigid",  # OR "DenseRigid" (takes ~ 1:10 sec)
        verbose=verbose,
    )

    # # Save linear registration matrix (sequence-native to T1)
    copyfile(
        src=tot1tx["fwdtransforms"][0],
        # Adapt path
        dst=p2reg / (f"raw{mri_sequence.upper()}-2-T1_0G" + tot1tx["fwdtransforms"][0].split("/")[-1].split("0G")[-1]),
    )

    # Save registered MRI file (if requested):
    if save_move and mri_sequence.lower() != "t1":  # for T1 we have the files in FreeSurfer folder
        tot1tx["warpedmovout"].to_file(p2reg / f"{mri_sequence.lower()}_in_t1-space.nii.gz")

    # # Brain mask for native space
    brain_mask = get_t1_brainmask(sic=sic, follow_up=follow_up)

    brain_mask_native = ants.apply_transforms(
        fixed=ants.from_nibabel(raw_move),
        moving=ants.from_nibabel(brain_mask),
        transformlist=tot1tx["invtransforms"],
        whichtoinvert=[1],
        verbose=verbose,
    ).astype("uint8")

    # Save mask
    brain_mask_native.to_file(filename=raw_dir / f"{FN_MASK.replace('T1', mri_sequence.upper())}")

    cprint(string=f"Registration from native {mri_sequence.upper()} to T1 for '{sic}' DONE.", col="y")


def register_t1_to_mni(sic: str, follow_up: bool, verbose: bool = False) -> None:
    """
    Register MRI in T1 (FreeSurfer) space to MNI.

    :param sic: SIC
    :param follow_up: whether to load LIFE follow-up data
    :param verbose: verbose or not
    :return: None
    """
    fn_t1_mni = "T1_brain2mni.nii.gz"
    p2_t1_mni = Path(P2MNI.format(sic=sic), fn_t1_mni)
    p2reg = P2DATA / sic / ("followup" if follow_up else "baseline")

    if p2_t1_mni.is_file() and not follow_up:
        # If file already there, then create symlinks
        with contextlib.suppress(FileExistsError):
            os.symlink(p2_t1_mni, p2reg / fn_t1_mni)  # for MNI-file
            os.symlink(
                P2MNI.format(sic=sic) + "transforms2mni/",
                p2reg / "transforms2mni",  # w/o ending '/' !
                target_is_directory=True,
            )  # for warper-file

    else:
        from nilearn.datasets import load_mni152_template

        from relevancerelated.dataloader.LIFE.LIFE import load_sic_mri

        cprint(string=f"For '{sic}' T1 in MNI space & corresponding warper file doesn't exist. Create ...", col="y")

        # Load MNI template (fixed-reference)
        mni_temp = load_mni152_template(resolution=2)  # note that new nilearn versions take a different MNI image
        if mni_temp.shape != (91, 109, 91):
            from relevancerelated.dataloader.LIFE.LIFE import get_mni_template

            msg = "mni_temp must be of shape (91, 109, 91)"
            cprint(string=msg, col="r")
            print("Loading template via get_mni_template() instead ...")  # for new nilearn versions
            mni_temp = get_mni_template(low_res=True, reorient=False, prune=False, original_template=True, as_nii=True)
            if mni_temp.shape != (91, 109, 91):
                raise ValueError(msg)

        # Load T1-MRI brain-masked (in FreeSurfer space): moving
        sic, t1_mri = load_sic_mri(
            _sic=sic, mri_sequence="t1", follow_up=follow_up, bm=True, regis=False, as_numpy=False
        )
        t1_mri = nb.Nifti1Image(dataobj=t1_mri.get_fdata(), affine=t1_mri.affine)  # MGH to nii

        # Apply forward warping of the brain masked Flair in T1 space [about 5.3 sec]
        mnitx = ants.registration(
            fixed=ants.from_nibabel(mni_temp),
            moving=ants.from_nibabel(t1_mri),
            type_of_transform="SyN",
            verbose=verbose,
        )

        # Save files:
        mnitx["warpedmovout"].to_file(filename=p2reg / fn_t1_mni)  # t1 in MNI-space

        (p2reg / "transforms2mni").mkdir(exist_ok=True)

        save_ants_warpers(
            tx=mnitx, folder_path=p2reg / "transforms2mni", image_name="transform"
        )  # naming analog to F.L.

        cprint(string=f"Registration of T1-MRI to MNI for '{sic}' DONE.", col="y")


def register_mri_sequence_in_t1_space_to_mni(
    sic: str, mri_sequence: str, follow_up: bool, verbose: bool = False
) -> None:
    """
    Register MRI of a SIC in t1 (FreeSurfer) space of for a given sequence to MNI space.

    :param sic: SIC
    :param mri_sequence: MRI sequence (T1, FLAIR, SWI)
    :param follow_up: True: take LIFE follow-up data
    :param verbose: verbose or not
    :return: None
    """
    if mri_sequence.lower() not in params.mri_sequences:
        msg = f"{mri_sequence} is not known OR implemented yet."
        raise ValueError(msg)

    fn_t1_mni = "T1_brain2mni.nii.gz"

    p2reg = P2DATA / sic / ("followup" if follow_up else "baseline") / mri_sequence.lower()
    p2_seq_mni = Path(p2reg, fn_t1_mni.replace("T1", mri_sequence.upper()))

    # Check whether already done:
    if p2_seq_mni.is_file():
        if verbose:
            print(f"{mri_sequence.title()} of '{sic}' was registered to MNI-space already.")
        return

    # Register native to T1 space (necessary)
    register_native_to_t1_space(
        sic=sic, mri_sequence=mri_sequence, follow_up=follow_up, save_move=True, verbose=verbose
    )

    # Load brain-masked MRI of a given sequence in T1-space: moving image
    from relevancerelated.dataloader.LIFE.LIFE import load_sic_mri

    sic, mri_move = load_sic_mri(
        _sic=sic,
        mri_sequence=mri_sequence,
        follow_up=follow_up,
        bm=True,
        norm=False,
        raw=False,
        as_numpy=False,
        raiserr=False,
        regis=False,
    )  # (256, 256, 256)

    if mri_move is None:
        cprint(
            string=f"There is no {mri_sequence} MRI in FreeSurfer-T1 space for {sic} that can be registered to MNI.",
            col="r",
        )
        return

    if mri_sequence == "t1":
        mri_move = nb.Nifti1Image(mri_move.get_fdata(), affine=mri_move.affine)

    # Get T1 in MNI space as fixed reference & corresponding warping file:
    if not (P2DATA / sic / "followup" if follow_up else "baseline" / fn_t1_mni).is_file():
        register_t1_to_mni(sic=sic, follow_up=follow_up, verbose=verbose)
    t1_mni = nb.load(filename=P2DATA / sic / ("followup" if follow_up else "baseline") / fn_t1_mni)
    # Following could be done with the MNI template but should be more accurate with this approach

    # Load transformation/warping file:
    mnitx = get_list_ants_warper(
        folderpath=P2DATA / sic / ("followup" if follow_up else "baseline") / "transforms2mni", inverse=False
    )

    # Apply forward warping of a given brain-masked MRI sequence to MNI-space
    sequence_in_mni = ants.apply_transforms(
        fixed=ants.from_nibabel(t1_mni), moving=ants.from_nibabel(mri_move), transformlist=mnitx, verbose=verbose
    )

    # Save file
    sequence_in_mni.to_file(filename=p2reg / f"{fn_t1_mni.replace('T1', mri_sequence.upper())}")

    cprint(string=f"Registration of {mri_sequence.upper()}-MRI to MNI for '{sic}' DONE.", col="y")


def register_mris(sic_list: list | np.ndarray, mri_sequence: str, mni: bool, follow_up: bool) -> None:
    """
    Register MRIs of multiple SICs to new space in parallel.

    :param sic_list: list of SICs
    :param mri_sequence: MRI sequence (T1, FLAIR, SWI)
    :param mni: to MNI space or not
    :param follow_up: True: use LIFE follow-up data
    :return: None
    """
    import concurrent.futures

    cprint(
        string=f"\nStart registering n={len(sic_list)} {mri_sequence.upper()}-MRIs "
        f"('{'follow-up' if follow_up else 'base-line'}') to {'MNI' if mni else 'T1-space'} ...\n",
        col="b",
    )

    start = datetime.now()

    if mni:
        # register MRI sequence to MNI space
        with concurrent.futures.ProcessPoolExecutor(50) as executor:  # use parallel processes
            _ = executor.map(
                register_mri_sequence_in_t1_space_to_mni,  # function
                list(sic_list),  # arg 0: _sic
                [mri_sequence] * len(sic_list),  # arg 1: mri_sequence
                [follow_up] * len(sic_list),  # arg 2: follow_up
                [False] * len(sic_list),
            )  # arg 3: verbose

    else:
        # register the raw MRI sequence to the T1 (FreeSurfer) space
        with concurrent.futures.ProcessPoolExecutor(50) as executor:  # use parallel processes
            _ = executor.map(
                register_native_to_t1_space,  # function
                list(sic_list),  # arg 0: _sic
                [mri_sequence] * len(sic_list),  # arg 1: mri_sequence
                [follow_up] * len(sic_list),  # arg 2: follow_up
                [True] * len(sic_list),  # arg 3: save_move
                [False] * len(sic_list),
            )  # arg 4: verbose

    cprint(
        string=f"\nFinalized registration of n={len(sic_list)} {mri_sequence.upper()}-MRIs to "
        f"{'MNI' if mni else 'T1-space'} in {chop_microseconds(datetime.now() - start)} [hh:min:sec]\n",
        col="b",
    )


# ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >> END
