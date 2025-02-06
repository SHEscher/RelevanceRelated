"""
Extract perivascular spaces (PVS) from T1 & FLAIR images.

This code uses elements from https://github.com/pboutinaud/SHIVA_PVS/blob/main/predict_one_file.py

We execute this script via the following bash script:

    RelevanceRelated/analysis/scripts/run_pvs.sh

Authors: Simon M. Hofmann & Frauke Beyer
Years: 2024
"""

# %% Import
import argparse
import gc
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import tensorflow as tf
from nilearn import plotting
from tqdm import tqdm

from relevancerelated.configs import paths
from relevancerelated.dataloader.LIFE.LIFE import convert_id, get_table_of_available_mris, load_sic_mri
from relevancerelated.dataloader.prune_image import prune_mri, reverse_pruning
from relevancerelated.dataloader.transformation import file_to_ref_orientation
from relevancerelated.utils import cprint, normalize

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
MULTI_MODAL_PVS_SEG: bool = True
NORM: bool = True  # take normed T1 image for PVS estimate
BM: bool = True  # take brain-masked image for PVS estimate (note if NORM is True, then images are brain-masked

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def load_pvs_segmentation_model_paths(multi_modal: bool) -> list[Path]:
    """
    List paths to PVS segmentation models by Boutinaud et al.

    There are two PVS segmentation models that have been trained:
    * one using T1 images,
    * and the other using both T1 and FLAIR images, i.e., a multi-modal segmentation model.

    :param multi_modal: False: T1 or True: T1+FLAIR images have been used to train the model.
    """
    model_dir = Path(paths.pvs.MODELS, f"T1{'-FLAIR' if multi_modal else ''}.PVS")

    # Check for model files
    if not list(model_dir.glob("*.h5")):
        model_url = f"https://cloud.efixia.com/sharing/{'Dg49eKSPR' if multi_modal else 'wknXOu07H'}"
        Path(paths.pvs.MODELS).mkdir(exist_ok=True, parents=True)
        cprint(
            string=f"Download the model from following URL: {model_url} and\n"
            f"unpack the model tar file (*.tgz) in {paths.pvs.MODELS}",
            col="y",
        )
        print(f"When done, models (*.h5) should reside in {model_dir}/")
        raise FileNotFoundError

    return list(model_dir.glob("*.h5"))


def compute_pvs_maps():
    """
    Compute PVS maps.

    Run this function from the command line using the flags defined below.s
    """
    # Set model name
    pvs_model_name = f"T1{'-FLAIR' if FLAGS.multimodal else ''}.PVS"

    # Create logger
    path_to_logs = Path(paths.pvs.LOGS.format("fu" if FLAGS.follow_up else "bl"))
    path_to_logfile = Path(path_to_logs, f"create_pvs_maps.{pvs_model_name}.log")
    logger = logging.getLogger(name="pvs_extraction.py.logger")
    path_to_logs.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path_to_logfile)
    handler.setFormatter(fmt=logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s"))
    logger.addHandler(hdlr=handler)
    logger.setLevel(level=logging.INFO)

    # Set tensorflow logger
    tf.get_logger().setLevel("ERROR")

    # All subjects
    seqs_cols = ["brain.finalsurfs.mgz"] + (["FLAIR_raw"] if FLAGS.multimodal else [])
    subjects = get_table_of_available_mris(follow_up=FLAGS.follow_up)[seqs_cols]
    # keep only subjects with available sequence(s)
    subjects = subjects[subjects.sum(axis=1) == len(seqs_cols)].index.to_list()

    # Extract PVS from T1 (& FLAIR) images
    path_to_pvs_maps = paths.pvs.MAPS.format("fu" if FLAGS.follow_up else "bl")
    for sic in tqdm(subjects, desc="Extract PVS maps", total=len(subjects), colour="#eb9196"):
        # Prepare SIC and path
        sic_upg = convert_id(sic)  # convert from old into the new SIC version
        path_to_sic_pvs = Path(path_to_pvs_maps, pvs_model_name, sic_upg, "pvs_segmentation.nii.gz")

        # Check if SIC has a PVS map already
        if path_to_sic_pvs.exists():
            continue

        # Check if there was an issue with the current SIC before
        if not FLAGS.rerun:
            with path_to_logfile.open("r") as logfile:
                if sic_upg in logfile.read():
                    continue

        try:
            # 1) Load T1 (& FLAIR) image per subject
            _, t1_mri = load_sic_mri(
                _sic=sic,  # this will convert to the new SIC format by itself
                mri_sequence="t1",
                follow_up=FLAGS.follow_up,
                regis=False,
                norm=FLAGS.norm,  # norm=True -> "brain.finalsurfs.mgz" (this is brain-masked)
                bm=FLAGS.brainmask,  # norm=False -> "brainmask.mgz" if bm else "T1.mgz" (whole head)
                compressed=False,
                as_numpy=False,
            )
            _, flair_mri = (
                load_sic_mri(
                    _sic=sic,
                    mri_sequence="flair",
                    follow_up=FLAGS.follow_up,
                    regis=False,
                    norm=FLAGS.norm,
                    bm=FLAGS.brainmask,
                    compressed=False,
                    as_numpy=False,
                )
                if FLAGS.multimodal
                else (None, None)
            )

            # 2) Preprocess image
            # Bring image to RAS/LAS (nibabel) space
            t1_mri = file_to_ref_orientation(image_file=t1_mri, reference_space="RAS")
            if FLAGS.multimodal:
                flair_mri = file_to_ref_orientation(image_file=flair_mri, reference_space="RAS")
            if nib.orientations.aff2axcodes(aff=t1_mri.affine) != ("R", "A", "S"):
                msg = "Image does not have the 'RAS' orientation!"
                logger.error(msg=msg)

            # Crop images to (160 x 214 x 176) 1mm³
            t1_mri_pruned = prune_mri(x3d=t1_mri.get_fdata(), make_cube=False, max_axis=(160, 214, 176), padding=0)
            flair_mri_pruned = (
                prune_mri(x3d=flair_mri.get_fdata(), make_cube=False, max_axis=(160, 214, 176), padding=0)
                if FLAGS.multimodal
                else None
            )

            # Set max to 99th percentile of the intensities & rescale to (0, 1)
            t1_mri_pruned = np.minimum(t1_mri_pruned, np.percentile(a=t1_mri_pruned, q=99))
            t1_mri_pruned_normed = normalize(array=t1_mri_pruned, lower_bound=0.0, upper_bound=1.0)
            # the resulting max value should be shared across MRIs ideally, but we stick to Boutinaud et al.
            if FLAGS.multimodal:
                flair_mri_pruned = np.minimum(flair_mri_pruned, np.percentile(a=flair_mri_pruned, q=99))
            flair_mri_pruned_normed = (
                normalize(array=flair_mri_pruned, lower_bound=0.0, upper_bound=1.0) if FLAGS.multimodal else None
            )

            # Add channel dimension (160 x 214 x 176, 1)
            mri_pruned_normed_final = t1_mri_pruned_normed[..., np.newaxis]

            if FLAGS.multimodal:
                # Attach FLAIR as last channel: (160 x 214 x 176 x 2) voxels
                mri_pruned_normed_final = np.concatenate(
                    [mri_pruned_normed_final, flair_mri_pruned_normed[..., np.newaxis]], axis=-1
                )

            # Add batch dimension
            mri_pruned_normed_final = mri_pruned_normed_final[np.newaxis, ...]

            # 3) Extract PVS from T1 (& FLAIR) images using the U-Net of Boutinaud et al.
            estimator_model_files = load_pvs_segmentation_model_paths(multi_modal=FLAGS.multimodal)
            segmentation_estimates = []
            # We iterate over several models (ensemble) and then average the segmentation results
            for estimator_file in tqdm(
                iterable=sorted(estimator_model_files),
                desc=f"Estimate PVS segmentation for '{sic_upg}'",
                total=len(estimator_model_files),
                leave=False,
                colour="#32bd68",
            ):
                # TODO: Solve issue of memory overload in GPUs (the following lines are not sufficient) # noqa: FIX002
                tf.keras.backend.clear_session()
                gc.collect()

                # Load segmentation model
                model = tf.keras.models.load_model(estimator_file, compile=False, custom_objects={"tf": tf})

                # Estimate PVS map per model
                estimate = model.predict(mri_pruned_normed_final, batch_size=1, verbose=False)
                segmentation_estimates.append(estimate)

            # Average all segmentation estimates
            segmentation_estimates = np.mean(segmentation_estimates, axis=0)

            # Reverse pruning (cropping) of segmentation to original T1 image shape
            segmentation_estimates_org_shape = reverse_pruning(
                original_mri=t1_mri.get_fdata(),
                pruned_mri=t1_mri_pruned,
                pruned_stats_map=segmentation_estimates.squeeze(),
            )

            # Save PVS segmentation map as NIfTI
            segmentation_estimates_nii = nib.Nifti1Image(segmentation_estimates_org_shape, affine=t1_mri.affine)
            path_to_sic_pvs.parent.mkdir(parents=True, exist_ok=True)
            segmentation_estimates_nii.to_filename(filename=path_to_sic_pvs)

            # Log
            info_msg = (
                f"Successfully segmented PVS using '{pvs_model_name}' for '{sic_upg}' and saved corresponding "
                f"map to '{path_to_sic_pvs}'"
            )
            logger.info(msg=info_msg)

            if FLAGS.plot:
                plotting.plot_stat_map(
                    stat_map_img=segmentation_estimates_nii, bg_img=t1_mri, draw_cross=False, cmap="black_pink"
                )  # "black_red"
                plotting.show()
                input("\nPress Enter to continue ...")

        except Exception as e:
            issue_msg = f"Issue during PVS extraction with SIC '{sic_upg}'"
            # TODO: remove the following after GPU issue is solved  # noqa: FIX002
            #  one way would be to start the estimation process in a subprocess
            if "memory" in str(e) or "Graph execution error" in str(e):  # repr(e) == "ResourceExhaustedError()"
                # Currently, this handles our GPU memory overload issue, by exiting the process.
                # We then restart the script which will continue with the unprocessed data.
                cprint("\n" + issue_msg, col="r")
                raise e  # noqa: TRY201
            # All other exceptions we write in our log file and handle them later.
            logger.exception(issue_msg)  # TODO: keep this only as long GPU issue is not solved # noqa: FIX002


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Check if run on MPI CBS servers
    if not Path(paths.data.life.ROOT).exists():
        msg = "PVS maps can only be extracted on MPI CBS servers."
        raise ConnectionError(msg)

    # Add arg parser
    parser = argparse.ArgumentParser("PVS extraction")
    parser.add_argument(
        "-m",
        "--multimodal",
        action=argparse.BooleanOptionalAction,
        default=MULTI_MODAL_PVS_SEG,
        help="True: Use multi-modal (T1+FLAIR) data for PVS segmentation, else T1 data only.",
    )
    parser.add_argument(
        "-f",
        "--follow_up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extract PVS segmentation maps from LIFE follow-up data.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot PVS estimates",
    )
    parser.add_argument(
        "-n",
        "--norm",
        action=argparse.BooleanOptionalAction,
        default=NORM,
        help="Take normed MRIs ('brain.finalsurfs.mgz') for PVS estimate (these are brain-masked).",
    )
    parser.add_argument(
        "-b",
        "--brainmask",
        action=argparse.BooleanOptionalAction,
        default=BM,
        help="Take brain-masked image for PVS estimate (has only an effect if norm=False).",
    )

    parser.add_argument(
        "-r",
        "--rerun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rerun on SICs that have previously been identified having an issue with their PVS estimate.",
    )

    # Parse arguments
    FLAGS, unparsed = parser.parse_known_args()

    # %% Run main
    compute_pvs_maps()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
